// Copyright 2024 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// L3 offline replay benchmark. A single executable drives BOTH NDT engines
// on identical inputs: the C++ `MultiGridNormalDistributionsTransform` and the Rust port over the C
// ABI. The target map + kd-tree is built ONCE per engine; only the repeated `align` loop is timed
// (steady_clock), so this measures the align kernel, not map construction. Both engines see the same
// synthetic geometry from the same buffers, so the comparison is apples-to-apples by construction.
//
// Output: a JSON file with per-align latency samples (ms) + iteration_num + run metadata, consumed by
// bench/gen_report.py to render a self-contained HTML report.
//
// Usage: ndt_bench_replay [iters=200] [warmup=20] [out.json=ndt_bench.json] [interval=0.2]
//   `interval` sets the synthetic point spacing → point count (default 0.2 m ⇒ ~30,603 pts).
//
// WCET fixture mode (plan/ndt_wcet.md, M4):
//   ndt_bench_replay --fixture out.json fixture1.ndtfix [fixture2.ndtfix ...]
// Replays frozen "NDTFIX01" fixtures (bench/fixtures/*.ndtfix, written by the Rust
// `wcet_fixtures` / `wcet_search` examples) through BOTH engines on identical buffers and asserts
// `iteration_num` equality per fixture (bit-exactness ⇒ equal work ⇒ the timing comparison is
// algorithm-fair). Env: WCET_ITERS (default 100), WCET_WARMUP (default 10).
//
// Real-drive capture mode (operational envelope):
//   ndt_bench_replay --capture out.json <NDT_CAPTURE_DIR>
// Replays a capture directory (params.bin + tiles/ + frames/, written by the node's
// NDT_CAPTURE_DIR hook) through BOTH engines: frames are grouped into epochs by their active
// tile-id set; per epoch both maps are rebuilt with tiles added in the captured (sorted) id
// order — C++ uses zero-padded numeric names so its std::map string order matches the Rust
// engine's numeric id order. Per frame: WCET_REPEATS timed aligns per engine (default 5, median
// reported) + an iteration-equality check. Rust-side counters come from
// `wcet_frame --capture` (separate pass); merge with bench/wcet_realdata.py.

#include <autoware/ndt_scan_matcher/ndt_omp/multigrid_ndt_omp.h>

#include "autoware_ndt_scan_matcher_rs.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace
{
using Ndt = pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>;
using Clock = std::chrono::steady_clock;

// Three orthogonal planes over [0, length] at `interval` spacing — the standard-sequence fixture
// geometry. Fills both a PCL cloud (C++ engine) and the flat xyz buffer (Rust FFI) from identical
// points.
void make_half_cubic(
  float length, float interval, pcl::PointCloud<pcl::PointXYZ> & cloud, std::vector<float> & flat)
{
  const int n = static_cast<int>(length / interval) + 1;
  const auto add = [&](float x, float y, float z) {
    pcl::PointXYZ p;
    p.x = x;
    p.y = y;
    p.z = z;
    cloud.push_back(p);
    flat.push_back(x);
    flat.push_back(y);
    flat.push_back(z);
  };
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const float u = interval * static_cast<float>(j);
      const float v = interval * static_cast<float>(i);
      add(u, v, 0.0F);
      add(0.0F, u, v);
      add(u, 0.0F, v);
    }
  }
  cloud.is_dense = true;
}

double ms_since(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void write_samples(FILE * f, const std::vector<double> & xs)
{
  std::fputc('[', f);
  for (size_t i = 0; i < xs.size(); ++i) {
    std::fprintf(f, "%s%.6f", (i == 0 ? "" : ","), xs[i]);
  }
  std::fputc(']', f);
}

// ---------------------------------------------------------------------------
// WCET fixture mode (plan/ndt_wcet.md, M4)
// ---------------------------------------------------------------------------

// One frozen align input, mirroring the Rust `autoware_ndt_rs::fixture` "NDTFIX01" layout
// (little-endian; map stored as tiles — one setInputTarget id each — because leaf overlap across
// tiles is what drives the per-point neighbor count K).
struct WcetFixture
{
  std::vector<std::vector<float>> tiles;  // flat xyz per tile
  std::vector<float> source;              // flat xyz
  std::array<float, 16> guess16;          // row-major 4x4
  double trans_epsilon;
  double step_size;
  double resolution;
  double outlier_ratio;
  int32_t max_iterations;
};

bool read_exact(FILE * f, void * dst, size_t n)
{
  return std::fread(dst, 1, n, f) == n;
}

// Loads a "NDTFIX01" fixture; prints its own error and returns false on any malformed input.
// Assumes a little-endian host (x86-64 / AArch64 — everything this bench targets).
bool load_fixture(const char * path, WcetFixture & fx)
{
  FILE * f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "wcet: cannot open %s\n", path);
    return false;
  }
  bool ok = false;
  uint64_t n_tiles = 0;
  uint64_t n_source = 0;
  int32_t reserved = 0;
  char magic[8];
  // Sanity caps mirror the Rust reader (fixture.rs): 4096 tiles / 50M points.
  constexpr uint64_t kMaxTiles = 4096;
  constexpr uint64_t kMaxPoints = 50'000'000;
  do {
    if (!read_exact(f, magic, 8) || std::memcmp(magic, "NDTFIX01", 8) != 0) break;
    if (!read_exact(f, &n_tiles, 8) || !read_exact(f, &n_source, 8)) break;
    if (n_tiles > kMaxTiles || n_source > kMaxPoints) break;
    if (!read_exact(f, fx.guess16.data(), 16 * sizeof(float))) break;
    if (!read_exact(f, &fx.trans_epsilon, 8) || !read_exact(f, &fx.step_size, 8)) break;
    if (!read_exact(f, &fx.resolution, 8) || !read_exact(f, &fx.outlier_ratio, 8)) break;
    if (!read_exact(f, &fx.max_iterations, 4) || !read_exact(f, &reserved, 4)) break;
    fx.tiles.clear();
    fx.tiles.reserve(n_tiles);
    bool tiles_ok = true;
    for (uint64_t t = 0; t < n_tiles; ++t) {
      uint64_t n_pts = 0;
      if (!read_exact(f, &n_pts, 8) || n_pts > kMaxPoints) {
        tiles_ok = false;
        break;
      }
      std::vector<float> flat(static_cast<size_t>(n_pts) * 3);
      if (!flat.empty() && !read_exact(f, flat.data(), flat.size() * sizeof(float))) {
        tiles_ok = false;
        break;
      }
      fx.tiles.push_back(std::move(flat));
    }
    if (!tiles_ok) break;
    fx.source.resize(static_cast<size_t>(n_source) * 3);
    if (!fx.source.empty() && !read_exact(f, fx.source.data(), fx.source.size() * sizeof(float)))
      break;
    ok = true;
  } while (false);
  std::fclose(f);
  if (!ok) {
    std::fprintf(stderr, "wcet: malformed fixture %s\n", path);
  }
  return ok;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr flat_to_cloud(const std::vector<float> & flat)
{
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->reserve(flat.size() / 3);
  for (size_t i = 0; i + 2 < flat.size(); i += 3) {
    pcl::PointXYZ p;
    p.x = flat[i];
    p.y = flat[i + 1];
    p.z = flat[i + 2];
    cloud->push_back(p);
  }
  cloud->is_dense = true;
  return cloud;
}

// Optional per-align allocation counting: if bench/alloc_count.so is LD_PRELOADed, its
// reset/get hooks are visible via dlsym and we can count C++-engine mallocs per timed window.
// Without the preload both pointers stay null and counting is skipped (allocs reported as -1).
using AllocResetFn = void (*)();
using AllocGetFn = unsigned long long (*)();

struct AllocHooks
{
  AllocResetFn reset = nullptr;
  AllocGetFn get = nullptr;
};

AllocHooks find_alloc_hooks()
{
  AllocHooks h;
  h.reset = reinterpret_cast<AllocResetFn>(dlsym(RTLD_DEFAULT, "alloc_count_reset"));
  h.get = reinterpret_cast<AllocGetFn>(dlsym(RTLD_DEFAULT, "alloc_count_get"));
  return h;
}

std::string fixture_stem(const std::string & path)
{
  const size_t slash = path.find_last_of('/');
  std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
  const size_t dot = base.find_last_of('.');
  return (dot == std::string::npos) ? base : base.substr(0, dot);
}

// Replays each frozen fixture through BOTH engines on identical buffers; asserts iteration_num
// equality (equal work — the precondition for a fair timing comparison); emits one JSON with
// per-fixture sample arrays. Returns the process exit code.
int run_fixture_mode(const std::string & out_path, const std::vector<std::string> & paths)
{
  const int iters = std::getenv("WCET_ITERS") ? std::atoi(std::getenv("WCET_ITERS")) : 100;
  const int warmup = std::getenv("WCET_WARMUP") ? std::atoi(std::getenv("WCET_WARMUP")) : 10;
  const AllocHooks hooks = find_alloc_hooks();

  FILE * f = std::fopen(out_path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "wcet: cannot open %s for writing\n", out_path.c_str());
    return 1;
  }
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"benchmark\": \"WCET fixture replay (frozen adversarial inputs)\",\n");
  std::fprintf(f, "  \"meta\": {\n");
  std::fprintf(f, "    \"iters\": %d,\n", iters);
  std::fprintf(f, "    \"warmup\": %d,\n", warmup);
  std::fprintf(f, "    \"num_threads\": 1,\n");
  std::fprintf(f, "    \"clock\": \"steady_clock\",\n");
  std::fprintf(f, "    \"unit\": \"ms\",\n");
  std::fprintf(
    f, "    \"alloc_counting\": %s,\n", (hooks.reset && hooks.get) ? "true" : "false");
  std::fprintf(f, "    \"note\": \"align loop only; map+kdtree built once per engine per fixture\"\n");
  std::fprintf(f, "  },\n");
  std::fprintf(f, "  \"fixtures\": {\n");

  int rc = 0;
  bool first = true;
  for (const auto & path : paths) {
    WcetFixture fx;
    if (!load_fixture(path.c_str(), fx)) {
      rc = 1;
      continue;
    }
    // The C++ engine hardcodes outlier_ratio_ = 0.55 (no setter); a fixture with any other value
    // cannot be replayed comparably.
    if (fx.outlier_ratio != 0.55) {
      std::fprintf(stderr, "wcet: %s: outlier_ratio %.3f != 0.55 (C++ hardcoded)\n", path.c_str(),
        fx.outlier_ratio);
      rc = 1;
      continue;
    }
    size_t n_map = 0;
    for (const auto & t : fx.tiles) n_map += t.size() / 3;
    const size_t n_src = fx.source.size() / 3;

    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        guess(r, c) = fx.guess16[(r * 4) + c];
      }
    }

    // ---- C++ engine ----
    pclomp::NdtParams params{};
    params.trans_epsilon = fx.trans_epsilon;
    params.step_size = fx.step_size;
    params.resolution = static_cast<float>(fx.resolution);
    params.max_iterations = fx.max_iterations;
    params.search_method = pclomp::KDTREE;
    params.num_threads = 1;  // serial baseline (matches the fixture contract)
    params.regularization_scale_factor = 0.0F;
    params.use_line_search = false;

    Ndt ndt;
    ndt.setParams(params);
    for (size_t t = 0; t < fx.tiles.size(); ++t) {
      ndt.addTarget(flat_to_cloud(fx.tiles[t]), std::to_string(t));
    }
    ndt.createVoxelKdtree();
    auto source = flat_to_cloud(fx.source);
    auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for (int i = 0; i < warmup; ++i) {
      ndt.align(*aligned, guess, source);
    }
    long long cpp_allocs = -1;
    if (hooks.reset && hooks.get) hooks.reset();
    std::vector<double> cpp_ms;
    cpp_ms.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
      const auto t0 = Clock::now();
      ndt.align(*aligned, guess, source);
      cpp_ms.push_back(ms_since(t0));
    }
    if (hooks.reset && hooks.get) {
      cpp_allocs = static_cast<long long>(hooks.get() / static_cast<unsigned long long>(iters));
    }
    const int cpp_iter = ndt.getResult().iteration_num;

    // ---- Rust engine (over the C ABI) ----
    struct AwNdtEngine * eng =
      autoware_ndt_scan_matcher_rs_ndt_engine_new(fx.resolution, 6, 0.01);
    autoware_ndt_scan_matcher_rs_ndt_engine_set_params(
      eng, fx.trans_epsilon, fx.step_size, fx.resolution, fx.max_iterations, fx.outlier_ratio, 1);
    for (size_t t = 0; t < fx.tiles.size(); ++t) {
      autoware_ndt_scan_matcher_rs_ndt_engine_add_target(
        eng, fx.tiles[t].data(), fx.tiles[t].size() / 3, t);
    }
    autoware_ndt_scan_matcher_rs_ndt_engine_create_kdtree(eng);
    for (int i = 0; i < warmup; ++i) {
      autoware_ndt_scan_matcher_rs_ndt_engine_align(eng, fx.guess16.data(), fx.source.data(), n_src);
    }
    long long rs_allocs = -1;
    if (hooks.reset && hooks.get) hooks.reset();
    std::vector<double> rs_ms;
    rs_ms.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
      const auto t0 = Clock::now();
      autoware_ndt_scan_matcher_rs_ndt_engine_align(eng, fx.guess16.data(), fx.source.data(), n_src);
      rs_ms.push_back(ms_since(t0));
    }
    if (hooks.reset && hooks.get) {
      rs_allocs = static_cast<long long>(hooks.get() / static_cast<unsigned long long>(iters));
    }
    int32_t rs_iter = 0;
    AwNdtAlignOutput rout{};
    rout.iteration_num = &rs_iter;
    autoware_ndt_scan_matcher_rs_ndt_engine_get_result(eng, &rout);
    autoware_ndt_scan_matcher_rs_ndt_engine_free(eng);

    // ---- equal-work assert ----
    const bool iter_match = (cpp_iter == rs_iter);
    if (!iter_match) {
      std::fprintf(stderr, "wcet: ITERATION MISMATCH on %s: cpp=%d rust=%d\n", path.c_str(),
        cpp_iter, rs_iter);
      rc = 1;
    }

    const std::string name = fixture_stem(path);
    std::fprintf(f, "%s    \"%s\": {\n", (first ? "" : ",\n"), name.c_str());
    first = false;
    std::fprintf(f, "      \"n_map\": %zu,\n", n_map);
    std::fprintf(f, "      \"n_tiles\": %zu,\n", fx.tiles.size());
    std::fprintf(f, "      \"n_source\": %zu,\n", n_src);
    std::fprintf(f, "      \"max_iterations\": %d,\n", fx.max_iterations);
    std::fprintf(f, "      \"iter_match\": %s,\n", iter_match ? "true" : "false");
    std::fprintf(
      f, "      \"cpp\": { \"iteration_num\": %d, \"allocs_per_align\": %lld, \"samples_ms\": ",
      cpp_iter, cpp_allocs);
    write_samples(f, cpp_ms);
    std::fprintf(f, " },\n");
    std::fprintf(
      f, "      \"rust\": { \"iteration_num\": %d, \"allocs_per_align\": %lld, \"samples_ms\": ",
      rs_iter, rs_allocs);
    write_samples(f, rs_ms);
    std::fprintf(f, " }\n    }");

    std::fprintf(stderr, "wcet: %-18s map=%zu src=%zu iter cpp=%d rust=%d %s\n", name.c_str(),
      n_map, n_src, cpp_iter, rs_iter, iter_match ? "OK" : "MISMATCH");
  }
  std::fprintf(f, "\n  }\n}\n");
  std::fclose(f);
  std::fprintf(stderr, "wcet: wrote %s\n", out_path.c_str());
  return rc;
}

// ---------------------------------------------------------------------------
// Real-drive capture mode (NDT_CAPTURE_DIR sidecar format)
// ---------------------------------------------------------------------------

struct CaptureParams
{
  double trans_epsilon, step_size, resolution, outlier_ratio;
  int32_t max_iterations;
};

bool read_capture_params(const std::string & dir, CaptureParams & p)
{
  FILE * f = std::fopen((dir + "/params.bin").c_str(), "rb");
  if (f == nullptr) return false;
  int32_t reserved = 0;
  const bool ok = read_exact(f, &p.trans_epsilon, 8) && read_exact(f, &p.step_size, 8) &&
                  read_exact(f, &p.resolution, 8) && read_exact(f, &p.outlier_ratio, 8) &&
                  read_exact(f, &p.max_iterations, 4) && read_exact(f, &reserved, 4);
  std::fclose(f);
  return ok;
}

std::string hex_of(const std::string & id)
{
  static const char * kHex = "0123456789abcdef";
  std::string s;
  s.reserve(id.size() * 2);
  for (unsigned char b : id) {
    s.push_back(kHex[b >> 4]);
    s.push_back(kHex[b & 0x0f]);
  }
  return s;
}

struct CaptureFrame
{
  std::array<float, 16> guess16;
  std::vector<std::string> ids;
  std::vector<float> source;  // flat xyz
};

bool read_capture_frame(const std::string & path, CaptureFrame & fr)
{
  FILE * f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  bool ok = false;
  do {
    if (!read_exact(f, fr.guess16.data(), 16 * sizeof(float))) break;
    uint64_t n_ids = 0;
    if (!read_exact(f, &n_ids, 8) || n_ids > 4096) break;
    fr.ids.clear();
    bool ids_ok = true;
    for (uint64_t i = 0; i < n_ids; ++i) {
      uint64_t len = 0;
      if (!read_exact(f, &len, 8) || len > 4096) {
        ids_ok = false;
        break;
      }
      std::string id(static_cast<size_t>(len), '\0');
      if (len > 0 && !read_exact(f, id.data(), static_cast<size_t>(len))) {
        ids_ok = false;
        break;
      }
      fr.ids.push_back(std::move(id));
    }
    if (!ids_ok) break;
    uint64_t n_src = 0;
    if (!read_exact(f, &n_src, 8) || n_src > 50'000'000ULL) break;
    fr.source.resize(static_cast<size_t>(n_src) * 3);
    if (!fr.source.empty() &&
        !read_exact(f, fr.source.data(), fr.source.size() * sizeof(float)))
      break;
    ok = true;
  } while (false);
  std::fclose(f);
  return ok;
}

// Replays a capture directory through BOTH engines (epoch-rebuilt maps, per-frame equal-work
// check). Returns the process exit code.
int run_capture_mode(const std::string & out_path, const std::string & dir)
{
  const int repeats = std::getenv("WCET_REPEATS") ? std::atoi(std::getenv("WCET_REPEATS")) : 5;
  // Chain mode (WCET_CHAIN=1): feed BOTH engines the previous frame's Rust result pose as the
  // guess (frame 0: WCET_CHAIN_SEED="x,y,z,qx,qy,qz,qw") -- a deterministic offline
  // re-localization replay that decouples the envelope from the live EKF loop.
  const bool chain = std::getenv("WCET_CHAIN") != nullptr;
  Eigen::Matrix4f chain_guess = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f chain_prev = Eigen::Matrix4f::Identity();
  bool have_prev = false;
  if (chain) {
    const char * seed = std::getenv("WCET_CHAIN_SEED");
    if (seed != nullptr) {
      double v[7] = {0, 0, 0, 0, 0, 0, 1};
      std::sscanf(seed, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
        &v[6]);
      const Eigen::Quaternionf q(static_cast<float>(v[6]), static_cast<float>(v[3]),
        static_cast<float>(v[4]), static_cast<float>(v[5]));
      chain_guess.topLeftCorner<3, 3>() = q.normalized().toRotationMatrix();
      chain_guess(0, 3) = static_cast<float>(v[0]);
      chain_guess(1, 3) = static_cast<float>(v[1]);
      chain_guess(2, 3) = static_cast<float>(v[2]);
    }
  }
  CaptureParams cp{};
  if (!read_capture_params(dir, cp)) {
    std::fprintf(stderr, "capture: cannot read %s/params.bin\n", dir.c_str());
    return 1;
  }
  std::vector<std::string> frame_paths;
  for (const auto & e : std::filesystem::directory_iterator(dir + "/frames")) {
    if (e.path().extension() == ".bin") frame_paths.push_back(e.path().string());
  }
  std::sort(frame_paths.begin(), frame_paths.end());
  std::fprintf(stderr, "capture: %zu frames, %d repeats/frame\n", frame_paths.size(), repeats);

  // Tile cache: hex id -> (pcl cloud, flat xyz).
  std::map<std::string, std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<float>>>
    tile_cache;
  const auto load_tile = [&](const std::string & id) -> bool {
    const std::string hex = hex_of(id);
    if (tile_cache.count(hex) != 0) return true;
    FILE * f = std::fopen((dir + "/tiles/" + hex + ".bin").c_str(), "rb");
    if (f == nullptr) return false;
    uint64_t n = 0;
    bool ok = read_exact(f, &n, 8) && n <= 50'000'000ULL;
    std::vector<float> flat;
    if (ok) {
      flat.resize(static_cast<size_t>(n) * 3);
      ok = flat.empty() || read_exact(f, flat.data(), flat.size() * sizeof(float));
    }
    std::fclose(f);
    if (!ok) return false;
    tile_cache.emplace(hex, std::make_pair(flat_to_cloud(flat), std::move(flat)));
    return true;
  };

  pclomp::NdtParams params{};
  params.trans_epsilon = cp.trans_epsilon;
  params.step_size = cp.step_size;
  params.resolution = static_cast<float>(cp.resolution);
  params.max_iterations = cp.max_iterations;
  params.search_method = pclomp::KDTREE;
  params.num_threads = 1;
  params.regularization_scale_factor = 0.0F;
  params.use_line_search = false;

  FILE * f = std::fopen(out_path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "capture: cannot open %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(f, "{\n  \"frames\": [\n");

  int rc = 0;
  bool first = true;
  size_t epochs = 0;
  size_t mismatches = 0;
  std::vector<std::string> cur_ids;
  std::unique_ptr<Ndt> ndt;
  struct AwNdtEngine * eng = nullptr;
  auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  for (size_t fi = 0; fi < frame_paths.size(); ++fi) {
    CaptureFrame fr;
    if (!read_capture_frame(frame_paths[fi], fr)) {
      std::fprintf(stderr, "capture: malformed frame %s\n", frame_paths[fi].c_str());
      rc = 1;
      continue;
    }
    if (fr.ids != cur_ids) {
      // New epoch: rebuild both maps, tiles added in the captured (sorted) id order with
      // zero-padded numeric names (C++ std::map string order == Rust numeric id order).
      ndt = std::make_unique<Ndt>();
      ndt->setParams(params);
      if (eng != nullptr) autoware_ndt_scan_matcher_rs_ndt_engine_free(eng);
      eng = autoware_ndt_scan_matcher_rs_ndt_engine_new(cp.resolution, 6, 0.01);
      autoware_ndt_scan_matcher_rs_ndt_engine_set_params(
        eng, cp.trans_epsilon, cp.step_size, cp.resolution, cp.max_iterations, cp.outlier_ratio,
        1);
      bool tiles_ok = true;
      for (size_t i = 0; i < fr.ids.size(); ++i) {
        if (!load_tile(fr.ids[i])) {
          std::fprintf(stderr, "capture: missing tile for frame %zu\n", fi);
          tiles_ok = false;
          break;
        }
        const auto & entry = tile_cache.at(hex_of(fr.ids[i]));
        char name[16];
        std::snprintf(name, sizeof(name), "%06zu", i);
        ndt->addTarget(entry.first, name);
        autoware_ndt_scan_matcher_rs_ndt_engine_add_target(
          eng, entry.second.data(), entry.second.size() / 3, static_cast<uint64_t>(i));
      }
      if (!tiles_ok) {
        rc = 1;
        cur_ids.clear();
        continue;
      }
      ndt->createVoxelKdtree();
      autoware_ndt_scan_matcher_rs_ndt_engine_create_kdtree(eng);
      cur_ids = fr.ids;
      ++epochs;
    }

    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
    if (chain) {
      guess = chain_guess;
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) fr.guess16[(r * 4) + c] = guess(r, c);
    } else {
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) guess(r, c) = fr.guess16[(r * 4) + c];
    }
    auto source = flat_to_cloud(fr.source);
    const size_t n_src = fr.source.size() / 3;

    // Median-of-repeats per engine (interleaving avoided: all C++ then all Rust per frame).
    std::vector<double> cpp_ms_v, rs_ms_v;
    cpp_ms_v.reserve(static_cast<size_t>(repeats));
    rs_ms_v.reserve(static_cast<size_t>(repeats));
    for (int k = 0; k < repeats; ++k) {
      const auto t0 = Clock::now();
      ndt->align(*aligned, guess, source);
      cpp_ms_v.push_back(ms_since(t0));
    }
    for (int k = 0; k < repeats; ++k) {
      const auto t0 = Clock::now();
      autoware_ndt_scan_matcher_rs_ndt_engine_align(eng, fr.guess16.data(), fr.source.data(),
        n_src);
      rs_ms_v.push_back(ms_since(t0));
    }
    std::sort(cpp_ms_v.begin(), cpp_ms_v.end());
    std::sort(rs_ms_v.begin(), rs_ms_v.end());
    const int cpp_iter = ndt->getResult().iteration_num;
    int32_t rs_iter = 0;
    std::array<float, 16> rs_pose{};
    AwNdtAlignOutput rout{};
    rout.iteration_num = &rs_iter;
    rout.pose = rs_pose.data();
    autoware_ndt_scan_matcher_rs_ndt_engine_get_result(eng, &rout);
    if (chain) {
      Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) pose(r, c) = rs_pose[(r * 4) + c];
      // Constant-velocity extrapolation, mirroring the Rust replayer.
      chain_guess = have_prev ? Eigen::Matrix4f(pose * (chain_prev.inverse() * pose)) : pose;
      chain_prev = pose;
      have_prev = true;
    }
    const bool match = (cpp_iter == rs_iter);
    if (!match) ++mismatches;

    std::fprintf(f,
      "%s    { \"seq\": %zu, \"n_source\": %zu, \"n_tiles\": %zu, \"cpp_ms\": %.4f, "
      "\"rust_ms\": %.4f, \"iter_cpp\": %d, \"iter_rust\": %d, \"match\": %s }",
      (first ? "" : ",\n"), fi, n_src, fr.ids.size(), cpp_ms_v[cpp_ms_v.size() / 2],
      rs_ms_v[rs_ms_v.size() / 2], cpp_iter, rs_iter, match ? "true" : "false");
    first = false;
  }
  if (eng != nullptr) autoware_ndt_scan_matcher_rs_ndt_engine_free(eng);
  std::fprintf(f, "\n  ]\n}\n");
  std::fclose(f);
  std::fprintf(stderr, "capture: %zu epochs, %zu iteration mismatches -> %s\n", epochs,
    mismatches, out_path.c_str());
  return (rc != 0 || mismatches > 0) ? 1 : 0;
}
}  // namespace

int main(int argc, char ** argv)
{
  // WCET fixture mode: ndt_bench_replay --fixture out.json fix1.ndtfix [fix2.ndtfix ...]
  if (argc >= 4 && std::strcmp(argv[1], "--fixture") == 0) {
    std::vector<std::string> paths(argv + 3, argv + argc);
    return run_fixture_mode(argv[2], paths);
  }
  // Real-drive capture mode: ndt_bench_replay --capture out.json <NDT_CAPTURE_DIR>
  if (argc == 4 && std::strcmp(argv[1], "--capture") == 0) {
    return run_capture_mode(argv[2], argv[3]);
  }

  const int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
  const int warmup = (argc > 2) ? std::atoi(argv[2]) : 20;
  const std::string out_path = (argc > 3) ? argv[3] : "ndt_bench.json";
  const float interval = (argc > 4) ? static_cast<float>(std::atof(argv[4])) : 0.2F;
  const float length = 20.0F;

  // Target map + source = target translated by a known offset (identity guess should recover it).
  auto target = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  std::vector<float> target_flat;
  make_half_cubic(length, interval, *target, target_flat);

  const std::array<float, 3> offset = {0.2F, -0.15F, 0.1F};
  auto source = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  std::vector<float> source_flat;
  source_flat.reserve(target_flat.size());
  for (const auto & p : *target) {
    pcl::PointXYZ q;
    q.x = p.x + offset[0];
    q.y = p.y + offset[1];
    q.z = p.z + offset[2];
    source->push_back(q);
    source_flat.push_back(q.x);
    source_flat.push_back(q.y);
    source_flat.push_back(q.z);
  }
  const size_t n_pts = target->size();

  pclomp::NdtParams params{};
  params.trans_epsilon = 0.01;
  params.step_size = 0.1;
  params.resolution = 2.0F;
  params.max_iterations = 30;
  params.search_method = pclomp::KDTREE;
  params.num_threads = 1;  // serial-vs-serial baseline
  params.regularization_scale_factor = 0.0F;
  params.use_line_search = false;

  const Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
  std::array<float, 16> guess16{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      guess16[(r * 4) + c] = guess(r, c);
    }
  }

  // ---- C++ engine: build map once, time the align loop ----
  Ndt ndt;
  ndt.setParams(params);
  ndt.setInputTarget(target);
  auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  for (int i = 0; i < warmup; ++i) {
    ndt.align(*aligned, guess, source);
  }
  std::vector<double> cpp_ms;
  cpp_ms.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = Clock::now();
    ndt.align(*aligned, guess, source);
    cpp_ms.push_back(ms_since(t0));
  }
  const int cpp_iter = ndt.getResult().iteration_num;

  // ---- Rust engine (over the C ABI): build map once, time the align loop ----
  struct AwNdtEngine * eng = autoware_ndt_scan_matcher_rs_ndt_engine_new(2.0, 6, 0.01);
  autoware_ndt_scan_matcher_rs_ndt_engine_set_params(eng, 0.01, 0.1, 2.0, 30, 0.55, 1);
  autoware_ndt_scan_matcher_rs_ndt_engine_add_target(eng, target_flat.data(), n_pts, 0);
  autoware_ndt_scan_matcher_rs_ndt_engine_create_kdtree(eng);
  for (int i = 0; i < warmup; ++i) {
    autoware_ndt_scan_matcher_rs_ndt_engine_align(eng, guess16.data(), source_flat.data(), n_pts);
  }
  std::vector<double> rs_ms;
  rs_ms.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = Clock::now();
    autoware_ndt_scan_matcher_rs_ndt_engine_align(eng, guess16.data(), source_flat.data(), n_pts);
    rs_ms.push_back(ms_since(t0));
  }
  int32_t rs_iter = 0;
  AwNdtAlignOutput rout{};
  rout.iteration_num = &rs_iter;
  autoware_ndt_scan_matcher_rs_ndt_engine_get_result(eng, &rout);
  autoware_ndt_scan_matcher_rs_ndt_engine_free(eng);

  // ---- emit JSON ----
  FILE * f = std::fopen(out_path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "ndt_bench_replay: cannot open %s for writing\n", out_path.c_str());
    return 1;
  }
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"benchmark\": \"L3 offline align replay (synthetic half-cubic)\",\n");
  std::fprintf(f, "  \"meta\": {\n");
  std::fprintf(f, "    \"n_points\": %zu,\n", n_pts);
  std::fprintf(f, "    \"iters\": %d,\n", iters);
  std::fprintf(f, "    \"warmup\": %d,\n", warmup);
  std::fprintf(f, "    \"resolution\": 2.0,\n");
  std::fprintf(f, "    \"max_iterations\": 30,\n");
  std::fprintf(f, "    \"num_threads\": 1,\n");
  std::fprintf(f, "    \"clock\": \"steady_clock\",\n");
  std::fprintf(f, "    \"unit\": \"ms\",\n");
  std::fprintf(f, "    \"note\": \"align loop only; map+kdtree built once per engine\"\n");
  std::fprintf(f, "  },\n");
  std::fprintf(f, "  \"engines\": {\n");
  std::fprintf(f, "    \"cpp\": { \"label\": \"C++ (multigrid_ndt_omp)\", \"iteration_num\": %d, \"samples_ms\": ", cpp_iter);
  write_samples(f, cpp_ms);
  std::fprintf(f, " },\n");
  std::fprintf(f, "    \"rust\": { \"label\": \"Rust (autoware_ndt_scan_matcher_rs)\", \"iteration_num\": %d, \"samples_ms\": ", rs_iter);
  write_samples(f, rs_ms);
  std::fprintf(f, " }\n");
  std::fprintf(f, "  }\n");
  std::fprintf(f, "}\n");
  std::fclose(f);

  std::fprintf(
    stderr, "ndt_bench_replay: %zu pts, %d aligns/engine (+%d warmup) -> %s\n", n_pts, iters, warmup,
    out_path.c_str());
  return 0;
}
