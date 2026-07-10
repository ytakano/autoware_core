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

#include <autoware/ndt_scan_matcher/ndt_omp/multigrid_ndt_omp.h>

#include "autoware_ndt_scan_matcher_rs.h"

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <dlfcn.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
}  // namespace

int main(int argc, char ** argv)
{
  // WCET fixture mode: ndt_bench_replay --fixture out.json fix1.ndtfix [fix2.ndtfix ...]
  if (argc >= 4 && std::strcmp(argv[1], "--fixture") == 0) {
    std::vector<std::string> paths(argv + 3, argv + argc);
    return run_fixture_mode(argv[2], paths);
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
