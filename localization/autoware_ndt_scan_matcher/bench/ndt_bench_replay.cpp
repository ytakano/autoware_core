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

#ifdef NDT_TRACE
// C1 trace certificate (plan/paper_fix2.md): per-pass trace comparison between the traced C++
// engine (bench/traced/, ndt_trace.hpp) and the Rust engine's mirrored `wcet-trace` records.
#include <ndt_trace.hpp>

// The Rust trace FFI (struct AwNdtPassTrace + ..._ndt_engine_get_trace) comes from the
// cbindgen-generated header above; the symbol itself exists only in `wcet-trace` builds,
// which NDT_BUILD_TRACED enables.
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

// One frozen align input, mirroring the Rust `realtime_ndt_scan_matcher::fixture` "NDTFIX01" layout
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
// Software cache-eviction approximation for the cold-cache series (WCET_EVICT_BYTES):
// stream-write then stride-read the buffer between timed samples, outside the timed region.
// This is an approximation, not a guaranteed flush (plan/ndt_timing_measurement_policy.md,
// "Cache Measurement Policy").
void evict_caches(std::vector<char> & buf)
{
  for (size_t i = 0; i < buf.size(); i += 64) {
    buf[i] = static_cast<char>(i);
  }
  volatile long long sink = 0;
  for (size_t i = 0; i < buf.size(); i += 64) {
    sink += buf[i];
  }
  (void)sink;
}

#ifdef NDT_TRACE
// Snapshot of one engine's per-align trace, normalized for comparison.
struct TraceSnap
{
  uint64_t passes = 0;
  uint64_t line_search_loops = 0;  // C++ only
  uint64_t kd_a = 0;               // C++: FLANN distance evals | Rust: own kd counter
  uint64_t kd_b = 0;               // C++: FLANN accum_dist evals | Rust: 0
  bool poisoned = false;
  size_t stored = 0;
  ndt_trace::PassTrace pass[ndt_trace::kMaxPasses];
};

TraceSnap snap_cpp()
{
  const ndt_trace::TraceState & s = ndt_trace::state();
  TraceSnap out;
  out.passes = s.passes;
  out.line_search_loops = s.line_search_loops;
  out.kd_a = s.kd_dist_total;
  out.kd_b = s.kd_accum_total;
  out.poisoned = s.poisoned;
  out.stored = static_cast<size_t>(std::min<uint64_t>(s.passes, ndt_trace::kMaxPasses));
  for (size_t i = 0; i < out.stored; ++i) out.pass[i] = s.pass[i];
  return out;
}

TraceSnap snap_rust(struct AwNdtEngine * eng)
{
  TraceSnap out;
  std::array<AwNdtPassTrace, ndt_trace::kMaxPasses> buf{};
  uint64_t total = 0;
  const bool ok = autoware_ndt_scan_matcher_rs_ndt_engine_get_trace(
    eng, buf.data(), buf.size(), &total);
  out.passes = total;
  out.poisoned = !ok;
  out.stored = static_cast<size_t>(std::min<uint64_t>(total, ndt_trace::kMaxPasses));
  for (size_t i = 0; i < out.stored; ++i) {
    out.pass[i].points = buf[i].points;
    out.pass[i].neighbors = buf[i].neighbors;
    out.pass[i].kd_dist = buf[i].kd_nodes;
    out.pass[i].kd_accum = 0;
    out.pass[i].nbr_hash = buf[i].nbr_hash;
    out.pass[i].score_bits = buf[i].score_bits;
    out.pass[i].grad_hash = buf[i].grad_hash;
    out.pass[i].hess_hash = buf[i].hess_hash;
    out.kd_a += buf[i].kd_nodes;
  }
  return out;
}

// Per-leg comparison result. Structural legs (pass count, per-pass points/neighbors and the
// neighbor-identity hash) certify equal work; the numeric legs (score/gradient/Hessian bits)
// certify the operand stream. kd counters are engine-own metrics, recorded but never compared.
struct TraceCert
{
  bool valid = false;        // both traces present, unpoisoned
  bool structural = false;
  bool score = false;
  bool grad = false;
  bool hess = false;
  uint64_t score_max_ulp = 0;  // max per-pass score ULP distance (0 == bit-identical)
  int first_div_pass = -1;   // first pass where any compared leg diverges
  const char * first_div_leg = "";
};

// ULP distance between two f64 bit patterns (IEEE-754 total order via sign-magnitude flip).
uint64_t ulp_delta_f64(uint64_t a, uint64_t b)
{
  const auto key = [](uint64_t x) -> uint64_t {
    return (x & 0x8000000000000000ULL) != 0 ? ~x : (x | 0x8000000000000000ULL);
  };
  const uint64_t ka = key(a);
  const uint64_t kb = key(b);
  return ka > kb ? ka - kb : kb - ka;
}

TraceCert compare_traces(const TraceSnap & c, const TraceSnap & r)
{
  TraceCert out;
  out.valid = !c.poisoned && !r.poisoned;
  if (!out.valid) return out;
  out.structural = (c.passes == r.passes);
  out.score = out.structural;
  out.grad = out.structural;
  out.hess = out.structural;
  if (!out.structural) {
    out.first_div_pass = static_cast<int>(std::min(c.passes, r.passes));
    out.first_div_leg = "passes";
    return out;
  }
  const size_t n = std::min(c.stored, r.stored);
  for (size_t i = 0; i < n; ++i) {
    const auto & a = c.pass[i];
    const auto & b = r.pass[i];
    const char * leg = nullptr;
    if (a.points != b.points) leg = "points";
    else if (a.neighbors != b.neighbors) leg = "neighbors";
    else if (a.nbr_hash != b.nbr_hash) leg = "nbr_hash";
    if (leg != nullptr) {
      out.structural = out.score = out.grad = out.hess = false;
      out.first_div_pass = static_cast<int>(i);
      out.first_div_leg = leg;
      return out;
    }
    if (a.score_bits != b.score_bits) {
      out.score_max_ulp = std::max(out.score_max_ulp, ulp_delta_f64(a.score_bits, b.score_bits));
      if (out.score) {
        out.score = false;
        if (out.first_div_pass < 0) { out.first_div_pass = static_cast<int>(i); out.first_div_leg = "score"; }
      }
    }
    if (out.grad && a.grad_hash != b.grad_hash) {
      out.grad = false;
      if (out.first_div_pass < 0) { out.first_div_pass = static_cast<int>(i); out.first_div_leg = "grad"; }
    }
    if (out.hess && a.hess_hash != b.hess_hash) {
      out.hess = false;
      if (out.first_div_pass < 0) { out.first_div_pass = static_cast<int>(i); out.first_div_leg = "hess"; }
    }
  }
  return out;
}

void write_trace_json(FILE * f, const TraceSnap & c, const TraceSnap & r, const TraceCert & t)
{
  std::fprintf(f,
    "\"trace\": { \"valid\": %s, \"passes_cpp\": %llu, \"passes_rust\": %llu, "
    "\"structural_match\": %s, \"score_match\": %s, \"grad_match\": %s, "
    "\"hess_match\": %s, \"first_div_pass\": %d, \"first_div_leg\": \"%s\", "
    "\"line_search_loops\": %llu, \"cpp_kd_dist\": %llu, \"cpp_kd_accum\": %llu, "
    "\"rust_kd\": %llu, \"score_max_ulp\": %llu }",
    t.valid ? "true" : "false", static_cast<unsigned long long>(c.passes),
    static_cast<unsigned long long>(r.passes), t.structural ? "true" : "false",
    t.score ? "true" : "false", t.grad ? "true" : "false", t.hess ? "true" : "false",
    t.first_div_pass, t.first_div_leg, static_cast<unsigned long long>(c.line_search_loops),
    static_cast<unsigned long long>(c.kd_a), static_cast<unsigned long long>(c.kd_b),
    static_cast<unsigned long long>(r.kd_a), static_cast<unsigned long long>(t.score_max_ulp));
}

// Startup self-test: the C++ FNV-1a mirror must reproduce the Rust crate's shared vectors
// (realtime_ndt_scan_matcher::wcet trace_tests).
bool trace_selftest()
{
  using ndt_trace::fnv1a_u64;
  using ndt_trace::kFnvOffset;
  bool ok = fnv1a_u64(kFnvOffset, 0) == 0xa8c7f832281a39c5ULL;
  double one = 1.0;
  uint64_t bits = 0;
  std::memcpy(&bits, &one, 8);
  ok = ok && fnv1a_u64(kFnvOffset, bits) == 0xaab1693229ba1db8ULL;
  if (!ok) std::fprintf(stderr, "trace: FNV-1a self-test FAILED (C++/Rust mirror broken)\n");
  return ok;
}
#endif

int run_fixture_mode(const std::string & out_path, const std::vector<std::string> & paths)
{
  const int iters = std::getenv("WCET_ITERS") ? std::atoi(std::getenv("WCET_ITERS")) : 100;
  const int warmup = std::getenv("WCET_WARMUP") ? std::atoi(std::getenv("WCET_WARMUP")) : 10;
  // Campaign controls (plan/ndt_timing_measurement_policy.md): engine selection so the
  // orchestrator can randomize C++/Rust order across invocations, and between-sample cache
  // eviction for the cold series. Defaults preserve the original both-engines/warm behavior.
  const char * eng_env = std::getenv("WCET_ENGINE");
  const std::string engine_sel = (eng_env != nullptr) ? eng_env : "both";
  if (engine_sel != "both" && engine_sel != "cpp" && engine_sel != "rust") {
    std::fprintf(stderr, "wcet: WCET_ENGINE must be cpp|rust|both (got '%s')\n",
      engine_sel.c_str());
    return 1;
  }
  const bool run_cpp = engine_sel != "rust";
  const bool run_rust = engine_sel != "cpp";
  const long long evict_bytes =
    std::getenv("WCET_EVICT_BYTES") ? std::atoll(std::getenv("WCET_EVICT_BYTES")) : 0;
  std::vector<char> evict_buf(evict_bytes > 0 ? static_cast<size_t>(evict_bytes) : 0);
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
  std::fprintf(f, "    \"engine\": \"%s\",\n", engine_sel.c_str());
  std::fprintf(f, "    \"evict_bytes\": %lld,\n", evict_bytes);
  std::fprintf(f, "    \"cache_condition\": \"%s\",\n", evict_bytes > 0 ? "cold" : "warm");
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
    std::vector<double> cpp_ms;
    long long cpp_allocs = -1;
    int cpp_iter = -1;
    pclomp::NdtResult cpp_res{};
#ifdef NDT_TRACE
    TraceSnap trace_cpp{};
    TraceSnap trace_rust{};
    bool have_trace_cpp = false;
    bool have_trace_rust = false;
#endif
    if (run_cpp) {
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
      if (hooks.reset && hooks.get) hooks.reset();
      cpp_ms.reserve(static_cast<size_t>(iters));
      for (int i = 0; i < iters; ++i) {
        if (!evict_buf.empty()) evict_caches(evict_buf);
        const auto t0 = Clock::now();
        ndt.align(*aligned, guess, source);
        cpp_ms.push_back(ms_since(t0));
      }
      if (hooks.reset && hooks.get) {
        cpp_allocs = static_cast<long long>(hooks.get() / static_cast<unsigned long long>(iters));
      }
      cpp_res = ndt.getResult();
      cpp_iter = cpp_res.iteration_num;
#ifdef NDT_TRACE
      ndt_trace::reset();
      ndt.align(*aligned, guess, source);
      trace_cpp = snap_cpp();
      have_trace_cpp = true;
#endif
    }

    // ---- Rust engine (over the C ABI) ----
    std::vector<double> rs_ms;
    long long rs_allocs = -1;
    int32_t rs_iter = -1;
    std::array<float, 16> rs_pose{};
    float rs_tp = 0.0F;
    float rs_nvtl = 0.0F;
    if (run_rust) {
      struct AwNdtEngine * eng =
        autoware_ndt_scan_matcher_rs_ndt_engine_new(fx.resolution, 6, 0.01);
      autoware_ndt_scan_matcher_rs_ndt_engine_set_params(
        eng, fx.trans_epsilon, fx.step_size, fx.resolution, fx.max_iterations, fx.outlier_ratio,
        1);
      for (size_t t = 0; t < fx.tiles.size(); ++t) {
        autoware_ndt_scan_matcher_rs_ndt_engine_add_target(
          eng, fx.tiles[t].data(), fx.tiles[t].size() / 3, t);
      }
      autoware_ndt_scan_matcher_rs_ndt_engine_create_kdtree(eng);
      for (int i = 0; i < warmup; ++i) {
        autoware_ndt_scan_matcher_rs_ndt_engine_align(
          eng, fx.guess16.data(), fx.source.data(), n_src);
      }
      if (hooks.reset && hooks.get) hooks.reset();
      rs_ms.reserve(static_cast<size_t>(iters));
      for (int i = 0; i < iters; ++i) {
        if (!evict_buf.empty()) evict_caches(evict_buf);
        const auto t0 = Clock::now();
        autoware_ndt_scan_matcher_rs_ndt_engine_align(
          eng, fx.guess16.data(), fx.source.data(), n_src);
        rs_ms.push_back(ms_since(t0));
      }
      if (hooks.reset && hooks.get) {
        rs_allocs = static_cast<long long>(hooks.get() / static_cast<unsigned long long>(iters));
      }
      rs_iter = 0;
      AwNdtAlignOutput rout{};
      rout.iteration_num = &rs_iter;
      rout.pose = rs_pose.data();
      rout.transform_probability = &rs_tp;
      rout.nearest_voxel_likelihood = &rs_nvtl;
      autoware_ndt_scan_matcher_rs_ndt_engine_get_result(eng, &rout);
#ifdef NDT_TRACE
      trace_rust = snap_rust(eng);
      have_trace_rust = true;
#endif
      autoware_ndt_scan_matcher_rs_ndt_engine_free(eng);
    }
    const bool both = run_cpp && run_rust;

    // ---- equal-work assert (only meaningful when both engines ran in this invocation;
    // single-engine cells are paired and re-asserted by the campaign orchestrator's merge)
    bool iter_match = true;
    bool pose_match = true;
    double trans_delta = 0.0;
    double rot_delta = 0.0;
    double tp_delta = 0.0;
    double nvtl_delta = 0.0;
    if (both) {
      iter_match = (cpp_iter == rs_iter);
      if (!iter_match) {
        std::fprintf(stderr, "wcet: ITERATION MISMATCH on %s: cpp=%d rust=%d\n", path.c_str(),
          cpp_iter, rs_iter);
        rc = 1;
      }

      // ---- pose/score certificate (plan/paper_fix.md B1): the equal-work certificate's
      // second leg. Both engines already expose the final pose and both scores, so agreement
      // is asserted per fixture with no extra instrumentation.
      for (int r = 0; r < 3; ++r) {
        const double dt = static_cast<double>(cpp_res.pose(r, 3)) -
                          static_cast<double>(rs_pose[(static_cast<size_t>(r) * 4) + 3]);
        trans_delta += dt * dt;
        for (int c = 0; c < 3; ++c) {
          rot_delta = std::max(
            rot_delta, std::abs(
                         static_cast<double>(cpp_res.pose(r, c)) -
                         static_cast<double>(rs_pose[(static_cast<size_t>(r) * 4) + c])));
        }
      }
      trans_delta = std::sqrt(trans_delta);
      tp_delta = std::abs(
        static_cast<double>(cpp_res.transform_probability) - static_cast<double>(rs_tp));
      nvtl_delta = std::abs(
        static_cast<double>(cpp_res.nearest_voxel_transformation_likelihood) -
        static_cast<double>(rs_nvtl));
      pose_match =
        trans_delta < 1e-4 && rot_delta < 1e-5 && tp_delta < 1e-4 && nvtl_delta < 1e-4;
      if (!pose_match) {
        std::fprintf(stderr,
          "wcet: POSE/SCORE MISMATCH on %s: dtrans=%.3e drot=%.3e dtp=%.3e dnvtl=%.3e\n",
          path.c_str(), trans_delta, rot_delta, tp_delta, nvtl_delta);
        rc = 1;
      }
    }

    const std::string name = fixture_stem(path);
    std::fprintf(f, "%s    \"%s\": {\n", (first ? "" : ",\n"), name.c_str());
    first = false;
    std::fprintf(f, "      \"n_map\": %zu,\n", n_map);
    std::fprintf(f, "      \"n_tiles\": %zu,\n", fx.tiles.size());
    std::fprintf(f, "      \"n_source\": %zu,\n", n_src);
    std::fprintf(f, "      \"max_iterations\": %d,\n", fx.max_iterations);
    if (both) {
      std::fprintf(f, "      \"iter_match\": %s,\n", iter_match ? "true" : "false");
      std::fprintf(
        f,
        "      \"cert\": { \"pose_match\": %s, \"trans_delta_m\": %.3e, \"rot_delta\": %.3e, "
        "\"tp_delta\": %.3e, \"nvtl_delta\": %.3e },\n",
        pose_match ? "true" : "false", trans_delta, rot_delta, tp_delta, nvtl_delta);
#ifdef NDT_TRACE
      if (have_trace_cpp && have_trace_rust) {
        const TraceCert tc = compare_traces(trace_cpp, trace_rust);
        std::fprintf(f, "      ");
        write_trace_json(f, trace_cpp, trace_rust, tc);
        std::fprintf(f, ",\n");
        std::fprintf(stderr,
          "trace: %-18s passes %llu/%llu structural %s score %s grad %s hess %s ls=%llu "
          "kd_cpp=%llu+%llu kd_rust=%llu%s%s\n",
          fixture_stem(path).c_str(), static_cast<unsigned long long>(trace_cpp.passes),
          static_cast<unsigned long long>(trace_rust.passes), tc.structural ? "OK" : "DIVERGE",
          tc.score ? "OK" : "DIVERGE", tc.grad ? "OK" : "DIVERGE", tc.hess ? "OK" : "DIVERGE",
          static_cast<unsigned long long>(trace_cpp.line_search_loops),
          static_cast<unsigned long long>(trace_cpp.kd_a),
          static_cast<unsigned long long>(trace_cpp.kd_b),
          static_cast<unsigned long long>(trace_rust.kd_a),
          tc.first_div_pass >= 0 ? " first_div=" : "",
          tc.first_div_pass >= 0 ? tc.first_div_leg : "");
        std::fprintf(stderr, "trace: %-18s score_max_ulp=%llu\n", fixture_stem(path).c_str(),
          static_cast<unsigned long long>(tc.score_max_ulp));
        if (!tc.structural) rc = 1;
      }
#endif
    }
    bool first_engine = true;
    if (run_cpp) {
      std::fprintf(
        f, "      \"cpp\": { \"iteration_num\": %d, \"allocs_per_align\": %lld, \"samples_ms\": ",
        cpp_iter, cpp_allocs);
      write_samples(f, cpp_ms);
      std::fprintf(f, " }");
      first_engine = false;
    }
    if (run_rust) {
      std::fprintf(f, "%s", first_engine ? "" : ",\n");
      std::fprintf(
        f, "      \"rust\": { \"iteration_num\": %d, \"allocs_per_align\": %lld, \"samples_ms\": ",
        rs_iter, rs_allocs);
      write_samples(f, rs_ms);
      std::fprintf(f, " }");
    }
    std::fprintf(f, "\n    }");

    if (both) {
      std::fprintf(stderr,
        "wcet: %-18s map=%zu src=%zu iter cpp=%d rust=%d %s pose/score %s (dtrans=%.1e)\n",
        name.c_str(), n_map, n_src, cpp_iter, rs_iter, iter_match ? "OK" : "MISMATCH",
        pose_match ? "OK" : "MISMATCH", trans_delta);
    } else {
      std::fprintf(stderr, "wcet: %-18s map=%zu src=%zu engine=%s iter=%d\n", name.c_str(),
        n_map, n_src, engine_sel.c_str(), run_cpp ? cpp_iter : static_cast<int>(rs_iter));
    }
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
      // WCET_DUMP: compare the two engines' voxel maps (leaf means as sorted multisets).
      if (std::getenv("WCET_DUMP") != nullptr) {
        auto cpd = ndt->getVoxelPCD();
        std::vector<std::array<float, 3>> cppm;
        for (const auto & q : cpd) cppm.push_back({q.x, q.y, q.z});
        const double leaf3[3] = {cp.resolution, cp.resolution, cp.resolution};
        struct AwNdtVoxelGridMap * rmap =
          autoware_ndt_scan_matcher_rs_voxel_grid_map_new(leaf3, 6, 0.01);
        for (size_t i = 0; i < fr.ids.size(); ++i) {
          const auto & e2 = tile_cache.at(hex_of(fr.ids[i]));
          autoware_ndt_scan_matcher_rs_voxel_grid_map_add_target(
            rmap, e2.second.data(), e2.second.size() / 3, static_cast<uint64_t>(i));
        }
        autoware_ndt_scan_matcher_rs_voxel_grid_map_create_kdtree(rmap);
        std::vector<std::array<float, 3>> rustm;
        for (uint32_t i = 0;; ++i) {
          double mean[3];
          double icov[9];
          if (!autoware_ndt_scan_matcher_rs_voxel_grid_map_leaf(rmap, i, mean, icov)) break;
          rustm.push_back({static_cast<float>(mean[0]), static_cast<float>(mean[1]),
            static_cast<float>(mean[2])});
        }
        autoware_ndt_scan_matcher_rs_voxel_grid_map_free(rmap);
        // icov comparison: fetch each Rust leaf's C++ counterpart via a tiny radiusSearch on a
        // standalone MultiVoxelGridCovariance built exactly as the NDT builds its target cells.
        {
          pclomp::MultiVoxelGridCovariance<pcl::PointXYZ> mv;
          mv.setLeafSize(static_cast<float>(cp.resolution), static_cast<float>(cp.resolution),
            static_cast<float>(cp.resolution));
          for (size_t i = 0; i < fr.ids.size(); ++i) {
            const auto & e2 = tile_cache.at(hex_of(fr.ids[i]));
            char name[16];
            std::snprintf(name, sizeof(name), "%06zu", i);
            mv.setInputCloudAndFilter(e2.first, name);
          }
          mv.createKdtree();
          struct AwNdtVoxelGridMap * rmap2 = autoware_ndt_scan_matcher_rs_voxel_grid_map_new(
            leaf3, 6, 0.01);
          for (size_t i = 0; i < fr.ids.size(); ++i) {
            const auto & e2 = tile_cache.at(hex_of(fr.ids[i]));
            autoware_ndt_scan_matcher_rs_voxel_grid_map_add_target(
              rmap2, e2.second.data(), e2.second.size() / 3, static_cast<uint64_t>(i));
          }
          autoware_ndt_scan_matcher_rs_voxel_grid_map_create_kdtree(rmap2);
          size_t checked = 0, icov_exact = 0, icov_small = 0, icov_big = 0, mean_exact = 0;
          double worst = 0.0;
          for (uint32_t i = 0; i < 400000U; ++i) {
            double rmean[3];
            double ricov[9];
            if (!autoware_ndt_scan_matcher_rs_voxel_grid_map_leaf(rmap2, i, rmean, ricov)) break;
            pcl::PointXYZ q;
            q.x = static_cast<float>(rmean[0]);
            q.y = static_cast<float>(rmean[1]);
            q.z = static_cast<float>(rmean[2]);
            std::vector<pclomp::MultiVoxelGridCovariance<pcl::PointXYZ>::LeafConstPtr> ls;
            mv.radiusSearch(q, 0.001, ls, 1);
            if (ls.empty()) continue;
            ++checked;
            const auto & cm = ls[0]->getMean();
            const auto & ci = ls[0]->getInverseCov();
            if (cm(0) == rmean[0] && cm(1) == rmean[1] && cm(2) == rmean[2]) ++mean_exact;
            double rel = 0.0;
            bool exact = true;
            for (int r = 0; r < 3; ++r)
              for (int c = 0; c < 3; ++c) {
                const double a = ci(r, c);
                const double b = ricov[(r * 3) + c];
                if (a != b) exact = false;
                const double denom = std::max(std::abs(a), 1e-12);
                rel = std::max(rel, std::abs(a - b) / denom);
              }
            if (exact) ++icov_exact;
            else if (rel < 1e-6) ++icov_small;
            else ++icov_big;
            worst = std::max(worst, rel);
          }
          std::fprintf(stderr,
            "DUMP icov: checked=%zu mean_exact=%zu icov_exact=%zu small(<1e-6)=%zu big=%zu "
            "worst_rel=%.3e\n",
            checked, mean_exact, icov_exact, icov_small, icov_big, worst);
          // Neighbor-set comparison at the actual (transformed) source points of this frame.
          {
            Eigen::Matrix4f g;
            for (int r = 0; r < 4; ++r)
              for (int c = 0; c < 4; ++c) g(r, c) = fr.guess16[(r * 4) + c];
            size_t pts = 0, k_equal = 0, set_equal = 0, k_diff = 0;
            std::vector<uint32_t> ridx(128);
            for (size_t pi = 0; pi + 2 < fr.source.size(); pi += 3) {
              const Eigen::Vector4f sp(fr.source[pi], fr.source[pi + 1], fr.source[pi + 2], 1.0F);
              const Eigen::Vector4f tp4 = g * sp;
              pcl::PointXYZ q;
              q.x = tp4(0);
              q.y = tp4(1);
              q.z = tp4(2);
              std::vector<pclomp::MultiVoxelGridCovariance<pcl::PointXYZ>::LeafConstPtr> ls;
              mv.radiusSearch(q, cp.resolution, ls);
              const float qf[3] = {tp4(0), tp4(1), tp4(2)};
              const uint32_t rk = autoware_ndt_scan_matcher_rs_voxel_grid_map_radius_search(
                rmap2, qf, cp.resolution, 64, ridx.data(), 128);
              ++pts;
              if (ls.size() == rk) {
                ++k_equal;
                // Compare leaf identity via bitwise means (order-insensitive).
                std::vector<std::array<double, 3>> a, b;
                for (const auto & lp : ls) {
                  const auto & m = lp->getMean();
                  a.push_back({m(0), m(1), m(2)});
                }
                for (uint32_t j = 0; j < rk; ++j) {
                  double rm[3];
                  double ric[9];
                  autoware_ndt_scan_matcher_rs_voxel_grid_map_leaf(rmap2, ridx[j], rm, ric);
                  b.push_back({rm[0], rm[1], rm[2]});
                }
                std::sort(a.begin(), a.end());
                std::sort(b.begin(), b.end());
                if (a == b) ++set_equal;
              } else {
                ++k_diff;
                if (k_diff <= 3) {
                  std::fprintf(stderr,
                    "  nbr-diff at pt %zu: cpp K=%zu rust K=%u (query %.6f,%.6f,%.6f)\n", pi / 3,
                    ls.size(), rk, static_cast<double>(q.x), static_cast<double>(q.y),
                    static_cast<double>(q.z));
                }
              }
            }
            std::fprintf(stderr,
              "DUMP nbr: pts=%zu K-equal=%zu set-equal=%zu K-diff=%zu\n", pts, k_equal,
              set_equal, k_diff);
          }
          autoware_ndt_scan_matcher_rs_voxel_grid_map_free(rmap2);
        }
        std::sort(cppm.begin(), cppm.end());
        std::sort(rustm.begin(), rustm.end());
        size_t diff = 0;
        for (size_t i = 0; i < std::min(cppm.size(), rustm.size()); ++i) {
          if (std::memcmp(cppm[i].data(), rustm[i].data(), 12) != 0) ++diff;
        }
        std::fprintf(stderr, "DUMP map: cpp leaves=%zu rust leaves=%zu mean-mismatches=%zu\n",
          cppm.size(), rustm.size(), diff);
        if (diff > 0) {
          for (size_t i = 0, shown = 0; i < std::min(cppm.size(), rustm.size()) && shown < 3;
               ++i) {
            if (std::memcmp(cppm[i].data(), rustm[i].data(), 12) != 0) {
              std::fprintf(stderr, "  leaf %zu: cpp (%.9g,%.9g,%.9g) rust (%.9g,%.9g,%.9g)\n", i,
                cppm[i][0], cppm[i][1], cppm[i][2], rustm[i][0], rustm[i][1], rustm[i][2]);
              ++shown;
            }
          }
        }
      }
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
#ifdef NDT_TRACE
      ndt_trace::reset();
#endif
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
    // WCET_DUMP=1: per-iteration score/pose comparison for divergence root-causing.
    if (std::getenv("WCET_DUMP") != nullptr) {
      const auto res = ndt->getResult();
      std::vector<float> rtp(64), rnvl(64);
      uint32_t rcount = 0;
      autoware_ndt_scan_matcher_rs_ndt_engine_get_score_arrays(
        eng, rtp.data(), rnvl.data(), 64, &rcount);
      std::fprintf(stderr, "DUMP frame %zu: cpp iters=%d rust iters(arrays)=%u\n", fi, cpp_iter,
        rcount);
      // First-step pose fork: translation delta between engines after Newton step 1.
      {
        std::vector<float> rtf(16 * 40);
        uint32_t rtn = 0;
        AwNdtAlignOutput ro2{};
        ro2.transformation_array = rtf.data();
        ro2.transforms_cap = 40;
        ro2.transforms_count = &rtn;
        autoware_ndt_scan_matcher_rs_ndt_engine_get_result(eng, &ro2);
        if (!res.transformation_array.empty() && rtn > 0) {
          const auto & cp1 = res.transformation_array[0];
          const double dx = static_cast<double>(cp1(0, 3)) - static_cast<double>(rtf[3]);
          const double dy = static_cast<double>(cp1(1, 3)) - static_cast<double>(rtf[7]);
          const double dz = static_cast<double>(cp1(2, 3)) - static_cast<double>(rtf[11]);
          std::fprintf(stderr, "  pose1 delta: (%.3e, %.3e, %.3e) m\n", dx, dy, dz);
        }
      }
      const size_t n_it =
        std::max(res.transform_probability_array.size(), static_cast<size_t>(rcount));
      for (size_t i = 0; i < n_it; ++i) {
        const float ct = i < res.transform_probability_array.size()
                           ? res.transform_probability_array[i]
                           : -1.0F;
        const float cn = i < res.nearest_voxel_transformation_likelihood_array.size()
                           ? res.nearest_voxel_transformation_likelihood_array[i]
                           : -1.0F;
        const float rt = i < rcount ? rtp[i] : -1.0F;
        const float rn = i < rcount ? rnvl[i] : -1.0F;
        uint32_t ctb;
        uint32_t rtb;
        std::memcpy(&ctb, &ct, 4);
        std::memcpy(&rtb, &rt, 4);
        std::fprintf(stderr,
          "  it %2zu: tp cpp=%.9e rust=%.9e %s (bits %08x/%08x)  nvl cpp=%.9e rust=%.9e %s\n", i,
          ct, rt, (ctb == rtb ? "==" : "DIFF"), ctb, rtb, cn, rn,
          ([&] { uint32_t a, b; std::memcpy(&a, &cn, 4); std::memcpy(&b, &rn, 4); return a == b; }()
             ? "=="
             : "DIFF"));
      }
    }
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
      "\"rust_ms\": %.4f, \"iter_cpp\": %d, \"iter_rust\": %d, \"match\": %s",
      (first ? "" : ",\n"), fi, n_src, fr.ids.size(), cpp_ms_v[cpp_ms_v.size() / 2],
      rs_ms_v[rs_ms_v.size() / 2], cpp_iter, rs_iter, match ? "true" : "false");
#ifdef NDT_TRACE
    {
      const TraceSnap tc_cpp = snap_cpp();
      const TraceSnap tc_rust = snap_rust(eng);
      const TraceCert tc = compare_traces(tc_cpp, tc_rust);
      std::fprintf(f, ", ");
      write_trace_json(f, tc_cpp, tc_rust, tc);
      // Pose/score deltas of the divergent frames (C2 rider): cheap, so emit per frame.
      double dtrans = 0.0;
      const auto res = ndt->getResult();
      for (int r = 0; r < 3; ++r) {
        const double dt = static_cast<double>(res.pose(r, 3)) -
                          static_cast<double>(rs_pose[(static_cast<size_t>(r) * 4) + 3]);
        dtrans += dt * dt;
      }
      std::fprintf(f, ", \"trans_delta_m\": %.3e", std::sqrt(dtrans));
    }
#endif
    std::fprintf(f, " }");
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
#ifdef NDT_TRACE
  if (!trace_selftest()) return 2;
#endif
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
