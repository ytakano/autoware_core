// C1 analysis-build trace instrumentation (plan/paper_fix2.md C1).
//
// This header exists ONLY in the traced analysis build (bench/traced/, CMake option
// NDT_BUILD_TRACED). It is never part of the shipped engine: the upstream sources under
// src/ndt_omp/ and include/ stay byte-identical; the traced engine library is built from the
// instrumented COPIES in bench/traced/.
//
// What it provides:
//  - a 64-bit FNV-1a fold (bit-identical to the Rust mirror in realtime_ndt_scan_matcher's
//    `wcet` module; shared test vectors in both test suites),
//  - a thread-local per-align trace: one PassTrace per derivative pass with the pass's point
//    count, neighbor count, the neighbor-identity hash (per point: an order-insensitive XOR of
//    per-leaf mean-bit hashes, folded across points in index order -- see point_record), and
//    the pass-final score/gradient/Hessian bits,
//  - a CountingL2 FLANN distance functor: the traced MultiVoxelGridCovariance's KdTreeFLANN is
//    instantiated with it, counting the candidate-point distance evaluations (and accum_dist
//    plane checks) FLANN performs -- the C++-native traversal-work counter Sigma_kd^C++. This
//    is a property of the C++ implementation and is never compared for equality with the Rust
//    kd counter (a different tree, a different metric).
//
// Determinism contract: the trace is only meaningful for the serial engine configuration
// (num_threads == 1). pass_begin() poisons the trace when called with num_threads > 1.

#ifndef NDT_TRACE_HPP_
#define NDT_TRACE_HPP_

#include <flann/algorithms/dist.h>

#include <Eigen/Core>

#include <cstdint>
#include <cstring>

namespace ndt_trace
{

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::size_t kMaxPasses = 40;

// FNV-1a over the 8 little-endian bytes of v. Mirrors realtime_ndt_scan_matcher exactly.
inline std::uint64_t fnv1a_u64(std::uint64_t h, std::uint64_t v)
{
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (8 * i)) & 0xffULL;
    h *= kFnvPrime;
  }
  return h;
}

inline std::uint64_t fnv1a_f64(std::uint64_t h, double v)
{
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(v), "f64 must be 8 bytes");
  std::memcpy(&bits, &v, sizeof(bits));
  return fnv1a_u64(h, bits);
}

struct PassTrace
{
  std::uint64_t points = 0;
  std::uint64_t neighbors = 0;
  std::uint64_t kd_dist = 0;    // CountingL2 full-distance evaluations in this pass
  std::uint64_t kd_accum = 0;   // CountingL2 accum_dist (plane) evaluations in this pass
  std::uint64_t nbr_hash = kFnvOffset;
  std::uint64_t score_bits = 0;
  std::uint64_t grad_hash = kFnvOffset;
  std::uint64_t hess_hash = kFnvOffset;
};

struct TraceState
{
  std::uint64_t passes = 0;          // total passes seen (may exceed kMaxPasses)
  std::uint64_t line_search_loops = 0;
  std::uint64_t kd_dist_total = 0;
  std::uint64_t kd_accum_total = 0;
  bool poisoned = false;             // set when num_threads > 1 (order not deterministic)
  PassTrace pass[kMaxPasses];
  // running counters for the CURRENT pass's kd baseline
  std::uint64_t kd_dist_at_pass_begin = 0;
  std::uint64_t kd_accum_at_pass_begin = 0;
};

inline TraceState & state()
{
  static thread_local TraceState s;
  return s;
}

inline void reset()
{
  state() = TraceState{};
}

inline void pass_begin(int num_threads)
{
  TraceState & s = state();
  if (num_threads > 1) {
    s.poisoned = true;
  }
  if (s.passes < kMaxPasses) {
    s.pass[s.passes] = PassTrace{};
  }
  s.kd_dist_at_pass_begin = s.kd_dist_total;
  s.kd_accum_at_pass_begin = s.kd_accum_total;
}

// Called once per source point, right after its radius search, BEFORE the empty-neighborhood
// early-continue -- so every point is counted (Rust counts points_processed the same way).
//
// Neighbor-identity hashing is ORDER-INSENSITIVE within a point (XOR of per-leaf hashes) and
// order-sensitive across points: pcl/FLANN returns each point's neighbors distance-sorted while
// the Rust kd walk returns them in visit order, so within-point order legitimately differs
// between the engines (it perturbs only the fp summation rounding, which the numeric legs
// report separately). The certificate claim is "the same neighbor SET per point".
template <typename NeighborhoodT>
inline void point_record(const NeighborhoodT & neighborhood)
{
  TraceState & s = state();
  if (s.passes >= kMaxPasses) {
    return;
  }
  PassTrace & p = s.pass[s.passes];
  p.points += 1;
  p.neighbors += static_cast<std::uint64_t>(neighborhood.size());
  std::uint64_t set_hash = 0;
  for (const auto & cell : neighborhood) {
    const Eigen::Vector3d & mean = cell->getMean();
    std::uint64_t leaf_hash = kFnvOffset;
    leaf_hash = fnv1a_f64(leaf_hash, mean(0));
    leaf_hash = fnv1a_f64(leaf_hash, mean(1));
    leaf_hash = fnv1a_f64(leaf_hash, mean(2));
    set_hash ^= leaf_hash;
  }
  p.nbr_hash = fnv1a_u64(p.nbr_hash, set_hash);
}

inline void pass_end(
  double score, const Eigen::Matrix<double, 6, 1> & gradient,
  const Eigen::Matrix<double, 6, 6> & hessian)
{
  TraceState & s = state();
  if (s.passes < kMaxPasses) {
    PassTrace & p = s.pass[s.passes];
    std::uint64_t bits = 0;
    std::memcpy(&bits, &score, sizeof(bits));
    p.score_bits = bits;
    for (int r = 0; r < 6; ++r) {
      p.grad_hash = fnv1a_f64(p.grad_hash, gradient(r));
    }
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        p.hess_hash = fnv1a_f64(p.hess_hash, hessian(r, c));
      }
    }
    p.kd_dist = s.kd_dist_total - s.kd_dist_at_pass_begin;
    p.kd_accum = s.kd_accum_total - s.kd_accum_at_pass_begin;
  }
  s.passes += 1;
}

inline void line_search_entered()
{
  state().line_search_loops += 1;
}

// FLANN distance functor: forwards to L2_Simple<float> bit-for-bit, counting evaluations.
// Instantiating pcl::KdTreeFLANN<PointT, CountingL2> compiles the kd-tree template code into
// the traced TU (a distinct specialization from the precompiled <PointT, L2_Simple<float>> in
// libpcl_kdtree), so the counters are guaranteed live and there is no ODR overlap.
struct CountingL2
{
  typedef bool is_kdtree_distance;
  typedef float ElementType;
  typedef flann::Accumulator<float>::Type ResultType;

  template <typename Iterator1, typename Iterator2>
  ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
  {
    state().kd_dist_total += 1;
    return base_(a, b, size, worst_dist);
  }

  template <typename U, typename V>
  inline ResultType accum_dist(const U & a, const V & b, int dim) const
  {
    state().kd_accum_total += 1;
    return base_.accum_dist(a, b, dim);
  }

  flann::L2_Simple<float> base_;
};

}  // namespace ndt_trace

#endif  // NDT_TRACE_HPP_
