// Analysis-only cross-language trace instrumentation.
//
// This header is compiled only for NDT_BUILD_TRACED. The shipped engine sources remain
// byte-identical; the traced library uses instrumented copies under bench/traced/.
//
// Each derivative pass records:
//  - point and neighbor counts;
//  - a domain-separated SHA-256 chain over each point's canonical, sorted
//    (grid ordinal, voxel id) list;
//  - a separate SHA-256 payload chain that additionally includes leaf means and regularized
//    inverse covariances;
//  - score bits and FNV-1a diagnostics for the final gradient and Hessian;
//  - C++-native FLANN distance and plane-evaluation counters, which are not compared with
//    the Rust kd counter.
//
// The trace is meaningful only for the serial engine configuration. pass_begin() marks a
// trace invalid when num_threads is greater than one.

#ifndef NDT_TRACE_HPP_
#define NDT_TRACE_HPP_

#include <flann/algorithms/dist.h>

#include <Eigen/Core>

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

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

constexpr std::size_t kDigestBytes = 32;
using Digest = std::array<unsigned char, kDigestBytes>;

class Sha256
{
public:
  Sha256() : ctx_(EVP_MD_CTX_new())
  {
    ok_ = ctx_ != nullptr && EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) == 1;
  }

  ~Sha256()
  {
    EVP_MD_CTX_free(ctx_);
  }

  Sha256(const Sha256 &) = delete;
  Sha256 & operator=(const Sha256 &) = delete;

  void bytes(const void * data, std::size_t size)
  {
    if (ok_ && EVP_DigestUpdate(ctx_, data, size) != 1) ok_ = false;
  }

  void u64(std::uint64_t value)
  {
    std::array<unsigned char, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<unsigned char>((value >> (i * 8)) & 0xffU);
    }
    this->bytes(bytes.data(), bytes.size());
  }

  void f64(double value)
  {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }

  bool finish(Digest & out)
  {
    unsigned int size = 0;
    if (!ok_ || EVP_DigestFinal_ex(ctx_, out.data(), &size) != 1) return false;
    return size == out.size();
  }

private:
  EVP_MD_CTX * ctx_;
  bool ok_;
};

template <std::size_t N>
inline void digest_domain(Sha256 & hash, const char (&domain)[N])
{
  hash.bytes(domain, N);
}

template <std::size_t N>
inline bool chain_digest(const char (&domain)[N], const Digest & previous, const Digest & point,
  Digest & out)
{
  Sha256 hash;
  digest_domain(hash, domain);
  hash.bytes(previous.data(), previous.size());
  hash.bytes(point.data(), point.size());
  return hash.finish(out);
}

struct PassTrace
{
  std::uint64_t points = 0;
  std::uint64_t neighbors = 0;
  std::uint64_t kd_dist = 0;    // CountingL2 full-distance evaluations in this pass
  std::uint64_t kd_accum = 0;   // CountingL2 accum_dist (plane) evaluations in this pass
  Digest shape_digest{};
  Digest payload_digest{};
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
// Neighbors are sorted by canonical leaf ID before the shape and payload digests are updated.
// Point and pass order remain significant. This makes the digest independent of FLANN versus kd-walk
// return order without discarding multiplicity.
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

  using CellPtr = typename NeighborhoodT::value_type;
  std::vector<CellPtr> canonical(neighborhood.begin(), neighborhood.end());
  std::sort(canonical.begin(), canonical.end(), [](const CellPtr & a, const CellPtr & b) {
    return std::make_pair(a->getTraceGridOrdinal(), a->getTraceVoxelId()) <
           std::make_pair(b->getTraceGridOrdinal(), b->getTraceVoxelId());
  });

  Sha256 shape;
  static constexpr char kShapePoint[] = "NDT-SHAPE-POINT-v1";
  digest_domain(shape, kShapePoint);
  shape.u64(static_cast<std::uint64_t>(canonical.size()));
  Sha256 payload;
  static constexpr char kPayloadPoint[] = "NDT-PAYLOAD-POINT-v1";
  digest_domain(payload, kPayloadPoint);
  payload.u64(static_cast<std::uint64_t>(canonical.size()));
  for (const auto & cell : canonical) {
    const std::uint64_t grid = cell->getTraceGridOrdinal();
    const std::uint64_t voxel = static_cast<std::uint64_t>(cell->getTraceVoxelId());
    shape.u64(grid);
    shape.u64(voxel);
    payload.u64(grid);
    payload.u64(voxel);
    const Eigen::Vector3d & mean = cell->getMean();
    const Eigen::Matrix3d & icov = cell->getInverseCov();
    for (int i = 0; i < 3; ++i) payload.f64(mean(i));
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) payload.f64(icov(r, c));
  }

  Digest point_shape{};
  Digest point_payload{};
  static constexpr char kShapeChain[] = "NDT-SHAPE-CHAIN-v1";
  static constexpr char kPayloadChain[] = "NDT-PAYLOAD-CHAIN-v1";
  Digest next_shape{};
  Digest next_payload{};
  if (!shape.finish(point_shape) || !payload.finish(point_payload) ||
      !chain_digest(kShapeChain, p.shape_digest, point_shape, next_shape) ||
      !chain_digest(kPayloadChain, p.payload_digest, point_payload, next_payload)) {
    s.poisoned = true;
    return;
  }
  p.shape_digest = next_shape;
  p.payload_digest = next_payload;
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
