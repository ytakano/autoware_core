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

//! Pure-Rust core of the Autoware NDT (Normal Distributions Transform) localization engine.
//!
//! This crate is **ROS-free** and `no_std`-capable (heap only, via `alloc`). It carries **no**
//! `extern "C"` — the C ABI and all ROS integration live in the sibling `autoware_ndt_scan_matcher_rs`
//! node crate, which wraps this crate's public Rust API. The numeric kernels mirror the C++
//! `Matrix4f` / `MultiVoxelGridCovariance` pipeline and are reusable under ROS, a bare-metal kernel,
//! or an async runtime.
//!
//! # Entry points
//!
//! - [`engine::NdtEngine`] — the persistent, `&self`-only NDT handle: load a target map, build the
//!   kd-tree, then align sensor clouds. This is the primary API.
//! - [`scan_matcher::ScanMatcher`] — portable node orchestration over [`engine::NdtEngine`] and the
//!   [`host`] ports (map update + single-scan match).
//! - [`ndt::align`] — the RT-critical, WCET-bounded alignment kernel the engine drives.
//! - [`tpe::TreeStructuredParzenEstimator`] — the align-service pose-search sampler.
//!
//! Pose guesses and result matrices use [`nalgebra`] types (`Matrix4<f32>`, `Matrix6<f64>`). The
//! exact `nalgebra` version this crate is built against is re-exported as [`nalgebra`] so callers can
//! construct those matrices without independently pinning a (possibly mismatched) version.
//!
//! # Cargo features
//!
//! | Feature | Default | Effect |
//! |---|---|---|
//! | `std` | yes | Host build: lock-free `ArcSwap` engine state + thread-local align scratch; the engine is `Sync`. |
//! | `parallel` | yes | rayon-backed derivative reduction (implies `std`); bit-identical to serial, a pure throughput option. |
//! | `mt` | no | Multi-core `no_std` (kernel): `awkernel_sync` mutex cells + **caller-owned** [`engine::MatchScratch`] (the implicit-scratch align API is compiled out); engine is `Sync`. Ignored when `std` is on. |
//!
//! With **no** features (`--no-default-features`) the engine is a single-core `no_std` build
//! (`RefCell` cells, engine-owned scratch, intentionally `!Sync`). See the [`engine`] module docs for
//! the full interior-mutability matrix.
//!
//! # Example
//!
//! Load a one-tile target map, build the kd-tree, and align a source cloud from an identity guess:
//!
//! ```
//! use autoware_ndt_rs::engine::NdtEngine;
//! use autoware_ndt_rs::nalgebra::Matrix4;
//!
//! // Empty engine: 2.0 m voxels; `MultiVoxelGridCovariance` defaults (min 6 points / eig 0.01).
//! let engine = NdtEngine::new(2.0, 6, 0.01);
//!
//! // Register a target map tile (id 0) and build the kd-tree over the voxel centroids.
//! let target: Vec<[f32; 3]> = (0u8..64).map(|i| [f32::from(i) * 0.05, 0.0, 0.0]).collect();
//! engine.add_target(&target, 0);
//! engine.create_kdtree();
//! assert!(engine.has_target());
//!
//! // Align a source cloud from an identity initial guess, then read the result back.
//! let source = target.clone();
//! engine.align(&Matrix4::identity(), &source);
//! let result = engine.result();
//! assert!(result.iteration_num >= 0);
//! ```

// no_std for the awkernel/no_std build; `std` (default) for dev and the ROS-node build. The
// `mt` feature makes the no_std engine multi-core-safe (awkernel_sync cells + caller-owned
// `MatchScratch`); without it the no_std build is single-core (`RefCell`, `!Sync`).
// Test builds always use std (the test harness + `Vec`/etc. need it), regardless of features.
#![cfg_attr(not(any(test, feature = "std")), no_std)]

// Heap types (Vec/Box/BTreeMap) for the engine data structures. The allocator is provided by std
// (host) or the kernel; the no_std rlib defers it to the final binary.
extern crate alloc;

pub mod convergence;
pub mod cov_estimate;
pub mod covariance;
pub mod derivatives;
pub mod engine;
pub mod helper;
// Portable node orchestration: the `Host` port traits + the `no_std` scan matcher over the engine
// (reusable on ROS / bare-metal / the Tokio example).
pub mod host;
mod kdtree;
pub mod ndt;
// no_std port of autoware_localization_util::SmartPoseBuffer (time-ordered pose interpolation buffer);
// reused by the node crate's Rust-owned pose buffers.
pub mod pose_buffer;
pub mod scan_matcher;
pub mod tpe;
pub mod transform;
pub mod voxel_grid;
// Deterministic algorithmic-cost counters for the WCET analysis (plan/ndt_wcet.md); opt-in.
#[cfg(feature = "wcet-count")]
pub mod wcet;

/// Re-export of the exact [`nalgebra`] version this crate is built against.
///
/// The engine's pose API is expressed in `nalgebra` matrices (`Matrix4<f32>` guesses/results,
/// `Matrix6<f64>` Hessians/covariances). Construct and read them through this re-export so the type
/// identities match the engine's — a locally pinned `nalgebra` of a different version would be a
/// distinct, incompatible type.
pub use nalgebra;

/// Saturating `left + right` — a trivial build/link smoke test.
///
/// # Arguments
/// * `left`, `right` — the addends; the sum saturates at [`u64::MAX`] instead of overflowing.
#[must_use]
pub fn add(left: u64, right: u64) -> u64 {
    left.saturating_add(right)
}

/// Size rayon's process-global worker pool used by the `parallel` derivative reduction.
///
/// The parallel backend runs on rayon's process-global thread pool; this sizes that pool to
/// `num_threads` workers. It is orthogonal to [`ndt::NdtParams::num_threads`], which only selects
/// serial vs parallel (`> 1`). If never called, the pool defaults to the number of logical CPUs
/// (or `RAYON_NUM_THREADS`, if set).
///
/// Best-effort and idempotent — call it **once, early** (before any align). Returns `true` iff this
/// call initialized the pool; `false` if the pool was already initialized (a previous call, or a
/// prior parallel run initialized it lazily) or `num_threads == 0`. Never panics.
///
/// # Arguments
/// * `num_threads` — desired global worker count; `0` is a no-op returning `false`.
#[cfg(feature = "parallel")]
#[must_use]
pub fn init_thread_pool(num_threads: usize) -> bool {
    num_threads != 0
        && rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(add(2, 2), 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn init_thread_pool_zero_is_noop_and_pool_is_usable() {
        // Zero threads is always a deterministic no-op.
        assert!(!init_thread_pool(0), "0 threads must be a no-op");
        // Best-effort: `sized` is true only if THIS call initialized the process-global pool — a
        // prior test in the same binary may have already done so — so assert the pool is usable
        // rather than the flag value.
        let sized = init_thread_pool(4);
        assert!(
            rayon::current_num_threads() >= 1,
            "global pool usable after init (sized={sized})"
        );
    }
}
