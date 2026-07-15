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

//! C-ABI node shell for the Autoware `autoware_ndt_scan_matcher` NDT localization core.
//!
//! This crate is the **ROS-node C ABI**: it wraps the pure-Rust engine crate
//! [`realtime_ndt_scan_matcher`] with `extern "C"` shims (the `autoware_ndt_scan_matcher_rs_*`
//! symbols the C++ package links against) plus the ROS-node callback glue. All numeric kernels and
//! the portable engine API live in `realtime_ndt_scan_matcher`; this crate only validates the FFI boundary
//! (per rust-c-ffi-safety) and marshals data to/from the engine's public Rust API.
//!
//! At node construction, C++ supplies an immutable runtime work envelope through
//! `node_handle::AwNdtParams`: maximum source points (`Pmax`), maximum published active leaves
//! (`Lmax`), and maximum iterations (`Imax`, `0..=30`). Rust validates the envelope, preallocates
//! callback scratch, and rejects oversized scans or maps without growing real-time buffers. Finding
//! a 65th neighbor is non-fatal: the first 64 are retained, the result is forced non-converged, and
//! the sensor callback emits a warning.
//!
//! The crate is always `std` (it is the ROS-node shell). The only build knob is the `ros` feature,
//! which turns on the rosidl bindgen bindings (`ros_msgs`) and the `Pose`-pointer FFI shims.
//!
//! # Layout
//!
//! - `ffi_engine`, `ffi_ndt`, `ffi_voxel_grid`, `ffi_tpe`, `ffi_covariance`, and `ffi_cov_estimate`
//!   are the C-ABI shims over the matching `realtime_ndt_scan_matcher` modules.
//! - `ffi` and `ffi_host` provide the panic-safe boundary (`catch_unwind`) and the ROS side-effects
//!   host vtable.
//! - `node`, `node_handle`, `node_map_update`, `node_align_service`, and `sensor_points` contain the
//!   ROS-node callback glue.
//! - [`ffi_ptr`] — audited C-ABI pointer helpers + guard macros (the single home for raw-pointer
//!   dereferences at the FFI boundary).

// Heap types (`Vec`) are used by several FFI shims via `alloc::`; the allocator is std's.
extern crate alloc;

// C-ABI shims over the pure engine modules.
mod ffi_cov_estimate;
mod ffi_covariance;
mod ffi_engine;
// Shared row-major matrix marshaling (chunked-slice based — no index arithmetic) for the shims.
mod ffi_matrix;
mod ffi_ndt;
mod ffi_tpe;
mod ffi_voxel_grid;

// Panic-safe C ABI boundary helpers (catch_unwind) + AwStatus/Error for the object-level FFI.
pub mod ffi;
// ROS side-effects host vtable (AwHost: clock/log/TF) — the C-ABI adapter for the portable Host seam.
pub mod ffi_host;
// Audited C-ABI pointer helpers + guard macros — the single home for raw-pointer dereferences at
// the FFI boundary.
pub mod ffi_ptr;
// ROS-gated `geometry_msgs::Pose` helper glue (bridges to the pure `realtime_ndt_scan_matcher::helper`).
#[cfg(feature = "ros")]
mod helper_ros;
// ROS-node callback glue.
pub mod node;
// Opaque object-level node handle (NdtScanMatcherRs) + AwNdtParams.
pub mod node_handle;
// ROS map-update glue: drives the portable `apply_map_update` (the `MapSource` port) from C++.
pub mod node_map_update;
// ROS align-service deterministic gate/response decisions.
pub mod node_align_service;
// ROS sensor-callback prologue (decode/TF/transform/validation → base_link cloud).
pub mod sensor_points;

// rosidl-generated geometry_msgs C structs (bindgen). ROS-node build only; the no_std
// build leaves `ros` off. bindgen output is allow-listed (its lint profile differs from ours).
#[cfg(feature = "ros")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unsafe_code,
    clippy::all,
    clippy::pedantic,
    // bindgen's layout tests use `as` casts / indexing that our restriction lints would flag.
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "bindgen-generated"
)]
mod ros_msgs {
    include!(concat!(env!("OUT_DIR"), "/ros_msgs.rs"));
}

use crate::ffi_ptr::ffi_ref;

/// Re-export of the exact [`nalgebra`] version this crate is built against (the engine crate's).
///
/// The engine's pose API is expressed in `nalgebra` matrices (`Matrix4<f32>` guesses/results,
/// `Matrix6<f64>` Hessians/covariances). Construct and read them through this re-export so the type
/// identities match the engine's — a locally pinned `nalgebra` of a different version would be a
/// distinct, incompatible type.
pub use realtime_ndt_scan_matcher::nalgebra;

/// C ABI entry point for the C++ side of `autoware_ndt_scan_matcher`.
///
/// Delegates to [`realtime_ndt_scan_matcher::add`]. Both arguments are passed by value and the return is a
/// plain `u64`, so there are no pointers or lifetimes crossing the boundary to validate.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub extern "C" fn autoware_ndt_scan_matcher_rs_add(left: u64, right: u64) -> u64 {
    realtime_ndt_scan_matcher::add(left, right)
}

/// C ABI entry point for [`realtime_ndt_scan_matcher::init_thread_pool`]. Passed by value; no pointers cross
/// the boundary. Returns `true` iff this call initialized the pool.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; no pointers cross here (value in, bool out)"
)]
#[unsafe(no_mangle)]
pub extern "C" fn autoware_ndt_scan_matcher_rs_init_thread_pool(num_threads: usize) -> bool {
    realtime_ndt_scan_matcher::init_thread_pool(num_threads)
}

/// Rotate the 3x3 position block of a 6x6 row-major pose covariance: `out = R * C * R^T`.
///
/// Delegates to [`realtime_ndt_scan_matcher::helper::rotate_covariance`].
///
/// # Safety
/// `src_cov` and `out_cov` must each point to a readable/writable array of 36 `f64`, and
/// `rot` to a readable array of 9 `f64` (row-major 3x3). All pointers must be non-null and
/// suitably aligned for `f64`. Does nothing if any pointer is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_rotate_covariance(
    src_cov: *const f64,
    rot: *const f64,
    out_cov: *mut f64,
) {
    let src = ffi_ref!(src_cov.cast::<[f64; 36]>(), else return);
    let rotation = ffi_ref!(rot.cast::<[f64; 9]>(), else return);
    let result = realtime_ndt_scan_matcher::helper::rotate_covariance(src, rotation);
    // SAFETY: `out_cov` is a valid, aligned, writable [f64; 36] per the contract, audited in ffi_ptr.
    unsafe {
        ffi_ptr::write_out(out_cov.cast::<[f64; 36]>(), result);
    }
}

/// Count the maximum consecutive direction inversions over a pose trajectory (zero-copy).
///
/// `poses` points to `num_poses` contiguous `geometry_msgs::msg::Pose`; the positions are read
/// in place (no flattening). Only `position.{x,y,z}` is used.
///
/// # Safety
/// When `num_poses > 0`, `poses` must point to `num_poses` contiguous, aligned, initialized
/// `geometry_msgs__msg__Pose` (the layout is asserted on the C++ side). Returns 0 if `poses` is
/// null or `num_poses` is 0.
#[cfg(feature = "ros")]
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_count_oscillation(
    poses: *const core::ffi::c_void,
    num_poses: usize,
) -> i32 {
    if num_poses == 0 {
        return 0;
    }
    // SAFETY: per the `# Safety` contract, `poses` points to `num_poses` valid, aligned Pose values.
    let slice = crate::ffi_ptr::ffi_slice!(poses, num_poses, ros_msgs::geometry_msgs__msg__Pose, else return 0);
    crate::helper_ros::count_oscillation_poses(slice)
}

#[cfg(test)]
// tests call the `extern "C"` shims via raw pointers (unsafe) and assert exact equality between a
// shim and the pure fn it delegates to (identical ops → bit-identical, so float_cmp is intended).
#[allow(
    unsafe_code,
    clippy::float_cmp,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(realtime_ndt_scan_matcher::add(2, 2), 4);
        assert_eq!(autoware_ndt_scan_matcher_rs_add(2, 2), 4);
    }

    #[test]
    fn rotate_covariance_ffi_matches_pure() {
        let mut src = [0.0_f64; 36];
        src[0] = 4.0;
        src[7] = 9.0;
        src[14] = 16.0;
        let rot = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let mut out = [0.0_f64; 36];
        // SAFETY: all three pointers are valid, aligned arrays of the documented length.
        unsafe {
            autoware_ndt_scan_matcher_rs_rotate_covariance(
                src.as_ptr(),
                rot.as_ptr(),
                out.as_mut_ptr(),
            );
        }
        assert_eq!(
            out,
            realtime_ndt_scan_matcher::helper::rotate_covariance(&src, &rot)
        );
    }

    #[test]
    fn rotate_covariance_ffi_null_is_noop() {
        let mut out = [1.0_f64; 36];
        let rot = [0.0_f64; 9];
        // SAFETY: a null `src_cov` must make the shim return without touching `out`.
        unsafe {
            autoware_ndt_scan_matcher_rs_rotate_covariance(
                core::ptr::null(),
                rot.as_ptr(),
                out.as_mut_ptr(),
            );
        }
        assert_eq!(out, [1.0_f64; 36]);
    }

    #[cfg(feature = "ros")]
    #[test]
    fn count_oscillation_ffi_matches_pure() {
        let xs = [0.0_f64, 1.0, 0.0, 1.0, 0.0];
        let positions: Vec<[f64; 3]> = xs.iter().map(|&x| [x, 0.0, 0.0]).collect();
        let poses: Vec<ros_msgs::geometry_msgs__msg__Pose> = xs
            .iter()
            .map(|&x| {
                let mut p = ros_msgs::geometry_msgs__msg__Pose::default();
                p.position.x = x;
                p
            })
            .collect();

        // SAFETY: `poses` is a valid contiguous array of `poses.len()` Pose values.
        let via_ffi = unsafe {
            autoware_ndt_scan_matcher_rs_count_oscillation(poses.as_ptr().cast(), poses.len())
        };
        assert_eq!(
            via_ffi,
            realtime_ndt_scan_matcher::helper::count_oscillation(&positions)
        );
        assert_eq!(via_ffi, 3);

        // SAFETY: null pointer must yield 0.
        assert_eq!(
            unsafe { autoware_ndt_scan_matcher_rs_count_oscillation(core::ptr::null(), 0) },
            0
        );
    }
}
