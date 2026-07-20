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
//! - `ffi_engine`, `ffi_voxel_grid`, `ffi_tpe`, and `ffi_covariance`
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
mod ffi_covariance;
mod ffi_engine;
// Shared row-major matrix marshaling (chunked-slice based — no index arithmetic) for the shims.
mod ffi_matrix;
mod ffi_tpe;
mod ffi_voxel_grid;

// Panic-safe C ABI boundary helpers (catch_unwind) + AwStatus/Error for the object-level FFI.
pub mod ffi;
// ROS side-effects host vtable (AwHost: clock/log/TF) — the C-ABI adapter for the portable Host seam.
pub mod ffi_host;
// Audited C-ABI pointer helpers + guard macros — the single home for raw-pointer dereferences at
// the FFI boundary.
pub mod ffi_ptr;
// ROS-node callback glue.
pub mod node;
// Opaque object-level node handle (NdtScanMatcherRs) + AwNdtParams.
pub mod node_handle;
// Status-returning staged map-update bridge used by the Rust backend.
pub mod node_map_update;
// ROS map-update glue: drives the portable `apply_map_update` (the `MapSource` port) from C++.
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

/// Re-export of the exact [`nalgebra`] version this crate is built against (the engine crate's).
///
/// The engine's pose API is expressed in `nalgebra` matrices (`Matrix4<f32>` guesses/results,
/// `Matrix6<f64>` Hessians/covariances). Construct and read them through this re-export so the type
/// identities match the engine's — a locally pinned `nalgebra` of a different version would be a
/// distinct, incompatible type.
pub use realtime_ndt_scan_matcher::nalgebra;

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
