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

//! C ABI shims over [`realtime_ndt_scan_matcher::engine`] — the opaque `NdtEngine` handle lifecycle + the
//! sensor-callback align and covariance orchestrators. The engine and its numeric kernels live in
//! the engine crate; this module owns the handle box lifecycle and marshals pointers/matrices.
//!
//! The engine is `Sync` and exposes `&self`-only methods, so every shim forms only `&*engine`
//! (never `&mut *engine`): concurrent calls on a shared `const AwNdtEngine*` are sound without an
//! external lock. The lifecycle shims (`new`/`free`/`clone`) own/reclaim the `Box`.

use realtime_ndt_scan_matcher::engine::{
    ConvergenceParams, CovEstimationParams, NdtEngine, estimate_pose_covariance, run_align,
};
use realtime_ndt_scan_matcher::nalgebra::Matrix4;

use crate::ffi_matrix::{
    matrix4_from_row_major, write_matrix4_row_major, write_matrix4_seq, write_matrix6_row_major,
};
use crate::ffi_ndt::AwNdtAlignOutput;
use crate::ffi_ptr::{self, ffi_mut, ffi_mut_slice, ffi_ref, ffi_slice};
use crate::node::AwConvergenceVerdict;

/// Create a new engine handle (empty map). Free with `..._ndt_engine_free`.
#[expect(unsafe_code, reason = "C ABI boundary; returns an owned handle")]
#[unsafe(no_mangle)]
pub extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_new(
    resolution: f64,
    min_points: i32,
    eig_mult: f64,
) -> *mut NdtEngine {
    ffi_ptr::into_handle(NdtEngine::new(resolution, min_points, eig_mult))
}

/// # Safety
/// `engine` is a handle from `..._ndt_engine_new`/`_clone` (or null → no-op); not used afterwards.
#[expect(unsafe_code, reason = "C ABI boundary; reclaims an owned handle")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_free(engine: *mut NdtEngine) {
    // SAFETY: `engine` is a handle from `_new`/`_clone` (or null → no-op); reclaimed once in ffi_ptr.
    unsafe { ffi_ptr::free_handle(engine) };
}

/// Deep-copy an engine (the node's map-update double-buffer). Null → null.
/// # Safety
/// `engine` is a valid handle or null.
#[expect(unsafe_code, reason = "C ABI boundary; clones an owned handle")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_clone(
    engine: *const NdtEngine,
) -> *mut NdtEngine {
    // SAFETY: `engine` is a valid handle or null; deep-copied once in ffi_ptr.
    unsafe { ffi_ptr::clone_handle(engine) }
}

/// # Safety
/// `engine` is a valid handle (or null → no-op).
#[expect(unsafe_code, reason = "C ABI boundary; dereferences a shared handle")]
#[allow(
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::allow_attributes,
    reason = "num_threads is a non-negative count crossing the C ABI as i32"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_set_params(
    engine: *const NdtEngine,
    trans_epsilon: f64,
    step_size: f64,
    resolution: f64,
    max_iterations: i32,
    outlier_ratio: f64,
    num_threads: i32,
) {
    let e = ffi_ref!(engine, else return);
    e.set_params(
        trans_epsilon,
        step_size,
        resolution,
        max_iterations,
        outlier_ratio,
        num_threads.max(0) as usize,
    );
}

/// # Safety
/// `engine` is a valid handle (or null → no-op).
#[expect(unsafe_code, reason = "C ABI boundary; dereferences a shared handle")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_set_regularization(
    engine: *const NdtEngine,
    pose_x: f32,
    pose_y: f32,
    scale: f32,
) {
    let e = ffi_ref!(engine, else return);
    e.set_regularization(pose_x, pose_y, scale);
}

/// Add a map tile of `n` xyz `f32` triples (`3 * n` floats) under `id`.
/// # Safety
/// `engine` is a valid handle (or null → no-op); `points` addresses `3 * n` readable `f32`.
#[expect(unsafe_code, reason = "C ABI boundary; reads a caller-owned cloud")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_add_target(
    engine: *const NdtEngine,
    points: *const f32,
    n: usize,
    id: u64,
) {
    let e = ffi_ref!(engine, else return);
    let pts = ffi_slice!(points, n, [f32; 3], else return);
    e.add_target(pts, id);
}

/// # Safety
/// `engine` is a valid handle (or null → no-op).
#[expect(unsafe_code, reason = "C ABI boundary; dereferences a shared handle")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_remove_target(
    engine: *const NdtEngine,
    id: u64,
) {
    ffi_ref!(engine, else return).remove_target(id);
}

/// Add a target tile keyed by the cell-id bytes (`points` is `3 * n` f32; `id` is `id_len` bytes).
/// The engine owns the cell-id → tile mapping. No-op if `engine`/`points` is null.
/// # Safety
/// `engine` is a valid handle (or null → no-op); `points` addresses `3 * n` f32; `id` addresses
/// `id_len` readable bytes (or null with `id_len` 0).
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads a caller-owned cloud + id bytes"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_add_target_str(
    engine: *const NdtEngine,
    points: *const f32,
    n: usize,
    id: *const u8,
    id_len: usize,
) {
    let e = ffi_ref!(engine, else return);
    let pts = ffi_slice!(points, n, [f32; 3], else return);
    // SAFETY: caller guarantees `id_len` readable bytes (or null/0 → empty), audited in ffi_ptr.
    let id_bytes = unsafe { ffi_ptr::slice_or_empty(id, id_len) };
    e.add_target_bytes(pts, id_bytes);
}

/// Remove the tile registered under the cell-id bytes (`id` is `id_len` bytes). No-op if `engine` is
/// null or the id is unknown.
/// # Safety
/// `engine` is a valid handle (or null → no-op); `id` addresses `id_len` readable bytes (or null/0).
#[expect(unsafe_code, reason = "C ABI boundary; reads caller-owned id bytes")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_remove_target_str(
    engine: *const NdtEngine,
    id: *const u8,
    id_len: usize,
) {
    let e = ffi_ref!(engine, else return);
    // SAFETY: caller guarantees `id_len` readable bytes (or null/0 → empty), audited in ffi_ptr.
    let id_bytes = unsafe { ffi_ptr::slice_or_empty(id, id_len) };
    e.remove_target_bytes(id_bytes);
}

/// Write the current tile cell-ids (`BTreeMap`-sorted). Always writes `*out_count` (number of ids)
/// and `*out_total_len` (sum of id byte lengths). If `out_lengths`/`out_bytes` are non-null and
/// `lengths_cap`/`bytes_cap` are large enough, fills `out_lengths[0..count]` (per-id byte lengths) and
/// `out_bytes[0..total_len]` (the ids concatenated, no separators). Two-pass: call with caps 0 to
/// size, allocate, call again. No-op if `engine`/`out_count`/`out_total_len` is null.
/// # Safety
/// `engine` is a valid handle (or null → no-op). `out_count`/`out_total_len` are writable `u32` (or
/// null → no-op). `out_lengths` addresses `lengths_cap` `u32` and `out_bytes` `bytes_cap` bytes (or
/// either null → that buffer is skipped).
#[expect(
    unsafe_code,
    reason = "C ABI boundary; writes caller-owned bounded buffers"
)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "id counts/lengths cross the C ABI as u32 (bounded by the verified caps)"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_get_current_map_ids(
    engine: *const NdtEngine,
    out_lengths: *mut u32,
    lengths_cap: u32,
    out_bytes: *mut u8,
    bytes_cap: u32,
    out_count: *mut u32,
    out_total_len: *mut u32,
) {
    // All-or-nothing: engine + both counters must be non-null before anything is written.
    let e = ffi_ref!(engine, else return);
    let out_count = ffi_mut!(out_count, else return);
    let out_total_len = ffi_mut!(out_total_len, else return);
    // `map_ids` is the engine's public accessor for the (engine-private) sorted cell-id map.
    let ids_owned = e.map_ids();
    let ids: alloc::vec::Vec<&[u8]> = ids_owned.iter().map(alloc::vec::Vec::as_slice).collect();
    let total: usize = ids.iter().map(|s| s.len()).sum();
    *out_count = ids.len() as u32;
    *out_total_len = total as u32;
    let fill_lengths = !out_lengths.is_null() && lengths_cap as usize >= ids.len();
    let fill_bytes = !out_bytes.is_null() && bytes_cap as usize >= total;
    if !fill_lengths || !fill_bytes {
        return;
    }
    // Caps verified ≥ the required sizes above; both buffers non-null. The byte cursor advances by
    // splitting the remaining slice (no offset arithmetic, no panicking slice indexing);
    // `split_at_mut_checked` cannot fail because `bytes.len() == total == Σ id.len()`.
    let lens = ffi_mut_slice!(out_lengths, ids.len(), else return);
    let bytes = ffi_mut_slice!(out_bytes, total, else return);
    let mut rest: &mut [u8] = bytes;
    for (len_slot, id) in lens.iter_mut().zip(ids.iter()) {
        *len_slot = id.len() as u32;
        let Some((head, tail)) = rest.split_at_mut_checked(id.len()) else {
            return;
        };
        head.copy_from_slice(id);
        rest = tail;
    }
}

/// # Safety
/// `engine` is a valid handle (or null → no-op).
#[expect(unsafe_code, reason = "C ABI boundary; dereferences a shared handle")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_create_kdtree(
    engine: *const NdtEngine,
) {
    ffi_ref!(engine, else return).create_kdtree();
}

/// Atomically publish `src`'s map state into `dst` (the map-update commit). No-op if either is null.
/// # Safety
/// `dst`/`src` are valid handles (or either null → no-op); both reborrowed as `&*ptr` only.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; dereferences two shared handles"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_commit_from(
    dst: *const NdtEngine,
    src: *const NdtEngine,
) {
    let d = ffi_ref!(dst, else return);
    let s = ffi_ref!(src, else return);
    d.commit_from(s);
}

/// # Safety
/// `engine` is a valid handle (or null → returns false).
#[expect(unsafe_code, reason = "C ABI boundary; dereferences a shared handle")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_has_target(
    engine: *const NdtEngine,
) -> bool {
    ffi_ref!(engine, else return false).has_target()
}

/// # Safety
/// `engine` is a valid handle (or null → returns 0).
#[expect(unsafe_code, reason = "C ABI boundary; dereferences a shared handle")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_max_iterations(
    engine: *const NdtEngine,
) -> i32 {
    ffi_ref!(engine, else return 0).max_iterations()
}

/// Align `source` (`3 * n` `f32`) from `guess` (16 row-major `f32`), storing the result in the
/// thread-local scratch (retrieve with `..._get_result`).
/// # Safety
/// `engine` is a valid handle (or null → no-op); `guess` addresses 16 `f32`, `source` `3 * n` `f32`.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads caller-owned guess + cloud"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_align(
    engine: *const NdtEngine,
    guess: *const f32,
    source: *const f32,
    n: usize,
) {
    let e = ffi_ref!(engine, else return);
    let guess_buf = ffi_slice!(guess, 16, else return);
    let src = ffi_slice!(source, n, [f32; 3], else return);
    e.align(&matrix4_from_row_major(guess_buf), src);
}

/// Score `cloud` (`3 * n` `f32`) without aligning.
/// # Safety
/// `engine` is a valid handle (or null → returns 0); `cloud` addresses `3 * n` readable `f32`.
#[expect(unsafe_code, reason = "C ABI boundary; reads a caller-owned cloud")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_calc_transformation_probability(
    engine: *const NdtEngine,
    cloud: *const f32,
    n: usize,
) -> f64 {
    let e = ffi_ref!(engine, else return 0.0);
    let c = ffi_slice!(cloud, n, [f32; 3], else return 0.0);
    e.calc_transformation_probability(c)
}

/// Nearest-voxel likelihood of `cloud` (`3 * n` `f32`) without aligning.
/// # Safety
/// `engine` is a valid handle (or null → returns 0); `cloud` addresses `3 * n` readable `f32`.
#[expect(unsafe_code, reason = "C ABI boundary; reads a caller-owned cloud")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_calc_nearest_voxel_likelihood(
    engine: *const NdtEngine,
    cloud: *const f32,
    n: usize,
) -> f64 {
    let e = ffi_ref!(engine, else return 0.0);
    let c = ffi_slice!(cloud, n, [f32; 3], else return 0.0);
    e.calc_nearest_voxel_likelihood(c)
}

/// Write the last align result into the C output buffers (same shape/contract as the `ndt_align`
/// FFI; each field null-skippable).
/// # Safety
/// `engine` is a valid handle (or null → no-op); `output` is a valid `AwNdtAlignOutput` whose
/// non-null pointers address their documented lengths.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; writes caller-owned output buffers"
)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "caps/counts cross the C ABI as u32; matrix marshaling is arithmetic-free (ffi_matrix)"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_get_result(
    engine: *const NdtEngine,
    output: *const AwNdtAlignOutput,
) {
    let e = ffi_ref!(engine, else return);
    let out = ffi_ref!(output, else return);
    let result = e.result();
    // SAFETY: each non-null output pointer addresses its documented length; the derefs are audited
    // in ffi_ptr (null → skipped: `opt_slice_mut` → None, `write_out` → false without writing).
    unsafe {
        if let Some(pose) = ffi_ptr::opt_slice_mut(out.pose, 16) {
            write_matrix4_row_major(pose, &result.pose);
        }
        ffi_ptr::write_out(out.iteration_num, result.iteration_num);
        ffi_ptr::write_out(out.transform_probability, result.transform_probability);
        ffi_ptr::write_out(
            out.nearest_voxel_likelihood,
            result.nearest_voxel_likelihood,
        );
        if let Some(h) = ffi_ptr::opt_slice_mut(out.hessian, 36) {
            write_matrix6_row_major(h, &result.hessian);
        }
        if !out.transformation_array.is_null() && !out.transforms_count.is_null() {
            let cap = out.transforms_cap as usize;
            if let Some(buf) =
                ffi_ptr::opt_slice_mut(out.transformation_array, cap.saturating_mul(16))
            {
                write_matrix4_seq(buf, &result.transformation_array);
            }
            ffi_ptr::write_out(
                out.transforms_count,
                result.transformation_array.len() as u32,
            );
        }
    }
}

/// Per-point nearest-voxel score for `cloud` (`3 * n` `f32`): writes `n` scores to `out_scores`
/// (`out_scores[i] > 0` iff point `i` found a neighbor).
/// # Safety
/// `engine` is a valid handle (or null → no-op); `cloud` addresses `3 * n` `f32`, `out_scores` `n`.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads a cloud, writes per-point scores"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_calc_nearest_voxel_score_each_point(
    engine: *const NdtEngine,
    cloud: *const f32,
    n: usize,
    out_scores: *mut f32,
) {
    let e = ffi_ref!(engine, else return);
    let c = ffi_slice!(cloud, n, [f32; 3], else return);
    let out = ffi_mut_slice!(out_scores, n, else return);
    let mut scores = alloc::vec::Vec::new();
    e.nearest_voxel_score_each_point(c, &mut scores);
    out.copy_from_slice(&scores);
}

/// Write the last align's per-iteration score traces. `out_tp`/`out_nvl` receive up to `cap` `f32`
/// each; `*out_count` gets the true length (== `iteration_num + 1`).
/// # Safety
/// `engine` is a valid handle (or null → no-op); `out_tp`/`out_nvl` address `cap` `f32`, `out_count`
/// one `u32`.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; writes caller-owned score-trace buffers"
)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    clippy::allow_attributes,
    reason = "trace length crosses the C ABI as u32; k = min(len, cap) bounds the slicing"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_get_score_arrays(
    engine: *const NdtEngine,
    out_tp: *mut f32,
    out_nvl: *mut f32,
    cap: u32,
    out_count: *mut u32,
) {
    let e = ffi_ref!(engine, else return);
    let tp_out = ffi_mut_slice!(out_tp, cap as usize, else return);
    let nvl_out = ffi_mut_slice!(out_nvl, cap as usize, else return);
    let count = ffi_mut!(out_count, else return);
    let (tp, nvl) = e.score_arrays();
    let k = tp.len().min(cap as usize);
    tp_out[..k].copy_from_slice(&tp[..k]);
    nvl_out[..k].copy_from_slice(&nvl[..k]);
    *count = tp.len() as u32;
}

// --- sensor-callback align orchestrator ---
// C ABI mirror of the pure `run_align` orchestrator (align + oscillation count + convergence
// verdict). The numeric work is the engine crate's; this only marshals the flat buffers.

/// C ABI mirror of [`ConvergenceParams`] (same field order/types). Plain scalars.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[expect(
    clippy::struct_field_names,
    reason = "C ABI mirror of ConvergenceParams; the converged_param_* names are fixed by the C++ ABI"
)]
pub struct AwAlignParams {
    pub converged_param_type: i32,
    pub converged_param_transform_probability: f64,
    pub converged_param_nearest_voxel_transformation_likelihood: f64,
}

/// C ABI result of [`autoware_ndt_scan_matcher_rs_node_run_align`]. `pose` is row-major 4x4; the
/// embedded verdict is [`AwConvergenceVerdict`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AwAlignOutcome {
    pub pose: [f32; 16],
    pub transform_probability: f32,
    pub nearest_voxel_likelihood: f32,
    pub iteration_num: i32,
    pub max_iterations: i32,
    pub oscillation_num: i32,
    pub verdict: AwConvergenceVerdict,
}

/// FFI entry: align the live engine + return the scalars/verdict ([`run_align`]). No-op if any
/// pointer is null.
///
/// # Safety
/// `engine` is a valid live handle (or null → no-op). It is reborrowed as `&NdtEngine` (the engine is
/// `Sync`, so concurrent `&self` calls on a shared `const AwNdtEngine*` are sound — no external lock
/// required). Rust does not retain the pointer past the call. `guess` addresses 16 `f32`, `source`
/// `3 * n` `f32`, `params` a valid [`AwAlignParams`], `out` a writable [`AwAlignOutcome`].
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads caller-owned guess/cloud/params, validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_node_run_align(
    engine: *const NdtEngine,
    guess: *const f32,
    source: *const f32,
    n: usize,
    params: *const AwAlignParams,
    out: *mut AwAlignOutcome,
) {
    if engine.is_null() || guess.is_null() || source.is_null() || params.is_null() || out.is_null()
    {
        return;
    }
    // SAFETY: non-null per the check; caller guarantees the documented lengths + valid structs.
    let (eng, guess_buf, src, prm) = unsafe {
        (
            &*engine,
            core::slice::from_raw_parts(guess, 16),
            core::slice::from_raw_parts(source.cast::<[f32; 3]>(), n),
            &*params,
        )
    };
    let outcome = run_align(
        eng,
        &matrix4_from_row_major(guess_buf),
        src,
        &ConvergenceParams {
            converged_param_type: prm.converged_param_type,
            converged_param_transform_probability: prm.converged_param_transform_probability,
            converged_param_nearest_voxel_transformation_likelihood: prm
                .converged_param_nearest_voxel_transformation_likelihood,
        },
    );
    let mat = &outcome.pose;
    // Row-major 4x4 as an explicit literal (nalgebra `(r,c)` indexing; no array computed-index).
    let pose = [
        mat[(0, 0)],
        mat[(0, 1)],
        mat[(0, 2)],
        mat[(0, 3)],
        mat[(1, 0)],
        mat[(1, 1)],
        mat[(1, 2)],
        mat[(1, 3)],
        mat[(2, 0)],
        mat[(2, 1)],
        mat[(2, 2)],
        mat[(2, 3)],
        mat[(3, 0)],
        mat[(3, 1)],
        mat[(3, 2)],
        mat[(3, 3)],
    ];
    let vd = &outcome.verdict;
    // SAFETY: `out` is non-null per the check and a valid, writable AwAlignOutcome per the contract.
    unsafe {
        *out = AwAlignOutcome {
            pose,
            transform_probability: outcome.transform_probability,
            nearest_voxel_likelihood: outcome.nearest_voxel_likelihood,
            iteration_num: outcome.iteration_num,
            max_iterations: outcome.max_iterations,
            oscillation_num: outcome.oscillation_num,
            verdict: AwConvergenceVerdict {
                valid_param_type: vd.valid_param_type,
                is_ok_iteration_num: vd.is_ok_iteration_num,
                is_local_optimal_solution_oscillation: vd.is_local_optimal_solution_oscillation,
                is_ok_score: vd.is_ok_score,
                is_converged: vd.is_converged,
                score: vd.score,
                score_threshold: vd.score_threshold,
            },
        };
    }
}

// --- sensor-callback covariance orchestrator ---
// C ABI mirror of the pure `estimate_pose_covariance` orchestrator (rotate/dispatch/scale/adjust
// against the live engine map). The numeric work is the engine crate's; this only marshals.

/// C ABI mirror of [`CovEstimationParams`] + the align inputs. `result_pose`/`initial_pose` are
/// row-major 4x4; `hessian` row-major 6x6; `output_pose_covariance` row-major 6x6; `rot3x3` row-major.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AwCovEstimationInput {
    pub result_pose: [f32; 16],
    pub hessian: [f64; 36],
    pub initial_pose: [f32; 16],
    pub source: *const f32,
    pub n_source: usize,
    pub estimation_type: i32,
    pub offset_x: *const f64,
    pub offset_y: *const f64,
    pub n_offsets: usize,
    pub scale_factor: f64,
    pub temperature: f64,
    pub main_nvtl: f32,
    pub output_pose_covariance: [f64; 36],
    pub map_to_base_link_rot3x3: [f64; 9],
}

/// C ABI output: `ndt_covariance` (36, written) + `publish_kind` (written) + two caller-owned pose
/// buffers (`pose_cap` poses of 16 row-major floats each) filled up to cap, with the true counts
/// written back (cap+count contract, like `get_score_arrays`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AwCovEstimationOutput {
    pub ndt_covariance: [f64; 36],
    pub publish_kind: i32,
    pub multi_ndt_result_poses: *mut f32,
    pub multi_initial_poses: *mut f32,
    pub pose_cap: u32,
    pub multi_ndt_result_count: u32,
    pub multi_initial_count: u32,
}

/// FFI entry for the covariance orchestrator ([`estimate_pose_covariance`]). No-op if any pointer is
/// null.
///
/// # Safety
/// `engine` is a valid live handle (or null → no-op), reborrowed `&NdtEngine` for the call (the engine
/// is `Sync`, so concurrent `&self` calls are sound, no external lock required). `input` is a valid
/// [`AwCovEstimationInput`] whose `source` (`3*n_source` f32) and `offset_x`/`offset_y` (`n_offsets`
/// f64) are readable. `output` is a writable [`AwCovEstimationOutput`] whose pose buffers each address
/// `pose_cap * 16` writable f32 (or are null to skip). Nothing retained.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads caller-owned cloud/offsets, writes caller-owned buffers"
)]
#[allow(
    clippy::as_conversions,
    clippy::allow_attributes,
    reason = "pose_cap crosses the C ABI as u32 -> usize; matrix marshaling is arithmetic-free (ffi_matrix)"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_node_estimate_pose_covariance(
    engine: *const NdtEngine,
    input: *const AwCovEstimationInput,
    output: *mut AwCovEstimationOutput,
) {
    if engine.is_null() || input.is_null() || output.is_null() {
        return;
    }
    // SAFETY: non-null per the check; caller guarantees a valid live handle + input struct.
    let (eng, inp) = unsafe { (&*engine, &*input) };
    if inp.source.is_null() || inp.offset_x.is_null() || inp.offset_y.is_null() {
        return;
    }
    // SAFETY: caller guarantees the documented lengths.
    let (source, ox, oy) = unsafe {
        (
            core::slice::from_raw_parts(inp.source.cast::<[f32; 3]>(), inp.n_source),
            core::slice::from_raw_parts(inp.offset_x, inp.n_offsets),
            core::slice::from_raw_parts(inp.offset_y, inp.n_offsets),
        )
    };
    let result = eng.with_scratch(|scr| {
        estimate_pose_covariance(
            eng,
            &matrix4_from_row_major(&inp.result_pose),
            &inp.hessian,
            &matrix4_from_row_major(&inp.initial_pose),
            source,
            ox,
            oy,
            &CovEstimationParams {
                estimation_type: inp.estimation_type,
                scale_factor: inp.scale_factor,
                temperature: inp.temperature,
                main_nvtl: inp.main_nvtl,
                output_pose_covariance: inp.output_pose_covariance,
                map_to_base_link_rot3x3: inp.map_to_base_link_rot3x3,
            },
            scr,
        )
    });
    // SAFETY: `output` is non-null per the check and a writable AwCovEstimationOutput per the contract.
    let out = unsafe { &mut *output };
    out.ndt_covariance = result.ndt_covariance;
    out.publish_kind = result.publish_kind;
    let cap = out.pose_cap as usize;
    // SAFETY: each non-null pose buffer addresses `pose_cap * 16` writable f32 per the contract.
    unsafe {
        out.multi_ndt_result_count = fill_pose_buffer(
            out.multi_ndt_result_poses,
            cap,
            &result.multi_ndt_result_poses,
        );
        out.multi_initial_count =
            fill_pose_buffer(out.multi_initial_poses, cap, &result.multi_initial_poses);
    }
}

/// Write up to `cap` poses (16 row-major f32 each) into `buf`; return the TRUE pose count.
///
/// # Safety
/// `buf` is null (→ no write) or addresses `cap * 16` writable f32.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; writes a caller-owned bounded f32 buffer"
)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "the pose count crosses the C ABI as u32; matrix marshaling is arithmetic-free (ffi_matrix)"
)]
pub(crate) unsafe fn fill_pose_buffer(buf: *mut f32, cap: usize, poses: &[Matrix4<f32>]) -> u32 {
    let count = poses.len() as u32;
    if !buf.is_null() && cap > 0 {
        // SAFETY: per the contract, `buf` addresses `cap * 16` writable f32.
        let slice = unsafe { core::slice::from_raw_parts_mut(buf, cap.saturating_mul(16)) };
        write_matrix4_seq(slice, poses);
    }
    count
}

#[cfg(test)]
#[allow(
    unsafe_code,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::borrow_as_ptr,
    clippy::too_many_lines,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use alloc::vec::Vec;

    use realtime_ndt_scan_matcher::engine::MatchScratch;

    use super::*;

    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    fn configured(engine: &mut NdtEngine) {
        engine.set_params(0.01, 0.1, 1.0, 30, 0.55, 1);
    }

    fn two_tile_engine() -> (NdtEngine, Vec<[f32; 3]>) {
        let tile_a = dense_cluster(0.5, 0.5, 0.5);
        let tile_b = dense_cluster(4.5, 0.5, 0.5);
        let source: Vec<[f32; 3]> = tile_a
            .iter()
            .chain(tile_b.iter())
            .map(|p| [p[0] + 0.1, p[1] - 0.05, p[2]])
            .collect();
        let mut engine = NdtEngine::new(1.0, 6, 0.01);
        configured(&mut engine);
        engine.add_target(&tile_a, 0);
        engine.add_target(&tile_b, 1);
        engine.create_kdtree();
        (engine, source)
    }

    const TP_PARAMS: ConvergenceParams = ConvergenceParams {
        converged_param_type: 0, // TRANSFORM_PROBABILITY
        converged_param_transform_probability: 0.0,
        converged_param_nearest_voxel_transformation_likelihood: 0.0,
    };

    // The FFI shim writes exactly what the pure run_align computes.
    #[test]
    fn ffi_run_align_matches_pure() {
        let (engine, source) = two_tile_engine();
        let guess = Matrix4::<f32>::identity();
        let pure = run_align(&engine, &guess, &source, &TP_PARAMS);

        let (ffi_engine, ffi_source) = two_tile_engine();
        let guess16 = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let flat: Vec<f32> = ffi_source.iter().flat_map(|p| *p).collect();
        let params = AwAlignParams {
            converged_param_type: 0,
            converged_param_transform_probability: 0.0,
            converged_param_nearest_voxel_transformation_likelihood: 0.0,
        };
        let mut out = AwAlignOutcome::default();
        // SAFETY: engine valid; guess 16 f32; flat 3*n f32; params/out valid.
        unsafe {
            autoware_ndt_scan_matcher_rs_node_run_align(
                &ffi_engine,
                guess16.as_ptr(),
                flat.as_ptr(),
                ffi_source.len(),
                &params,
                &mut out,
            );
        }
        assert_eq!(out.transform_probability, pure.transform_probability);
        assert_eq!(out.nearest_voxel_likelihood, pure.nearest_voxel_likelihood);
        assert_eq!(out.iteration_num, pure.iteration_num);
        assert_eq!(out.max_iterations, pure.max_iterations);
        assert_eq!(out.oscillation_num, pure.oscillation_num);
        assert_eq!(out.verdict.is_converged, pure.verdict.is_converged);
        assert_eq!(out.pose[0], pure.pose[(0, 0)]);
        assert_eq!(out.pose[3], pure.pose[(0, 3)]);
        assert_eq!(out.pose[15], pure.pose[(3, 3)]);
    }

    // Null pointers are a no-op (no write, no panic).
    #[test]
    fn ffi_run_align_null_is_noop() {
        let mut out = AwAlignOutcome {
            iteration_num: 123,
            ..AwAlignOutcome::default()
        };
        let params = AwAlignParams {
            converged_param_type: 0,
            converged_param_transform_probability: 0.0,
            converged_param_nearest_voxel_transformation_likelihood: 0.0,
        };
        let guess16 = [0.0f32; 16];
        // SAFETY: a null engine must leave `out` untouched.
        unsafe {
            autoware_ndt_scan_matcher_rs_node_run_align(
                core::ptr::null_mut(),
                guess16.as_ptr(),
                guess16.as_ptr(),
                0,
                &params,
                &mut out,
            );
        }
        assert_eq!(out.iteration_num, 123);
    }

    // --- covariance orchestrator tests ---

    const ROT3X3_ID: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    const COV_OX: [f64; 4] = [0.3, -0.3, 0.0, 0.0];
    const COV_OY: [f64; 4] = [0.0, 0.0, 0.3, -0.3];

    fn out_cov() -> [f64; 36] {
        let mut c = [0.0_f64; 36];
        c[0] = 0.25;
        c[7] = 0.36;
        c[14] = 0.01;
        c[21] = 0.01;
        c[28] = 0.01;
        c[35] = 0.01;
        c
    }

    fn cov_params(estimation_type: i32, scale_factor: f64) -> CovEstimationParams {
        CovEstimationParams {
            estimation_type,
            scale_factor,
            temperature: 0.1,
            main_nvtl: 1.0,
            output_pose_covariance: out_cov(),
            map_to_base_link_rot3x3: ROT3X3_ID,
        }
    }

    // Align an engine once and return (engine, source, result_pose) for the cov tests.
    fn aligned_engine() -> (NdtEngine, Vec<[f32; 3]>, Matrix4<f32>) {
        let (engine, source) = two_tile_engine();
        let mut scratch = MatchScratch::new();
        engine.align_with(&Matrix4::<f32>::identity(), &source, &mut scratch);
        let pose = scratch.result().pose;
        (engine, source, pose)
    }

    #[test]
    fn ffi_estimate_cov_matches_pure() {
        let (engine, source, result_pose) = aligned_engine();
        let pure = estimate_pose_covariance(
            &engine,
            &result_pose,
            &[0.0; 36],
            &Matrix4::identity(),
            &source,
            &COV_OX,
            &COV_OY,
            &cov_params(2, 2.0),
            &mut MatchScratch::new(),
        );

        let (ffi_engine, ffi_source) = two_tile_engine();
        let mut pose16 = [0.0_f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                pose16[(r * 4) + c] = result_pose[(r, c)];
            }
        }
        let id16 = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let flat: Vec<f32> = ffi_source.iter().flat_map(|p| *p).collect();
        let input = AwCovEstimationInput {
            result_pose: pose16,
            hessian: [0.0; 36],
            initial_pose: id16,
            source: flat.as_ptr(),
            n_source: ffi_source.len(),
            estimation_type: 2,
            offset_x: COV_OX.as_ptr(),
            offset_y: COV_OY.as_ptr(),
            n_offsets: COV_OX.len(),
            scale_factor: 2.0,
            temperature: 0.1,
            main_nvtl: 1.0,
            output_pose_covariance: out_cov(),
            map_to_base_link_rot3x3: ROT3X3_ID,
        };
        let cap = (COV_OX.len() + 1) as u32;
        let mut res_buf = vec![0.0_f32; (cap as usize) * 16];
        let mut init_buf = vec![0.0_f32; (cap as usize) * 16];
        let mut out = AwCovEstimationOutput {
            ndt_covariance: [0.0; 36],
            publish_kind: -1,
            multi_ndt_result_poses: res_buf.as_mut_ptr(),
            multi_initial_poses: init_buf.as_mut_ptr(),
            pose_cap: cap,
            multi_ndt_result_count: 0,
            multi_initial_count: 0,
        };
        // SAFETY: engine valid; input pointers valid for their lengths; out buffers sized cap*16.
        unsafe {
            autoware_ndt_scan_matcher_rs_node_estimate_pose_covariance(
                &ffi_engine,
                &input,
                &mut out,
            );
        }
        assert_eq!(out.ndt_covariance, pure.ndt_covariance);
        assert_eq!(out.publish_kind, pure.publish_kind);
        assert_eq!(
            out.multi_ndt_result_count as usize,
            pure.multi_ndt_result_poses.len()
        );
        assert_eq!(
            out.multi_initial_count as usize,
            pure.multi_initial_poses.len()
        );
        // First marshaled result pose == pure's first result pose (row-major).
        assert_eq!(res_buf[0], pure.multi_ndt_result_poses[0][(0, 0)]);
        assert_eq!(res_buf[3], pure.multi_ndt_result_poses[0][(0, 3)]);
    }

    #[test]
    fn ffi_estimate_cov_null_is_noop() {
        let mut out = AwCovEstimationOutput {
            ndt_covariance: [0.0; 36],
            publish_kind: 42,
            multi_ndt_result_poses: core::ptr::null_mut(),
            multi_initial_poses: core::ptr::null_mut(),
            pose_cap: 0,
            multi_ndt_result_count: 0,
            multi_initial_count: 0,
        };
        // SAFETY: a null engine must leave `out` untouched.
        unsafe {
            autoware_ndt_scan_matcher_rs_node_estimate_pose_covariance(
                core::ptr::null_mut(),
                core::ptr::null(),
                &mut out,
            );
        }
        assert_eq!(out.publish_kind, 42);
    }

    // FFI handle round-trip == the pure engine API (marshaling check).
    #[test]
    fn ffi_engine_matches_pure() {
        let tile = dense_cluster(0.5, 0.5, 0.5);
        let flat: Vec<f32> = tile.iter().flatten().copied().collect();
        let source: Vec<[f32; 3]> = tile.iter().map(|p| [p[0] + 0.1, p[1], p[2]]).collect();
        let src_flat: Vec<f32> = source.iter().flatten().copied().collect();
        let guess16 = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        // pure
        let mut pure = NdtEngine::new(1.0, 6, 0.01);
        configured(&mut pure);
        pure.add_target(&tile, 0);
        pure.create_kdtree();
        pure.align(&matrix4_from_row_major(&guess16), &source);
        let want = pure.result().clone();
        let want_tp_score = pure.calc_transformation_probability(&source);
        let want_nvl_score = pure.calc_nearest_voxel_likelihood(&source);
        let want_maxit = pure.max_iterations();
        let mut want_scores = Vec::new();
        pure.nearest_voxel_score_each_point(&source, &mut want_scores);
        let (want_tp_arr, want_nvl_arr) = {
            let (a, b) = pure.score_arrays();
            (a.clone(), b.clone())
        };

        // FFI
        // SAFETY: every pointer below addresses its documented length.
        unsafe {
            let e = autoware_ndt_scan_matcher_rs_ndt_engine_new(1.0, 6, 0.01);
            autoware_ndt_scan_matcher_rs_ndt_engine_set_params(e, 0.01, 0.1, 1.0, 30, 0.55, 1);
            autoware_ndt_scan_matcher_rs_ndt_engine_add_target(e, flat.as_ptr(), tile.len(), 0);
            autoware_ndt_scan_matcher_rs_ndt_engine_create_kdtree(e);
            assert!(autoware_ndt_scan_matcher_rs_ndt_engine_has_target(e));
            autoware_ndt_scan_matcher_rs_ndt_engine_align(
                e,
                guess16.as_ptr(),
                src_flat.as_ptr(),
                source.len(),
            );

            let mut pose = [0.0_f32; 16];
            let mut iter = 0_i32;
            let mut tp = 0.0_f32;
            let mut nvl = 0.0_f32;
            let mut hessian = [0.0_f64; 36];
            let mut count = 0_u32;
            let out = AwNdtAlignOutput {
                pose: pose.as_mut_ptr(),
                iteration_num: &mut iter,
                transform_probability: &mut tp,
                nearest_voxel_likelihood: &mut nvl,
                hessian: hessian.as_mut_ptr(),
                transformation_array: core::ptr::null_mut(),
                transforms_cap: 0,
                transforms_count: &mut count,
            };
            autoware_ndt_scan_matcher_rs_ndt_engine_get_result(e, &out);

            assert_eq!(iter, want.iteration_num);
            assert_eq!(tp, want.transform_probability);
            assert_eq!(nvl, want.nearest_voxel_likelihood);
            for r in 0..4 {
                for c in 0..4 {
                    assert_eq!(pose[(r * 4) + c], want.pose[(r, c)]);
                }
            }

            // Remaining shims marshal to the same values as the pure API.
            assert_eq!(
                autoware_ndt_scan_matcher_rs_ndt_engine_max_iterations(e),
                want_maxit
            );
            assert_eq!(
                autoware_ndt_scan_matcher_rs_ndt_engine_calc_transformation_probability(
                    e,
                    src_flat.as_ptr(),
                    source.len()
                ),
                want_tp_score
            );
            assert_eq!(
                autoware_ndt_scan_matcher_rs_ndt_engine_calc_nearest_voxel_likelihood(
                    e,
                    src_flat.as_ptr(),
                    source.len()
                ),
                want_nvl_score
            );
            // set_regularization shim (no-op with scale 0); clone is an independent live handle.
            autoware_ndt_scan_matcher_rs_ndt_engine_set_regularization(e, 0.0, 0.0, 0.0);
            let e2 = autoware_ndt_scan_matcher_rs_ndt_engine_clone(e);
            assert!(autoware_ndt_scan_matcher_rs_ndt_engine_has_target(e2));
            autoware_ndt_scan_matcher_rs_ndt_engine_free(e2);
            // remove the only tile -> no target left.
            // per-point score + per-iteration score arrays marshal to the pure values.
            let mut ffi_scores = vec![0.0_f32; source.len()];
            autoware_ndt_scan_matcher_rs_ndt_engine_calc_nearest_voxel_score_each_point(
                e,
                src_flat.as_ptr(),
                source.len(),
                ffi_scores.as_mut_ptr(),
            );
            assert_eq!(ffi_scores, want_scores);

            let cap = (want_tp_arr.len() as u32) + 4;
            let mut tp_arr = vec![0.0_f32; cap as usize];
            let mut nvl_arr = vec![0.0_f32; cap as usize];
            let mut scount = 0_u32;
            autoware_ndt_scan_matcher_rs_ndt_engine_get_score_arrays(
                e,
                tp_arr.as_mut_ptr(),
                nvl_arr.as_mut_ptr(),
                cap,
                &mut scount,
            );
            assert_eq!(scount as usize, want_tp_arr.len());
            assert_eq!(&tp_arr[..scount as usize], want_tp_arr.as_slice());
            assert_eq!(&nvl_arr[..scount as usize], want_nvl_arr.as_slice());

            autoware_ndt_scan_matcher_rs_ndt_engine_remove_target(e, 0);
            assert!(!autoware_ndt_scan_matcher_rs_ndt_engine_has_target(e));
            autoware_ndt_scan_matcher_rs_ndt_engine_free(e);
        }
    }

    #[test]
    fn ffi_string_id_map_roundtrip() {
        let tile = dense_cluster(0.5, 0.5, 0.5);
        let flat: Vec<f32> = tile.iter().flat_map(|p| *p).collect();
        let mut engine = NdtEngine::new(1.0, 6, 0.01);
        configured(&mut engine);
        // SAFETY: valid engine; flat is 3*n f32; ids are valid byte slices.
        unsafe {
            autoware_ndt_scan_matcher_rs_ndt_engine_add_target_str(
                &engine,
                flat.as_ptr(),
                tile.len(),
                b"aa".as_ptr(),
                2,
            );
            autoware_ndt_scan_matcher_rs_ndt_engine_add_target_str(
                &engine,
                flat.as_ptr(),
                tile.len(),
                b"bbb".as_ptr(),
                3,
            );
        }
        // Pass 1: sizes.
        let mut count = 0u32;
        let mut total = 0u32;
        // SAFETY: count/total are writable; the buffers are null (size-only pass).
        unsafe {
            autoware_ndt_scan_matcher_rs_ndt_engine_get_current_map_ids(
                &engine,
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                0,
                &mut count,
                &mut total,
            );
        }
        assert_eq!(count, 2);
        assert_eq!(total, 5); // "aa" + "bbb"

        // Pass 2: fill.
        let mut lengths = vec![0u32; count as usize];
        let mut bytes = vec![0u8; total as usize];
        // SAFETY: buffers sized to count/total per pass 1.
        unsafe {
            autoware_ndt_scan_matcher_rs_ndt_engine_get_current_map_ids(
                &engine,
                lengths.as_mut_ptr(),
                count,
                bytes.as_mut_ptr(),
                total,
                &mut count,
                &mut total,
            );
        }
        assert_eq!(lengths, vec![2, 3]); // sorted: "aa", "bbb"
        assert_eq!(&bytes[0..2], b"aa");
        assert_eq!(&bytes[2..5], b"bbb");

        // SAFETY: null engine is a no-op (count untouched).
        let mut c2 = 99u32;
        let mut t2 = 99u32;
        unsafe {
            autoware_ndt_scan_matcher_rs_ndt_engine_get_current_map_ids(
                core::ptr::null(),
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                0,
                &mut c2,
                &mut t2,
            );
        }
        assert_eq!(c2, 99);
    }
}

/// One per-pass record of the C1 trace certificate (`wcet-trace` analysis builds only).
/// Field semantics mirror `realtime_ndt_scan_matcher::wcet::PassTrace`: structural work
/// (`points`, `neighbors`, this engine's own `kd_nodes`), neighbor shape/payload digests,
/// and the pass-final score/gradient/Hessian bit hashes.
#[cfg(feature = "wcet-trace")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AwNdtPassTrace {
    pub points: u64,
    pub neighbors: u64,
    pub kd_nodes: u64,
    pub shape_digest: [u8; 32],
    pub payload_digest: [u8; 32],
    pub score_bits: u64,
    pub grad_hash: u64,
    pub hess_hash: u64,
}

/// Copy the last align's per-pass trace into `out` (up to `cap` records) and write the TOTAL
/// pass count through `total_out` (it may exceed `cap`; the stored records are the first
/// `min(total, MAX_TRACE_PASSES)`). Returns `false` when the trace is unavailable (null handle)
/// or poisoned (the parallel backend ran, so per-pass order is not certificate-valid).
///
/// # Safety
/// `engine` is a valid handle (or null → returns `false`); `out` addresses `cap` records or is
/// null; `total_out` is a valid pointer or null.
#[cfg(feature = "wcet-trace")]
#[expect(
    unsafe_code,
    reason = "C ABI boundary; writes caller-owned output buffers"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_engine_get_trace(
    engine: *const NdtEngine,
    out: *mut AwNdtPassTrace,
    cap: u64,
    total_out: *mut u64,
) -> bool {
    use realtime_ndt_scan_matcher::wcet::MAX_TRACE_PASSES;

    let e = ffi_ref!(engine, else return false);
    let trace = e.result().trace;
    let total = u64::try_from(trace.len).unwrap_or(u64::MAX);
    let stored = trace.len.min(MAX_TRACE_PASSES);
    // SAFETY: each non-null output pointer addresses its documented length; the derefs are
    // audited in ffi_ptr (null → skipped).
    unsafe {
        ffi_ptr::write_out(total_out, total);
        if let Some(slice) = ffi_ptr::opt_slice_mut(out, usize::try_from(cap).unwrap_or(0)) {
            for (dst, src) in slice.iter_mut().zip(trace.passes.iter().take(stored)) {
                *dst = AwNdtPassTrace {
                    points: src.points,
                    neighbors: src.neighbors,
                    kd_nodes: src.kd_nodes,
                    shape_digest: src.shape_digest,
                    payload_digest: src.payload_digest,
                    score_bits: src.score_bits,
                    grad_hash: src.grad_hash,
                    hess_hash: src.hess_hash,
                };
            }
        }
    }
    !trace.poisoned
}
