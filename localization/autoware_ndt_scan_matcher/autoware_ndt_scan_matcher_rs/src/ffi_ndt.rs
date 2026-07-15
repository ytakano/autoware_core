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

//! C ABI shim for the one-shot NDT align entry over [`realtime_ndt_scan_matcher::ndt::align`]. It builds a
//! target [`VoxelGridMap`] from flat inputs and marshals the result matrices row-major; the numeric
//! kernel lives in the engine crate.

use realtime_ndt_scan_matcher::ndt::{
    AlignResult, AlignWorkspace, NdtParams, Regularization, align,
};
use realtime_ndt_scan_matcher::voxel_grid::VoxelGridMap;

use crate::ffi_matrix::{
    matrix4_from_row_major, write_matrix4_row_major, write_matrix4_seq, write_matrix6_row_major,
};
use crate::ffi_ptr::{self, ffi_ref, ffi_slice};

/// Inputs to the align FFI. Point clouds are `len + *const f32` (xyz triples); `guess` is 16 `f32`
/// (row-major 4x4). `regularization_scale == 0` disables regularization.
#[repr(C)]
pub struct AwNdtAlignInput {
    pub target_xyz: *const f32,
    pub n_target: usize,
    pub source_xyz: *const f32,
    pub n_source: usize,
    pub resolution: f64,
    pub step_size: f64,
    pub trans_epsilon: f64,
    pub max_iterations: i32,
    pub outlier_ratio: f64,
    pub regularization_scale: f32,
    pub regularization_pose_x: f32,
    pub regularization_pose_y: f32,
    pub guess: *const f32,
}

/// Output buffers for the align FFI (all optional / null-skippable). `pose` 16, `hessian` 36 (row-
/// major), `transformation_array` holds up to `transforms_cap` poses (16 `f32` each); the true count
/// is written to `transforms_count`.
#[repr(C)]
pub struct AwNdtAlignOutput {
    pub pose: *mut f32,
    pub iteration_num: *mut i32,
    pub transform_probability: *mut f32,
    pub nearest_voxel_likelihood: *mut f32,
    pub hessian: *mut f64,
    pub transformation_array: *mut f32,
    pub transforms_cap: u32,
    pub transforms_count: *mut u32,
}

/// Run NDT `align` from flat C inputs (builds the target `VoxelGridMap` with the C++ defaults
/// `min_points = 6`, `eig_mult = 0.01`, `leaf_size = resolution`).
///
/// This one-shot compatibility entry point constructs unbounded map and alignment workspaces and
/// has no status return. It is intended for parity tests and tooling, not for deployment admission
/// control; use the bounded engine API when `Pmax`, `Lmax`, and `Imax` must be enforced.
///
/// # Safety
/// `input`/`output` must be valid pointers (or null → no-op). Each non-null pointer must address the
/// documented length: `target_xyz` `3*n_target` f32, `source_xyz` `3*n_source` f32, `guess` 16,
/// `pose` 16, `hessian` 36, `transformation_array` `transforms_cap*16`. No-op if `target_xyz`,
/// `source_xyz`, or `guess` is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "caps/counts cross the C ABI as u32; matrix marshaling is arithmetic-free (ffi_matrix)"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_ndt_align(
    input: *const AwNdtAlignInput,
    output: *const AwNdtAlignOutput,
) {
    let inp = ffi_ref!(input, else return);
    let outp = ffi_ref!(output, else return);
    let target = ffi_slice!(inp.target_xyz, inp.n_target, [f32; 3], else return);
    let source = ffi_slice!(inp.source_xyz, inp.n_source, [f32; 3], else return);
    let guess_arr = ffi_slice!(inp.guess, 16, else return);

    let guess = matrix4_from_row_major(guess_arr);

    let mut map = VoxelGridMap::new([inp.resolution; 3], 6, 0.01);
    map.add_target(target, 0);
    if map.create_kdtree().is_err() {
        return;
    }

    let reg = if inp.regularization_scale == 0.0 {
        None
    } else {
        Some(Regularization {
            pose_xy: [inp.regularization_pose_x, inp.regularization_pose_y],
            scale_factor: inp.regularization_scale,
        })
    };
    let params = NdtParams {
        trans_epsilon: inp.trans_epsilon,
        step_size: inp.step_size,
        resolution: inp.resolution,
        max_iterations: inp.max_iterations,
        outlier_ratio: inp.outlier_ratio,
        regularization: reg,
        // FFI defaults to the serial backend; wiring a num_threads field through the C ABI is a
        // separate follow-up (bit-for-bit serial==parallel is covered by the Rust-side test).
        num_threads: 1,
    };

    let Ok(max_iterations) = usize::try_from(inp.max_iterations) else {
        return;
    };
    let Ok(mut ws) = AlignWorkspace::try_with_capacity(source.len()) else {
        return;
    };
    let Ok(mut result) = AlignResult::try_with_capacity(max_iterations) else {
        return;
    };
    if align(&map, source, &guess, &params, &mut ws, &mut result).is_err() {
        return;
    }

    // SAFETY: each output pointer is either null (skipped) or valid for its documented length;
    // the derefs are audited in ffi_ptr (null → `opt_slice_mut` None / `write_out` false, i.e. skip).
    unsafe {
        if let Some(pose) = ffi_ptr::opt_slice_mut(outp.pose, 16) {
            write_matrix4_row_major(pose, &result.pose);
        }
        ffi_ptr::write_out(outp.iteration_num, result.iteration_num);
        ffi_ptr::write_out(outp.transform_probability, result.transform_probability);
        ffi_ptr::write_out(
            outp.nearest_voxel_likelihood,
            result.nearest_voxel_likelihood,
        );
        if let Some(h) = ffi_ptr::opt_slice_mut(outp.hessian, 36) {
            write_matrix6_row_major(h, &result.hessian);
        }
        if !outp.transformation_array.is_null() && !outp.transforms_count.is_null() {
            let cap = outp.transforms_cap as usize;
            if let Some(buf) =
                ffi_ptr::opt_slice_mut(outp.transformation_array, cap.saturating_mul(16))
            {
                write_matrix4_seq(buf, &result.transformation_array);
            }
            ffi_ptr::write_out(
                outp.transforms_count,
                result.transformation_array.len() as u32,
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::needless_range_loop,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::borrow_as_ptr,
    unsafe_code,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use realtime_ndt_scan_matcher::nalgebra::Matrix4;

    use super::*;

    // 8 points packed inside one voxel around (cx,cy,cz) for leaf_size 1.0.
    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> alloc::vec::Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    fn spread_target() -> (VoxelGridMap, alloc::vec::Vec<[f32; 3]>) {
        let mut pts: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::new();
        for &(cx, cy, cz) in &[
            (0.5_f32, 0.5_f32, 0.5_f32),
            (4.5, 0.5, 0.5),
            (0.5, 4.5, 0.5),
            (0.5, 0.5, 4.5),
            (4.5, 4.5, 4.5),
        ] {
            pts.extend(dense_cluster(cx, cy, cz));
        }
        let mut map = VoxelGridMap::new([1.0, 1.0, 1.0], 6, 0.01);
        map.add_target(&pts, 0);
        map.create_kdtree().expect("build kd-tree");
        (map, pts)
    }

    // FFI shim must equal the pure `align` on identical inputs (catches struct-layout / flat-buffer /
    // row-major marshaling bugs); null input/output must be a no-op. llvm-cov only sees Rust tests,
    // so this covers the FFI shim that the C++ differential test exercises out-of-process.
    #[test]
    fn ffi_ndt_align_matches_pure() {
        let (_, target) = spread_target();
        let t = [0.2_f32, -0.15, 0.1];
        let source: alloc::vec::Vec<[f32; 3]> = target
            .iter()
            .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
            .collect();
        let params = NdtParams {
            trans_epsilon: 0.01,
            step_size: 0.1,
            resolution: 2.0,
            max_iterations: 30,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1,
        };

        // Pure reference: same map the FFI builds internally (leaf = resolution, min 6, eig 0.01).
        let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
        map.add_target(&target, 0);
        map.create_kdtree().expect("build kd-tree");
        let mut ws = AlignWorkspace::try_with_capacity(source.len()).expect("reserve workspace");
        let mut pure = AlignResult::try_with_capacity(30).expect("reserve result");
        align(
            &map,
            &source,
            &Matrix4::identity(),
            &params,
            &mut ws,
            &mut pure,
        )
        .expect("pure align");

        // FFI call.
        let target_flat: alloc::vec::Vec<f32> =
            target.iter().flat_map(|p| p.iter().copied()).collect();
        let source_flat: alloc::vec::Vec<f32> =
            source.iter().flat_map(|p| p.iter().copied()).collect();
        let guess16: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let input = AwNdtAlignInput {
            target_xyz: target_flat.as_ptr(),
            n_target: target.len(),
            source_xyz: source_flat.as_ptr(),
            n_source: source.len(),
            resolution: 2.0,
            step_size: 0.1,
            trans_epsilon: 0.01,
            max_iterations: 30,
            outlier_ratio: 0.55,
            regularization_scale: 0.0,
            regularization_pose_x: 0.0,
            regularization_pose_y: 0.0,
            guess: guess16.as_ptr(),
        };
        let mut pose = [0.0_f32; 16];
        let mut iter = 0_i32;
        let mut tp = 0.0_f32;
        let mut nvl = 0.0_f32;
        let mut hess = [0.0_f64; 36];
        let mut transforms = [0.0_f32; 64 * 16];
        let mut count = 0_u32;
        let output = AwNdtAlignOutput {
            pose: pose.as_mut_ptr(),
            iteration_num: &mut iter,
            transform_probability: &mut tp,
            nearest_voxel_likelihood: &mut nvl,
            hessian: hess.as_mut_ptr(),
            transformation_array: transforms.as_mut_ptr(),
            transforms_cap: 64,
            transforms_count: &mut count,
        };
        // SAFETY: all pointers are valid for their documented lengths.
        unsafe { autoware_ndt_scan_matcher_rs_ndt_align(&input, &output) };

        // Same code path -> exact equality.
        assert_eq!(iter, pure.iteration_num);
        assert_eq!(tp, pure.transform_probability);
        assert_eq!(nvl, pure.nearest_voxel_likelihood);
        assert_eq!(count as usize, pure.transformation_array.len());
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(pose[(r * 4) + c], pure.pose[(r, c)]);
            }
        }
        for r in 0..6 {
            for c in 0..6 {
                assert_eq!(hess[(r * 6) + c], pure.hessian[(r, c)]);
            }
        }

        // Null input / output must be a no-op (no crash, no write).
        // SAFETY: null pointers exercise the documented no-op contract.
        unsafe {
            autoware_ndt_scan_matcher_rs_ndt_align(core::ptr::null(), &output);
            autoware_ndt_scan_matcher_rs_ndt_align(&input, core::ptr::null());
        }
    }
}
