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

//! C ABI shims over [`autoware_ndt_rs::cov_estimate`] — the multi-NDT covariance estimators driven
//! from flat C buffers. The estimators themselves (and the pure helpers they build on) live in the
//! engine crate; this module only builds the target map / marshals the pointers.

use autoware_ndt_rs::cov_estimate::{
    estimate_xy_covariance_by_multi_ndt, estimate_xy_covariance_by_multi_ndt_score,
    propose_poses_to_search,
};
use autoware_ndt_rs::ndt::{AlignResult, AlignWorkspace, NdtParams};
use autoware_ndt_rs::voxel_grid::VoxelGridMap;

use crate::ffi_matrix::{matrix4_from_row_major, write_matrix4_seq};
use crate::ffi_ptr::{ffi_mut_slice, ffi_ref, ffi_slice};

// ---- C ABI shims (pointers validated per rust-c-ffi-safety) ----

/// Flat inputs for the multi-NDT covariance FFI. All clouds are `[f32; 3]` triples (`3 * n` floats);
/// `main_pose` is 16 row-major `f32`; `offset_x`/`offset_y` are `n_offsets` `f64` each. `main_nvtl` /
/// `temperature` are only read by the `_score` variant.
#[repr(C)]
pub struct AwMultiNdtCovInput {
    pub target_xyz: *const f32,
    pub n_target: usize,
    pub source_xyz: *const f32,
    pub n_source: usize,
    pub main_pose: *const f32,
    pub offset_x: *const f64,
    pub offset_y: *const f64,
    pub n_offsets: usize,
    pub resolution: f64,
    pub step_size: f64,
    pub trans_epsilon: f64,
    pub max_iterations: i32,
    pub outlier_ratio: f64,
    pub main_nvtl: f64,
    pub temperature: f64,
}

/// Narrow an f64 to f32 — the main result's nearest-voxel likelihood is f32 by contract.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "deliberate f64->f32 narrowing at the C ABI boundary"
)]
fn cast_f64_to_f32(x: f64) -> f32 {
    x as f32
}

/// # Safety
/// `input` is a valid `AwMultiNdtCovInput` (or null → no-op) whose pointers address the documented
/// lengths; `out_mean` points to 2 writable `f64`, `out_cov` to 4 (row-major 2x2). No-op if any
/// required pointer is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_estimate_cov_multi_ndt(
    input: *const AwMultiNdtCovInput,
    out_mean: *mut f64,
    out_cov: *mut f64,
) {
    let inp = ffi_ref!(input, else return);
    if out_mean.is_null() || out_cov.is_null() {
        return;
    }
    let target = ffi_slice!(inp.target_xyz, inp.n_target, [f32; 3], else return);
    let source = ffi_slice!(inp.source_xyz, inp.n_source, [f32; 3], else return);
    let main_buf = ffi_slice!(inp.main_pose, 16, else return);
    let offset_x = ffi_slice!(inp.offset_x, inp.n_offsets, else return);
    let offset_y = ffi_slice!(inp.offset_y, inp.n_offsets, else return);
    let main_pose = matrix4_from_row_major(main_buf);
    let mut map = VoxelGridMap::new([inp.resolution; 3], 6, 0.01);
    map.add_target(target, 0);
    map.create_kdtree();
    let params = NdtParams {
        trans_epsilon: inp.trans_epsilon,
        step_size: inp.step_size,
        resolution: inp.resolution,
        max_iterations: inp.max_iterations,
        outlier_ratio: inp.outlier_ratio,
        regularization: None,
        num_threads: 1,
    };
    let poses = propose_poses_to_search(&main_pose, offset_x, offset_y);
    let main_ndt = AlignResult {
        pose: main_pose,
        ..AlignResult::default()
    };
    let mut ws = AlignWorkspace::new();
    let result =
        estimate_xy_covariance_by_multi_ndt(&main_ndt, &poses, &map, source, &params, &mut ws);
    ffi_mut_slice!(out_mean, 2, else return).copy_from_slice(&result.mean);
    ffi_mut_slice!(out_cov, 4, else return).copy_from_slice(&result.covariance);
}

/// # Safety
/// As [`autoware_ndt_scan_matcher_rs_estimate_cov_multi_ndt`]; additionally reads `main_nvtl` and
/// `temperature`.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_estimate_cov_multi_ndt_score(
    input: *const AwMultiNdtCovInput,
    out_mean: *mut f64,
    out_cov: *mut f64,
) {
    let inp = ffi_ref!(input, else return);
    if out_mean.is_null() || out_cov.is_null() {
        return;
    }
    let target = ffi_slice!(inp.target_xyz, inp.n_target, [f32; 3], else return);
    let source = ffi_slice!(inp.source_xyz, inp.n_source, [f32; 3], else return);
    let main_buf = ffi_slice!(inp.main_pose, 16, else return);
    let offset_x = ffi_slice!(inp.offset_x, inp.n_offsets, else return);
    let offset_y = ffi_slice!(inp.offset_y, inp.n_offsets, else return);
    let main_pose = matrix4_from_row_major(main_buf);
    let mut map = VoxelGridMap::new([inp.resolution; 3], 6, 0.01);
    map.add_target(target, 0);
    map.create_kdtree();
    let params = NdtParams {
        trans_epsilon: inp.trans_epsilon,
        step_size: inp.step_size,
        resolution: inp.resolution,
        max_iterations: inp.max_iterations,
        outlier_ratio: inp.outlier_ratio,
        regularization: None,
        num_threads: 1,
    };
    let poses = propose_poses_to_search(&main_pose, offset_x, offset_y);
    let main_ndt = AlignResult {
        pose: main_pose,
        nearest_voxel_likelihood: cast_f64_to_f32(inp.main_nvtl),
        ..AlignResult::default()
    };
    let mut ws = AlignWorkspace::new();
    let result = estimate_xy_covariance_by_multi_ndt_score(
        &main_ndt,
        &poses,
        &map,
        source,
        &params,
        inp.temperature,
        &mut ws,
    );
    ffi_mut_slice!(out_mean, 2, else return).copy_from_slice(&result.mean);
    ffi_mut_slice!(out_cov, 4, else return).copy_from_slice(&result.covariance);
}

/// # Safety
/// `main_pose` points to 16 `f32`; `offset_x`/`offset_y` to `n` `f64`; `out_poses` to `n * 16`
/// writable `f32` (row-major poses). No-op if any pointer is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_propose_poses_to_search(
    main_pose: *const f32,
    offset_x: *const f64,
    offset_y: *const f64,
    n: usize,
    out_poses: *mut f32,
) {
    let main_buf = ffi_slice!(main_pose, 16, else return);
    let ox = ffi_slice!(offset_x, n, else return);
    let oy = ffi_slice!(offset_y, n, else return);
    let out = ffi_mut_slice!(out_poses, n.saturating_mul(16), else return);
    let poses = propose_poses_to_search(&matrix4_from_row_major(main_buf), ox, oy);
    write_matrix4_seq(out, &poses);
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
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use autoware_ndt_rs::nalgebra::Matrix4;

    use super::*;

    fn flatten(pts: &[[f32; 3]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(pts.len() * 3);
        for p in pts {
            out.extend_from_slice(p);
        }
        out
    }

    fn pose16(m: &Matrix4<f32>) -> [f32; 16] {
        let mut out = [0.0_f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                out[(r * 4) + c] = m[(r, c)];
            }
        }
        out
    }

    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    fn spread_map() -> (VoxelGridMap, Vec<[f32; 3]>) {
        let mut pts: Vec<[f32; 3]> = Vec::new();
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
        map.create_kdtree();
        (map, pts)
    }

    fn params() -> NdtParams {
        NdtParams {
            trans_epsilon: 1e-3,
            step_size: 0.2,
            resolution: 1.0,
            max_iterations: 50,
            outlier_ratio: 0.55,
            regularization: None,
            num_threads: 1,
        }
    }

    // FFI == pure: the C-ABI shims marshal flat buffers into the same result as the Rust functions
    // (the C++ differential exercises them end-to-end; this pins the marshaling Rust-side).
    #[test]
    fn ffi_propose_matches_pure() {
        let mut main = Matrix4::<f32>::identity();
        main[(0, 3)] = 1.0;
        main[(1, 3)] = 2.0;
        let ox = [0.3, -0.4, 0.1];
        let oy = [0.1, 0.2, -0.3];
        let pure = propose_poses_to_search(&main, &ox, &oy);

        let main16 = pose16(&main);
        let mut rs = vec![0.0_f32; ox.len() * 16];
        // SAFETY: pointers address the documented lengths (16 / n / n / n*16).
        unsafe {
            autoware_ndt_scan_matcher_rs_propose_poses_to_search(
                main16.as_ptr(),
                ox.as_ptr(),
                oy.as_ptr(),
                ox.len(),
                rs.as_mut_ptr(),
            );
        }
        for (i, p) in pure.iter().enumerate() {
            for r in 0..4 {
                for c in 0..4 {
                    assert_eq!(p[(r, c)], rs[(i * 16) + (r * 4) + c], "pose[{i}]({r},{c})");
                }
            }
        }
    }

    #[test]
    fn ffi_multi_ndt_matches_pure() {
        let (map, pts) = spread_map();
        let target = flatten(&pts);
        let main_pose = Matrix4::<f32>::identity();
        let main_ndt = AlignResult {
            pose: main_pose,
            ..AlignResult::default()
        };
        let ox = [0.3, -0.3, 0.0, 0.0];
        let oy = [0.0, 0.0, 0.3, -0.3];
        let poses = propose_poses_to_search(&main_pose, &ox, &oy);
        let mut ws = AlignWorkspace::new();
        let pure =
            estimate_xy_covariance_by_multi_ndt(&main_ndt, &poses, &map, &pts, &params(), &mut ws);

        let main16 = pose16(&main_pose);
        let input = AwMultiNdtCovInput {
            target_xyz: target.as_ptr(),
            n_target: pts.len(),
            source_xyz: target.as_ptr(),
            n_source: pts.len(),
            main_pose: main16.as_ptr(),
            offset_x: ox.as_ptr(),
            offset_y: oy.as_ptr(),
            n_offsets: ox.len(),
            resolution: 1.0,
            step_size: 0.2,
            trans_epsilon: 1e-3,
            max_iterations: 50,
            outlier_ratio: 0.55,
            main_nvtl: 0.0,
            temperature: 0.1,
        };
        let mut mean = [0.0_f64; 2];
        let mut cov = [0.0_f64; 4];
        // SAFETY: input is valid; mean/cov have 2/4 writable f64.
        unsafe {
            autoware_ndt_scan_matcher_rs_estimate_cov_multi_ndt(
                &input,
                mean.as_mut_ptr(),
                cov.as_mut_ptr(),
            );
        }
        assert_eq!(mean, pure.mean);
        assert_eq!(cov, pure.covariance);
    }

    #[test]
    fn ffi_multi_ndt_score_matches_pure() {
        let (map, pts) = spread_map();
        let target = flatten(&pts);
        let main_pose = Matrix4::<f32>::identity();
        let main_ndt = AlignResult {
            pose: main_pose,
            nearest_voxel_likelihood: 2.5,
            ..AlignResult::default()
        };
        let ox = [0.3, -0.3, 0.0, 0.0];
        let oy = [0.0, 0.0, 0.3, -0.3];
        let poses = propose_poses_to_search(&main_pose, &ox, &oy);
        let mut ws = AlignWorkspace::new();
        let pure = estimate_xy_covariance_by_multi_ndt_score(
            &main_ndt,
            &poses,
            &map,
            &pts,
            &params(),
            0.05,
            &mut ws,
        );

        let main16 = pose16(&main_pose);
        let input = AwMultiNdtCovInput {
            target_xyz: target.as_ptr(),
            n_target: pts.len(),
            source_xyz: target.as_ptr(),
            n_source: pts.len(),
            main_pose: main16.as_ptr(),
            offset_x: ox.as_ptr(),
            offset_y: oy.as_ptr(),
            n_offsets: ox.len(),
            resolution: 1.0,
            step_size: 0.2,
            trans_epsilon: 1e-3,
            max_iterations: 50,
            outlier_ratio: 0.55,
            main_nvtl: 2.5,
            temperature: 0.05,
        };
        let mut mean = [0.0_f64; 2];
        let mut cov = [0.0_f64; 4];
        // SAFETY: input is valid; mean/cov have 2/4 writable f64.
        unsafe {
            autoware_ndt_scan_matcher_rs_estimate_cov_multi_ndt_score(
                &input,
                mean.as_mut_ptr(),
                cov.as_mut_ptr(),
            );
        }
        assert_eq!(mean, pure.mean);
        assert_eq!(cov, pure.covariance);
    }
}
