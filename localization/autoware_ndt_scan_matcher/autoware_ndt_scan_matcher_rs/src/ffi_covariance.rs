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

//! C ABI shims over [`realtime_ndt_scan_matcher::covariance`]. These wrap the pure covariance helpers with the
//! flat-buffer / row-major marshaling the C++ side expects. The numeric logic lives in the engine
//! crate; this module only validates pointers (per rust-c-ffi-safety) and marshals.

use realtime_ndt_scan_matcher::covariance::{
    adjust_diagonal_covariance, calc_weight_vec, calculate_weighted_mean_and_cov,
    laplace_xy_covariance, rotate_covariance_to_base_link, rotate_covariance_to_map,
};

use crate::ffi_ptr::{self, ffi_mut_slice, ffi_ref, ffi_slice};

// ---- C ABI shims (pointers validated per rust-c-ffi-safety) ----

/// # Safety
/// `scores`/`out` point to `n` readable/writable `f64`. No-op if either is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_calc_weight_vec(
    scores: *const f64,
    n: usize,
    temperature: f64,
    out: *mut f64,
) {
    let scores = ffi_slice!(scores, n, else return);
    let out = ffi_mut_slice!(out, n, else return);
    calc_weight_vec(scores, temperature, out);
}

/// # Safety
/// `poses2d` points to `2*n` `f64`, `weights` to `n` `f64`, `mean_out` to 2, `cov_out` to 4.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_calculate_weighted_mean_and_cov(
    poses2d: *const f64,
    weights: *const f64,
    n: usize,
    mean_out: *mut f64,
    cov_out: *mut f64,
) {
    if mean_out.is_null() || cov_out.is_null() {
        return;
    }
    let Some(poses_len) = n.checked_mul(2) else {
        return;
    };
    let poses = ffi_slice!(poses2d, poses_len, else return);
    let weights = ffi_slice!(weights, n, else return);
    let (mean, cov) = calculate_weighted_mean_and_cov(poses, weights);
    // SAFETY: mean_out has 2, cov_out has 4 f64 per the contract, audited in ffi_ptr.
    unsafe {
        ffi_ptr::write_out(mean_out.cast::<[f64; 2]>(), mean);
        ffi_ptr::write_out(cov_out.cast::<[f64; 4]>(), cov);
    }
}

/// # Safety
/// `hessian` points to 36 `f64` (row-major 6x6); `cov_out` to 4.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_laplace_xy_covariance(
    hessian: *const f64,
    cov_out: *mut f64,
) {
    let hessian = ffi_ref!(hessian.cast::<[f64; 36]>(), else return);
    let result = laplace_xy_covariance(hessian);
    // SAFETY: `cov_out` has 4 f64 per the contract, audited in ffi_ptr.
    unsafe { ffi_ptr::write_out(cov_out.cast::<[f64; 4]>(), result) };
}

/// # Safety
/// `cov`/`rot` point to 4 `f64` (row-major 2x2), `out` to 4. No-op if any is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_rotate_covariance_to_base_link(
    cov: *const f64,
    rot: *const f64,
    out: *mut f64,
) {
    let cov = ffi_ref!(cov.cast::<[f64; 4]>(), else return);
    let rot = ffi_ref!(rot.cast::<[f64; 4]>(), else return);
    let result = rotate_covariance_to_base_link(cov, rot);
    // SAFETY: `out` has 4 f64 per the contract, audited in ffi_ptr.
    unsafe { ffi_ptr::write_out(out.cast::<[f64; 4]>(), result) };
}

/// # Safety
/// `cov`/`rot` point to 4 `f64` (row-major 2x2), `out` to 4. No-op if any is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_rotate_covariance_to_map(
    cov: *const f64,
    rot: *const f64,
    out: *mut f64,
) {
    let cov = ffi_ref!(cov.cast::<[f64; 4]>(), else return);
    let rot = ffi_ref!(rot.cast::<[f64; 4]>(), else return);
    let result = rotate_covariance_to_map(cov, rot);
    // SAFETY: `out` has 4 f64 per the contract, audited in ffi_ptr.
    unsafe { ffi_ptr::write_out(out.cast::<[f64; 4]>(), result) };
}

/// # Safety
/// `cov`/`rot` point to 4 `f64` (row-major 2x2), `out` to 4. No-op if any is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_adjust_diagonal_covariance(
    cov: *const f64,
    rot: *const f64,
    fixed_cov00: f64,
    fixed_cov11: f64,
    out: *mut f64,
) {
    let cov = ffi_ref!(cov.cast::<[f64; 4]>(), else return);
    let rot = ffi_ref!(rot.cast::<[f64; 4]>(), else return);
    let result = adjust_diagonal_covariance(cov, rot, fixed_cov00, fixed_cov11);
    // SAFETY: `out` has 4 f64 per the contract, audited in ffi_ptr.
    unsafe { ffi_ptr::write_out(out.cast::<[f64; 4]>(), result) };
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    unsafe_code,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    // ---- FFI shims: marshaling must equal the pure fn; null pointers must be a no-op. ----
    // The pure fns are checked in the engine crate; these tests would fail if a shim mis-marshaled
    // length / row-col order / which buffer, so the equivalence assertion is load-bearing.

    #[test]
    fn ffi_calc_weight_vec_matches_pure_and_null_is_noop() {
        let scores = [0.0, libm::log(3.0), 0.5];
        let mut expected = [0.0_f64; 3];
        calc_weight_vec(&scores, 1.0, &mut expected);
        let mut got = [0.0_f64; 3];
        unsafe {
            autoware_ndt_scan_matcher_rs_calc_weight_vec(scores.as_ptr(), 3, 1.0, got.as_mut_ptr());
        }
        assert_eq!(got, expected);

        let mut sentinel = [42.0_f64; 3];
        unsafe {
            autoware_ndt_scan_matcher_rs_calc_weight_vec(
                core::ptr::null(),
                3,
                1.0,
                sentinel.as_mut_ptr(),
            );
        }
        assert_eq!(
            sentinel, [42.0_f64; 3],
            "null scores must leave output untouched"
        );
    }

    #[test]
    fn ffi_weighted_mean_and_cov_matches_pure() {
        let poses = [0.0, 0.0, 2.0, 0.0, 1.0, 1.0];
        let weights = [0.25, 0.5, 0.25];
        let (e_mean, e_cov) = calculate_weighted_mean_and_cov(&poses, &weights);
        let (mut mean, mut cov) = ([0.0_f64; 2], [0.0_f64; 4]);
        unsafe {
            autoware_ndt_scan_matcher_rs_calculate_weighted_mean_and_cov(
                poses.as_ptr(),
                weights.as_ptr(),
                3,
                mean.as_mut_ptr(),
                cov.as_mut_ptr(),
            );
        }
        assert_eq!(mean, e_mean);
        assert_eq!(cov, e_cov);
    }

    #[test]
    fn ffi_laplace_matches_pure_and_null_is_noop() {
        let mut h = [0.0_f64; 36];
        h[0] = -2.0;
        h[7] = -4.0;
        let expected = laplace_xy_covariance(&h);
        let mut got = [0.0_f64; 4];
        unsafe { autoware_ndt_scan_matcher_rs_laplace_xy_covariance(h.as_ptr(), got.as_mut_ptr()) };
        assert_eq!(got, expected);

        let mut sentinel = [7.0_f64; 4];
        unsafe {
            autoware_ndt_scan_matcher_rs_laplace_xy_covariance(
                core::ptr::null(),
                sentinel.as_mut_ptr(),
            );
        };
        assert_eq!(sentinel, [7.0_f64; 4]);
    }

    #[test]
    fn ffi_rotate_matches_pure_and_null_is_noop() {
        let cov = [4.0, 1.0, 1.0, 9.0];
        let (c, s) = (libm::cos(0.3), libm::sin(0.3));
        let rot = [c, -s, s, c];

        let e_map = rotate_covariance_to_map(&cov, &rot);
        let e_base = rotate_covariance_to_base_link(&cov, &rot);
        let (mut map, mut base) = ([0.0_f64; 4], [0.0_f64; 4]);
        unsafe {
            autoware_ndt_scan_matcher_rs_rotate_covariance_to_map(
                cov.as_ptr(),
                rot.as_ptr(),
                map.as_mut_ptr(),
            );
            autoware_ndt_scan_matcher_rs_rotate_covariance_to_base_link(
                cov.as_ptr(),
                rot.as_ptr(),
                base.as_mut_ptr(),
            );
        }
        assert_eq!(map, e_map);
        assert_eq!(base, e_base);

        // Both shims must leave their output untouched on a null input (each has its own branch).
        let mut s_map = [5.0_f64; 4];
        let mut s_base = [6.0_f64; 4];
        unsafe {
            autoware_ndt_scan_matcher_rs_rotate_covariance_to_map(
                core::ptr::null(),
                rot.as_ptr(),
                s_map.as_mut_ptr(),
            );
            autoware_ndt_scan_matcher_rs_rotate_covariance_to_base_link(
                core::ptr::null(),
                rot.as_ptr(),
                s_base.as_mut_ptr(),
            );
        }
        assert_eq!(s_map, [5.0_f64; 4]);
        assert_eq!(s_base, [6.0_f64; 4]);
    }

    #[test]
    fn ffi_adjust_diagonal_matches_pure_and_null_is_noop() {
        let cov = [1.0, 0.0, 0.0, 1.0];
        let identity = [1.0, 0.0, 0.0, 1.0];
        let expected = adjust_diagonal_covariance(&cov, &identity, 4.0, 9.0);
        let mut got = [0.0_f64; 4];
        unsafe {
            autoware_ndt_scan_matcher_rs_adjust_diagonal_covariance(
                cov.as_ptr(),
                identity.as_ptr(),
                4.0,
                9.0,
                got.as_mut_ptr(),
            );
        }
        assert_eq!(got, expected);

        let mut sentinel = [3.0_f64; 4];
        unsafe {
            autoware_ndt_scan_matcher_rs_adjust_diagonal_covariance(
                core::ptr::null(),
                identity.as_ptr(),
                4.0,
                9.0,
                sentinel.as_mut_ptr(),
            );
        }
        assert_eq!(sentinel, [3.0_f64; 4]);
    }
}
