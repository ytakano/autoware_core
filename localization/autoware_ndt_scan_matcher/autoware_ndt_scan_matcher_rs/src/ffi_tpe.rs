// Copyright 2026 Autoware Foundation
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

//! C ABI shims over [`realtime_ndt_scan_matcher::tpe`] — the opaque `AwTpe` handle + the propose/evaluate
//! entry points the align-service pose search drives from C++. The estimator itself lives in the
//! engine crate; this module is always compiled (the node crate is std).

use realtime_ndt_scan_matcher::tpe::{
    Direction, Error, INPUT_DIMENSION, PRIOR_DIMENSION, TreeStructuredParzenEstimator, Trial,
};

use crate::ffi_ptr::{self, ffi_mut, ffi_ref};

/// C ABI status: success.
pub const AW_TPE_STATUS_OK: i32 = 0;
/// C ABI status: a required pointer argument was null.
pub const AW_TPE_STATUS_NULL_POINTER: i32 = 1;
/// C ABI status: a prior/input array had the wrong length.
pub const AW_TPE_STATUS_INVALID_LENGTH: i32 = 2;
/// C ABI status: the direction code was neither `MINIMIZE` nor `MAXIMIZE`.
pub const AW_TPE_STATUS_INVALID_DIRECTION: i32 = 3;
/// C ABI status: a trial input/score was invalid (e.g. non-finite).
pub const AW_TPE_STATUS_INVALID_INPUT: i32 = 4;
/// C ABI status: an internal error (e.g. invalid KDE state, conversion failure).
pub const AW_TPE_STATUS_INTERNAL_ERROR: i32 = 5;
/// C ABI direction code for [`Direction::Minimize`].
pub const AW_TPE_DIRECTION_MINIMIZE: i32 = 0;
/// C ABI direction code for [`Direction::Maximize`].
pub const AW_TPE_DIRECTION_MAXIMIZE: i32 = 1;

/// Opaque handle wrapping a [`TreeStructuredParzenEstimator`] for the C ABI (an `AwTpe*` on the C++
/// side). Created/driven through the `autoware_ndt_scan_matcher_rs_tpe_*` `extern "C"` shims.
pub struct AwTpe {
    inner: TreeStructuredParzenEstimator,
}

fn direction_from_ffi(direction: i32) -> Result<Direction, i32> {
    match direction {
        AW_TPE_DIRECTION_MINIMIZE => Ok(Direction::Minimize),
        AW_TPE_DIRECTION_MAXIMIZE => Ok(Direction::Maximize),
        _ => Err(AW_TPE_STATUS_INVALID_DIRECTION),
    }
}

fn status_from_error(error: Error) -> i32 {
    match error {
        Error::InvalidMeanLen | Error::InvalidStddevLen => AW_TPE_STATUS_INVALID_LENGTH,
        Error::InvalidStartupTrials
        | Error::InvalidMean
        | Error::InvalidStddev
        | Error::InvalidTrialInput
        | Error::InvalidTrialScore => AW_TPE_STATUS_INVALID_INPUT,
        Error::InvalidKdeState | Error::ConversionFailed => AW_TPE_STATUS_INTERNAL_ERROR,
    }
}

#[expect(
    unsafe_code,
    reason = "C ABI helper; converts validated pointer and length to a slice"
)]
fn checked_slice<'a>(ptr: *const f64, len: usize, expected_len: usize) -> Result<&'a [f64], i32> {
    if len != expected_len {
        return Err(AW_TPE_STATUS_INVALID_LENGTH);
    }
    // SAFETY: caller promises `len` readable f64 values; deref audited in ffi_ptr (null → None).
    unsafe { ffi_ptr::opt_slice(ptr, len) }.ok_or(AW_TPE_STATUS_NULL_POINTER)
}

#[expect(
    unsafe_code,
    reason = "C ABI helper; converts validated pointer and length to a mutable slice"
)]
fn checked_output_slice<'a>(
    ptr: *mut f64,
    len: usize,
    expected_len: usize,
) -> Result<&'a mut [f64], i32> {
    if len != expected_len {
        return Err(AW_TPE_STATUS_INVALID_LENGTH);
    }
    // SAFETY: caller promises `len` writable f64 values; deref audited in ffi_ptr (null → None).
    unsafe { ffi_ptr::opt_slice_mut(ptr, len) }.ok_or(AW_TPE_STATUS_NULL_POINTER)
}

/// Construct an owned Rust TPE handle. Returns null on invalid input.
///
/// # Safety
/// `sample_mean` and `sample_stddev` must each point to five readable `f64` values. Rust copies the
/// slices and does not retain the pointers. Free the returned handle with
/// [`autoware_ndt_scan_matcher_rs_tpe_free`].
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads caller-owned mean/stddev buffers after null/length checks"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_tpe_new(
    direction: i32,
    n_startup_trials: i64,
    sample_mean: *const f64,
    sample_mean_len: usize,
    sample_stddev: *const f64,
    sample_stddev_len: usize,
    seed: u64,
) -> *mut AwTpe {
    let Ok(direction) = direction_from_ffi(direction) else {
        return core::ptr::null_mut();
    };
    let Ok(mean) = checked_slice(sample_mean, sample_mean_len, PRIOR_DIMENSION) else {
        return core::ptr::null_mut();
    };
    let Ok(stddev) = checked_slice(sample_stddev, sample_stddev_len, PRIOR_DIMENSION) else {
        return core::ptr::null_mut();
    };
    match TreeStructuredParzenEstimator::with_seed(direction, n_startup_trials, mean, stddev, seed)
    {
        Ok(inner) => ffi_ptr::into_handle(AwTpe { inner }),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Free an owned TPE handle returned by [`autoware_ndt_scan_matcher_rs_tpe_new`].
///
/// # Safety
/// `handle` must be null or a live handle returned by this crate. Passing the same handle twice is
/// invalid.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reclaims Box-owned opaque handle"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_tpe_free(handle: *mut AwTpe) {
    // SAFETY: `handle` is null or a live handle from `tpe_new`; reclaimed once in ffi_ptr.
    unsafe { ffi_ptr::free_handle(handle) };
}

/// Generate one TPE input into `out_input`.
///
/// # Safety
/// `handle` must be a live TPE handle, and `out_input` must point to six writable `f64` slots.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; writes caller-owned output buffer after null/length checks"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_tpe_get_next_input(
    handle: *mut AwTpe,
    out_input: *mut f64,
    out_len: usize,
) -> i32 {
    let tpe = ffi_mut!(handle, else return AW_TPE_STATUS_NULL_POINTER);
    let Ok(out) = checked_output_slice(out_input, out_len, INPUT_DIMENSION) else {
        return if out_len == INPUT_DIMENSION {
            AW_TPE_STATUS_NULL_POINTER
        } else {
            AW_TPE_STATUS_INVALID_LENGTH
        };
    };
    match tpe.inner.get_next_input() {
        Ok(input) => {
            for (dst, src) in out.iter_mut().zip(input.iter().copied()) {
                *dst = src;
            }
            AW_TPE_STATUS_OK
        }
        Err(error) => status_from_error(error),
    }
}

/// Add one evaluated trial to the TPE handle.
///
/// # Safety
/// `handle` must be a live TPE handle, and `input` must point to six readable `f64` values.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads caller-owned input buffer after null/length checks"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_tpe_add_trial(
    handle: *mut AwTpe,
    input: *const f64,
    input_len: usize,
    score: f64,
) -> i32 {
    let tpe = ffi_mut!(handle, else return AW_TPE_STATUS_NULL_POINTER);
    let Ok(input_slice) = checked_slice(input, input_len, INPUT_DIMENSION) else {
        return if input_len == INPUT_DIMENSION {
            AW_TPE_STATUS_NULL_POINTER
        } else {
            AW_TPE_STATUS_INVALID_LENGTH
        };
    };
    let Ok(input) = <[f64; INPUT_DIMENSION]>::try_from(input_slice) else {
        return AW_TPE_STATUS_INVALID_LENGTH;
    };
    match tpe.inner.add_trial(Trial { input, score }) {
        Ok(()) => AW_TPE_STATUS_OK,
        Err(error) => status_from_error(error),
    }
}

/// Return the number of trials currently stored in the TPE handle. Null returns zero.
///
/// # Safety
/// `handle` must be null or a live TPE handle.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads opaque handle after null check"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_tpe_trials_len(
    handle: *const AwTpe,
) -> usize {
    let tpe = ffi_ref!(handle, else return 0);
    tpe.inner.trials_len()
}

/// Return the current number of good trials in the TPE KDE partition. Null returns zero.
///
/// # Safety
/// `handle` must be null or a live TPE handle.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; reads opaque handle after null check"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_tpe_above_num(handle: *const AwTpe) -> usize {
    let tpe = ffi_ref!(handle, else return 0);
    tpe.inner.above_num()
}

#[cfg(test)]
#[allow(
    clippy::allow_attributes,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test code uses direct assertions and fixed-size fixture indexing"
)]
mod tests {
    use core::f64::consts::PI;

    use realtime_ndt_scan_matcher::tpe::Input;

    use super::*;

    fn sample_mean() -> [f64; 5] {
        [0.0, 0.0, 0.0, 0.0, 0.0]
    }

    fn sample_stddev() -> [f64; 5] {
        [1.0, 1.0, 0.1, 0.1, 0.1]
    }

    fn finite_input(input: &Input) {
        assert!(input.iter().all(|v| v.is_finite()));
        assert!(input[5] >= -PI);
        assert!(input[5] < PI);
    }

    #[allow(unsafe_code, reason = "test exercises the C ABI entry points directly")]
    #[test]
    fn ffi_rejects_nulls_and_invalid_lengths() {
        let mean = sample_mean();
        let stddev = sample_stddev();
        unsafe {
            assert!(
                autoware_ndt_scan_matcher_rs_tpe_new(
                    AW_TPE_DIRECTION_MAXIMIZE,
                    10,
                    mean.as_ptr(),
                    4,
                    stddev.as_ptr(),
                    stddev.len(),
                    0,
                )
                .is_null()
            );
            assert!(
                autoware_ndt_scan_matcher_rs_tpe_new(
                    99,
                    10,
                    mean.as_ptr(),
                    mean.len(),
                    stddev.as_ptr(),
                    stddev.len(),
                    0,
                )
                .is_null()
            );
            let handle = autoware_ndt_scan_matcher_rs_tpe_new(
                AW_TPE_DIRECTION_MAXIMIZE,
                10,
                mean.as_ptr(),
                mean.len(),
                stddev.as_ptr(),
                stddev.len(),
                0,
            );
            assert!(!handle.is_null());
            let mut out = [0.0; 6];
            assert_eq!(
                autoware_ndt_scan_matcher_rs_tpe_get_next_input(
                    core::ptr::null_mut(),
                    out.as_mut_ptr(),
                    out.len(),
                ),
                AW_TPE_STATUS_NULL_POINTER
            );
            assert_eq!(
                autoware_ndt_scan_matcher_rs_tpe_get_next_input(
                    handle,
                    core::ptr::null_mut(),
                    out.len(),
                ),
                AW_TPE_STATUS_NULL_POINTER
            );
            assert_eq!(
                autoware_ndt_scan_matcher_rs_tpe_get_next_input(handle, out.as_mut_ptr(), 5),
                AW_TPE_STATUS_INVALID_LENGTH
            );
            autoware_ndt_scan_matcher_rs_tpe_free(handle);
        }
    }

    #[allow(unsafe_code, reason = "test exercises the C ABI entry points directly")]
    #[test]
    fn ffi_generates_inputs_and_updates_trials() {
        let mean = sample_mean();
        let stddev = sample_stddev();
        unsafe {
            let handle = autoware_ndt_scan_matcher_rs_tpe_new(
                AW_TPE_DIRECTION_MAXIMIZE,
                5,
                mean.as_ptr(),
                mean.len(),
                stddev.as_ptr(),
                stddev.len(),
                42,
            );
            assert!(!handle.is_null());
            let mut input = [0.0; 6];
            assert_eq!(
                autoware_ndt_scan_matcher_rs_tpe_get_next_input(
                    handle,
                    input.as_mut_ptr(),
                    input.len(),
                ),
                AW_TPE_STATUS_OK
            );
            finite_input(&input);
            for i in 0..10 {
                let trial = [f64::from(i), 0.0, 0.0, 0.0, 0.0, 0.0];
                assert_eq!(
                    autoware_ndt_scan_matcher_rs_tpe_add_trial(
                        handle,
                        trial.as_ptr(),
                        trial.len(),
                        f64::from(i),
                    ),
                    AW_TPE_STATUS_OK
                );
            }
            assert_eq!(autoware_ndt_scan_matcher_rs_tpe_trials_len(handle), 10);
            assert_eq!(autoware_ndt_scan_matcher_rs_tpe_above_num(handle), 1);
            autoware_ndt_scan_matcher_rs_tpe_free(handle);
        }
    }

    #[allow(unsafe_code, reason = "test exercises the C ABI entry points directly")]
    #[test]
    fn ffi_fixed_seed_reproduces_sequence() {
        let mean = sample_mean();
        let stddev = sample_stddev();
        unsafe {
            let a = autoware_ndt_scan_matcher_rs_tpe_new(
                AW_TPE_DIRECTION_MAXIMIZE,
                5,
                mean.as_ptr(),
                mean.len(),
                stddev.as_ptr(),
                stddev.len(),
                77,
            );
            let b = autoware_ndt_scan_matcher_rs_tpe_new(
                AW_TPE_DIRECTION_MAXIMIZE,
                5,
                mean.as_ptr(),
                mean.len(),
                stddev.as_ptr(),
                stddev.len(),
                77,
            );
            assert!(!a.is_null());
            assert!(!b.is_null());
            for _ in 0..8 {
                let mut ia = [0.0; 6];
                let mut ib = [0.0; 6];
                assert_eq!(
                    autoware_ndt_scan_matcher_rs_tpe_get_next_input(a, ia.as_mut_ptr(), ia.len()),
                    AW_TPE_STATUS_OK
                );
                assert_eq!(
                    autoware_ndt_scan_matcher_rs_tpe_get_next_input(b, ib.as_mut_ptr(), ib.len()),
                    AW_TPE_STATUS_OK
                );
                assert_eq!(ia, ib);
            }
            autoware_ndt_scan_matcher_rs_tpe_free(a);
            autoware_ndt_scan_matcher_rs_tpe_free(b);
        }
    }
}
