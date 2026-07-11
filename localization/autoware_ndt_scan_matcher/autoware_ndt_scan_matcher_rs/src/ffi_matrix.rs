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

//! Row-major matrix marshaling helpers shared by the C-ABI shim modules.
//!
//! `chunks_exact*` carries the row/pose stride, so there is **no index arithmetic** — the
//! `(r * 4) + c` / `(k * 16) + …` patterns these replace needed an `arithmetic_side_effects`
//! suppression over integer math, which the project's float-only suppression rule forbids.
//! The only remaining suppression is `indexing_slicing` on fixed-size nalgebra matrices
//! (allowlisted). Writers fill `min(dst-chunks, rows/poses)`; short destinations truncate and
//! short sources leave the tail untouched, matching the old bounded loops.

use realtime_ndt_scan_matcher::nalgebra::{Matrix4, Matrix6};

/// Read a 16-float row-major buffer into a `Matrix4<f32>` (callers pass an `ffi_slice!(_, 16)`).
#[expect(
    clippy::indexing_slicing,
    reason = "(r, c) from enumerate() over 4-chunks index a fixed-size nalgebra Matrix4"
)]
#[must_use]
pub(crate) fn matrix4_from_row_major(buf: &[f32]) -> Matrix4<f32> {
    let mut m = Matrix4::<f32>::zeros();
    for (r, row) in buf.chunks_exact(4).take(4).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            m[(r, c)] = v;
        }
    }
    m
}

/// Write `m` into `dst` as 16 row-major `f32` (callers pass a 16-element slice).
// nalgebra `(r, c)` reads don't trip `indexing_slicing` (only the IndexMut writes in the reader do).
pub(crate) fn write_matrix4_row_major(dst: &mut [f32], m: &Matrix4<f32>) {
    for (r, row) in dst.chunks_exact_mut(4).take(4).enumerate() {
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = m[(r, c)];
        }
    }
}

/// Write `m` into `dst` as 36 row-major `f64` (callers pass a 36-element slice).
// nalgebra `(r, c)` reads don't trip `indexing_slicing` (only the IndexMut writes in the reader do).
pub(crate) fn write_matrix6_row_major(dst: &mut [f64], m: &Matrix6<f64>) {
    for (r, row) in dst.chunks_exact_mut(6).take(6).enumerate() {
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = m[(r, c)];
        }
    }
}

/// Write consecutive poses into `dst` as 16 row-major `f32` each; `zip` bounds the copy to
/// `min(dst.len() / 16, poses.len())` (the old `take(cap)` semantics).
pub(crate) fn write_matrix4_seq(dst: &mut [f32], poses: &[Matrix4<f32>]) {
    for (chunk, m) in dst.chunks_exact_mut(16).zip(poses.iter()) {
        write_matrix4_row_major(chunk, m);
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    fn seq_matrix(offset: f32) -> Matrix4<f32> {
        let mut m = Matrix4::<f32>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                m[(r, c)] = offset + ((r * 4) + c) as f32;
            }
        }
        m
    }

    #[test]
    fn matrix4_round_trips_row_major() {
        let m = seq_matrix(1.0);
        let mut buf = [0.0_f32; 16];
        write_matrix4_row_major(&mut buf, &m);
        for (i, &v) in buf.iter().enumerate() {
            assert_eq!(v, 1.0 + i as f32, "row-major order at {i}");
        }
        assert_eq!(matrix4_from_row_major(&buf), m);
    }

    #[test]
    fn matrix6_writes_row_major() {
        let mut m = Matrix6::<f64>::zeros();
        for r in 0..6 {
            for c in 0..6 {
                m[(r, c)] = ((r * 6) + c) as f64;
            }
        }
        let mut buf = [0.0_f64; 36];
        write_matrix6_row_major(&mut buf, &m);
        for (i, &v) in buf.iter().enumerate() {
            assert_eq!(v, i as f64);
        }
    }

    #[test]
    fn seq_truncates_to_capacity_and_source() {
        let poses = [seq_matrix(0.0), seq_matrix(100.0), seq_matrix(200.0)];
        // dst caps at 2 poses: the third must not be written.
        let mut buf = [-1.0_f32; 32];
        write_matrix4_seq(&mut buf, &poses);
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[16], 100.0);
        // Short source: trailing dst chunk untouched.
        let mut wide = [-1.0_f32; 48];
        write_matrix4_seq(&mut wide, &poses[..1]);
        assert_eq!(wide[15], 15.0);
        assert_eq!(wide[16], -1.0);
    }
}
