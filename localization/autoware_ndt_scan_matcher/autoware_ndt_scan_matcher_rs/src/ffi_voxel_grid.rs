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

//! C ABI shims over [`realtime_ndt_scan_matcher::voxel_grid`] — the opaque single-grid `VoxelGrid` handle and
//! the multi-grid `VoxelGridMap` handle (build / radius-search / leaf lookup). The data structures
//! live in the engine crate; this module only owns the handle lifecycle and marshals pointers.

use realtime_ndt_scan_matcher::voxel_grid::{VoxelGrid, VoxelGridMap};

use crate::ffi_ptr::{self, ffi_mut, ffi_read, ffi_ref, ffi_slice};

// ---- C ABI (opaque handle; the shape the NDT engine will reuse) ----

/// # Safety
/// `points` points to `n` `[f32;3]` triples, `leaf_size` to 3 `f64`. Returns an owned grid handle
/// (free with `..._voxel_grid_free`), or null if `points`/`leaf_size` is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_build(
    points: *const f32,
    n: usize,
    leaf_size: *const f64,
    min_points: i32,
    eig_mult: f64,
) -> *mut VoxelGrid {
    let Some(flat_len) = n.checked_mul(3) else {
        return core::ptr::null_mut();
    };
    let _ = flat_len;
    let pts = ffi_slice!(points, n, [f32; 3], else return core::ptr::null_mut());
    let ls = ffi_read!(leaf_size, [f64; 3], else return core::ptr::null_mut());
    ffi_ptr::into_handle(VoxelGrid::build(pts, ls, min_points, eig_mult))
}

/// # Safety
/// `grid` is a handle from `..._voxel_grid_build`; `point` points to 3 `f32`; `mean_out`/`icov_out`
/// to 3 / 9 writable `f64`. Returns true and writes outputs iff a leaf covers `point`.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_leaf_at(
    grid: *const VoxelGrid,
    point: *const f32,
    mean_out: *mut f64,
    icov_out: *mut f64,
) -> bool {
    let grid = ffi_ref!(grid, else return false);
    let p = ffi_read!(point, [f32; 3], else return false);
    if mean_out.is_null() || icov_out.is_null() {
        return false;
    }
    match grid.leaf_at(p) {
        Some(leaf) => {
            // SAFETY: non-null per the check above; 3/9 f64 per the contract, audited in ffi_ptr.
            unsafe {
                ffi_ptr::write_out(mean_out.cast::<[f64; 3]>(), leaf.mean);
                ffi_ptr::write_out(icov_out.cast::<[f64; 9]>(), leaf.icov);
            }
            true
        }
        None => false,
    }
}

/// # Safety
/// `grid` must be a handle from `..._voxel_grid_build` (or null); it must not be used afterwards.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_free(grid: *mut VoxelGrid) {
    // SAFETY: `grid` is a handle from `_build` (or null → no-op); reclaimed once in ffi_ptr.
    unsafe { ffi_ptr::free_handle(grid) };
}

/// # Safety
/// `leaf_size` points to 3 readable `f64` (or null -> returns null). Returns an owned handle
/// (free with `..._voxel_grid_map_free`).
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_map_new(
    leaf_size: *const f64,
    min_points: i32,
    eig_mult: f64,
) -> *mut VoxelGridMap {
    let ls = ffi_read!(leaf_size, [f64; 3], else return core::ptr::null_mut());
    ffi_ptr::into_handle(VoxelGridMap::new(ls, min_points, eig_mult))
}

/// # Safety
/// `map` is a valid handle; `points` points to `3*n` `f32`. No-op if `map`/`points` is null.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_map_add_target(
    map: *mut VoxelGridMap,
    points: *const f32,
    n: usize,
    id: u64,
) {
    let map = ffi_mut!(map, else return);
    let pts = ffi_slice!(points, n, [f32; 3], else return);
    map.add_target(pts, id);
}

/// # Safety
/// `map` is a valid handle (or null). Removes the grid registered under `id`.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_map_remove_target(
    map: *mut VoxelGridMap,
    id: u64,
) {
    ffi_mut!(map, else return).remove_target(id);
}

/// # Safety
/// `map` is a valid handle (or null). Builds the kd-tree over current grids' centroids.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_map_create_kdtree(
    map: *mut VoxelGridMap,
    max_active_leaves: usize,
) -> bool {
    ffi_mut!(map, else return false)
        .try_create_kdtree(max_active_leaves)
        .is_ok()
}

/// # Safety
/// `map` valid; `point` points to 3 `f32`; `out_idx` to `cap` writable `u32`. Writes up to `cap`
/// leaf indices and returns the total number found (`max_nn == 0` = unlimited).
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::allow_attributes,
    reason = "C ABI count marshaling: u32<->usize at the boundary (writes bounded by cap)"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_map_radius_search(
    map: *const VoxelGridMap,
    point: *const f32,
    radius: f64,
    max_nn: u32,
    out_idx: *mut u32,
    cap: u32,
) -> u32 {
    let map = ffi_ref!(map, else return 0);
    let p = ffi_read!(point, [f32; 3], else return 0);
    let mut found: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
    if map
        .radius_search(p, radius, max_nn as usize, &mut found)
        .is_err()
    {
        return 0;
    }
    // SAFETY: `out_idx` addresses `cap` writable u32 per the contract (null → skipped); the zip
    // bounds writes to `min(cap, found.len())`. Deref audited in ffi_ptr.
    if let Some(out) = unsafe { ffi_ptr::opt_slice_mut(out_idx, cap as usize) } {
        for (dst, &leaf_idx) in out.iter_mut().zip(found.iter()) {
            *dst = leaf_idx as u32;
        }
    }
    found.len() as u32
}

/// # Safety
/// `map` valid; `mean_out`/`icov_out` to 3 / 9 writable `f64`. Writes them and returns true iff
/// `idx` is a valid leaf index.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[allow(
    clippy::as_conversions,
    clippy::allow_attributes,
    reason = "C ABI: idx u32->usize"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_map_leaf(
    map: *const VoxelGridMap,
    idx: u32,
    mean_out: *mut f64,
    icov_out: *mut f64,
) -> bool {
    let map = ffi_ref!(map, else return false);
    if mean_out.is_null() || icov_out.is_null() {
        return false;
    }
    match map.leaf(idx as usize) {
        Some(leaf) => {
            // SAFETY: non-null per the check above; 3/9 f64 per the contract, audited in ffi_ptr.
            unsafe {
                ffi_ptr::write_out(mean_out.cast::<[f64; 3]>(), leaf.mean);
                ffi_ptr::write_out(icov_out.cast::<[f64; 9]>(), leaf.icov);
            }
            true
        }
        None => false,
    }
}

/// # Safety
/// `map` must be a handle from `..._voxel_grid_map_new` (or null); not used afterwards.
#[expect(
    unsafe_code,
    reason = "C ABI boundary; pointers validated per rust-c-ffi-safety"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autoware_ndt_scan_matcher_rs_voxel_grid_map_free(map: *mut VoxelGridMap) {
    // SAFETY: `map` is a handle from `_map_new` (or null → no-op); reclaimed once in ffi_ptr.
    unsafe { ffi_ptr::free_handle(map) };
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::unreadable_literal,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    unsafe_code,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    // 8 points packed inside the single voxel containing (cx,cy,cz) for leaf_size 2.0.
    fn dense_cluster(cx: f32, cy: f32, cz: f32) -> alloc::vec::Vec<[f32; 3]> {
        (0..8)
            .map(|i| {
                let f = i as f32 * 0.02;
                [cx + f, cy - f, cz + 0.5 * f]
            })
            .collect()
    }

    // ---- FFI shims: round-trip equals the pure path; null/cap contracts ----

    #[test]
    fn ffi_voxel_grid_build_leaf_at_matches_pure() {
        let pts = dense_cluster(1.0, 1.0, 1.0);
        let flat: alloc::vec::Vec<f32> = pts.iter().flat_map(|p| p.iter().copied()).collect();
        let leaf_size = [2.0_f64, 2.0, 2.0];

        let pure = VoxelGrid::build(&pts, leaf_size, 6, 0.01);
        let pure_leaf = pure.leaf_at([1.0, 1.0, 1.0]).expect("pure leaf");

        let grid = unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_build(
                flat.as_ptr(),
                pts.len(),
                leaf_size.as_ptr(),
                6,
                0.01,
            )
        };
        assert!(!grid.is_null());
        let q = [1.0_f32, 1.0, 1.0];
        let (mut mean, mut icov) = ([0.0_f64; 3], [0.0_f64; 9]);
        let hit = unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_leaf_at(
                grid,
                q.as_ptr(),
                mean.as_mut_ptr(),
                icov.as_mut_ptr(),
            )
        };
        assert!(hit);
        assert_eq!(mean, pure_leaf.mean);
        assert_eq!(icov, pure_leaf.icov);
        unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_free(grid) };

        // null contracts.
        assert!(
            unsafe {
                autoware_ndt_scan_matcher_rs_voxel_grid_build(
                    core::ptr::null(),
                    0,
                    leaf_size.as_ptr(),
                    6,
                    0.01,
                )
            }
            .is_null()
        );
        assert!(!unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_leaf_at(
                core::ptr::null(),
                q.as_ptr(),
                mean.as_mut_ptr(),
                icov.as_mut_ptr(),
            )
        });
        unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_free(core::ptr::null_mut()) }; // no-op, must not crash
    }

    #[test]
    fn ffi_map_radius_search_and_leaf_match_pure_with_cap() {
        let a = dense_cluster(1.0, 1.0, 1.0);
        let b = dense_cluster(21.0, 1.0, 1.0);
        let leaf_size = [2.0_f64, 2.0, 2.0];

        // Pure reference map.
        let mut pure = VoxelGridMap::new(leaf_size, 6, 0.01);
        pure.add_target(&a, 0);
        pure.add_target(&b, 1);
        pure.try_create_kdtree(418_000).expect("build kd-tree");

        // FFI map, same inputs.
        let fa: alloc::vec::Vec<f32> = a.iter().flat_map(|p| p.iter().copied()).collect();
        let fb: alloc::vec::Vec<f32> = b.iter().flat_map(|p| p.iter().copied()).collect();
        let map =
            unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_new(leaf_size.as_ptr(), 6, 0.01) };
        assert!(!map.is_null());
        unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_map_add_target(map, fa.as_ptr(), a.len(), 0);
            autoware_ndt_scan_matcher_rs_voxel_grid_map_add_target(map, fb.as_ptr(), b.len(), 1);
            assert!(autoware_ndt_scan_matcher_rs_voxel_grid_map_create_kdtree(
                map, 418_000
            ));
        }

        // Query near A: FFI count == pure count, and the returned leaf mean/icov match pure.
        let q = [1.0_f32, 1.0, 1.0];
        let mut idx = [u32::MAX; 8];
        let n = unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_map_radius_search(
                map,
                q.as_ptr(),
                1.5,
                0,
                idx.as_mut_ptr(),
                idx.len() as u32,
            )
        };
        let mut pure_hits = alloc::vec::Vec::new();
        pure.radius_search([1.0, 1.0, 1.0], 1.5, 0, &mut pure_hits)
            .expect("radius search");
        assert_eq!(n as usize, pure_hits.len());
        assert_eq!(n, 1);

        let (mut mean, mut icov) = ([0.0_f64; 3], [0.0_f64; 9]);
        assert!(unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_map_leaf(
                map,
                idx[0],
                mean.as_mut_ptr(),
                icov.as_mut_ptr(),
            )
        });
        let pure_leaf = pure.leaf(pure_hits[0]).expect("pure leaf");
        assert_eq!(mean, pure_leaf.mean);
        assert_eq!(icov, pure_leaf.icov);

        // cap == 0: nothing written, but the true total is still returned.
        let total = unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_map_radius_search(
                map,
                q.as_ptr(),
                1.5,
                0,
                idx.as_mut_ptr(),
                0,
            )
        };
        assert_eq!(total, 1, "returns total found even when cap is 0");

        // null contracts.
        assert_eq!(
            unsafe {
                autoware_ndt_scan_matcher_rs_voxel_grid_map_radius_search(
                    core::ptr::null(),
                    q.as_ptr(),
                    1.5,
                    0,
                    idx.as_mut_ptr(),
                    idx.len() as u32,
                )
            },
            0
        );
        assert!(!unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_map_leaf(
                map,
                9999,
                mean.as_mut_ptr(),
                icov.as_mut_ptr(),
            )
        });
        unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_free(map) };
        unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_free(core::ptr::null_mut()) }; // no-op
    }

    // FFI map: null handles are no-ops (must not deref); remove_target via the shim drops a grid;
    // a small cap truncates the written indices while the return value stays the true total.
    #[test]
    fn ffi_map_null_remove_and_cap_contracts() {
        let ls = [1.0_f64, 1.0, 1.0];

        // null-handle contracts: must not crash, and `new(null)` reports null.
        assert!(
            unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_new(core::ptr::null(), 6, 0.01) }
                .is_null()
        );
        unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_map_add_target(
                core::ptr::null_mut(),
                core::ptr::null(),
                0,
                0,
            );
            autoware_ndt_scan_matcher_rs_voxel_grid_map_remove_target(core::ptr::null_mut(), 0);
            assert!(!autoware_ndt_scan_matcher_rs_voxel_grid_map_create_kdtree(
                core::ptr::null_mut(),
                418_000
            ));
        }

        // Four clusters in four adjacent voxels (leaf_size 1.0), all within the query radius.
        let centers = [(0.4_f32, 0.4_f32), (1.4, 0.4), (2.4, 0.4), (3.4, 0.4)];
        let map = unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_new(ls.as_ptr(), 6, 0.01) };
        assert!(!map.is_null());
        for (id, &(cx, cy)) in centers.iter().enumerate() {
            let c = dense_cluster(cx, cy, 0.4);
            let flat: alloc::vec::Vec<f32> = c.iter().flat_map(|p| p.iter().copied()).collect();
            unsafe {
                autoware_ndt_scan_matcher_rs_voxel_grid_map_add_target(
                    map,
                    flat.as_ptr(),
                    c.len(),
                    id as u64,
                );
            }
        }
        // Remove the first cluster via the FFI shim (the dispatch path under test).
        unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_remove_target(map, 0) };
        unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_create_kdtree(map, 418_000) };

        let q = [1.9_f32, 0.4, 0.4];
        // cap smaller than the number found: only `cap` indices written, rest stay sentinel,
        // but the return is the true total (3 remaining clusters within radius).
        let mut idx = [u32::MAX; 8];
        let total = unsafe {
            autoware_ndt_scan_matcher_rs_voxel_grid_map_radius_search(
                map,
                q.as_ptr(),
                3.0,
                0,
                idx.as_mut_ptr(),
                2,
            )
        };
        assert_eq!(
            total, 3,
            "cluster 0 removed via FFI; 3 remain within radius"
        );
        assert!(
            idx[0] != u32::MAX && idx[1] != u32::MAX,
            "cap entries written"
        );
        assert_eq!(idx[2], u32::MAX, "entries beyond cap left untouched");

        unsafe { autoware_ndt_scan_matcher_rs_voxel_grid_map_free(map) };
    }
}
