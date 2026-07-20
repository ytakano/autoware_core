# Module index

One line per Rust source module in the **node crate** (`autoware_ndt_scan_matcher_rs/src/`). The
NDT algorithm modules live in the **engine crate** (`realtime_ndt_scan_matcher`) and are indexed in
its book.

## FFI boundary (the C ABI over the engine's Rust API)

- `lib.rs` — crate root, feature gating (`ros`), `nalgebra` re-export, the top-level C-ABI entry
  points.
- `ffi.rs` — panic-containment boundary (`catch_unwind`) + `AwStatus` / `Error` for the
  object-level FFI.
- `ffi_ptr.rs` — audited raw-pointer helpers + guard macros; the single home for pointer
  dereferences at the boundary (`core` + `alloc`).
- `ffi_host.rs` — the `AwHost` ROS side-effects vtable (clock / log / TF) + `AwStr` / `AwPose`;
  the C-ABI adapter for the engine's portable `Host` seam.
- `ffi_matrix.rs` — shared row-major matrix marshaling (chunked-slice based, no index arithmetic).
- `ffi_engine.rs` — bounded C-ABI handles for `NdtEngine` and caller-owned `MatchScratch`.
- `ffi_covariance.rs` — stateless C-ABI shims over covariance math.
- `ffi_tpe.rs` / `ffi_voxel_grid.rs` — C-ABI shims over the TPE search and the voxel grid.

## `std` ROS node shell

- `node.rs` — thin pose/trigger callbacks + convergence FFI.
- `node_handle.rs` — the opaque `NdtScanMatcherRs` handle, validated `Params` / `AwNdtParams`, and
  node state.
- `node_map_update.rs` — status-returning staged map publication over the `MapSource` vtable.
- `node_align_service.rs` — the align-service decisions + the Rust-owned TPE search + trace ABI.
- `sensor_points.rs` — the sensor-callback prologue (decode / TF / transform / validation).

## Generated

- `ros_msgs` — bindgen `geometry_msgs` C structs (`ros` feature; allow-listed).

> Source: the node crate `src/` module docs.
