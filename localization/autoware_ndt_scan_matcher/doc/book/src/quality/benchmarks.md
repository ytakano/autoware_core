# Benchmarking

How the Rust port's performance is measured against the legacy C++ engine, and the current numbers.

The guiding principle is **capture-once, replay-everywhere**: capture representative inputs once,
then replay them deterministically through both engines so a comparison is apples-to-apples and free
of live-system jitter. The predictable **serial** backend (`num_threads = 1`) is the fair baseline —
it is the WCET reference, and it isolates the kernel from thread-pool scheduling noise (see
[Parallelism and worker threads](../start/features.md#parallelism-and-worker-threads)).

## Tiers

- **L1 — node / end-to-end.** The full ROS node timed by its built-in `exe_time_ms` publisher (present
  identically on both engines), `NDT_USE_RUST` OFF vs ON. **L1a** (automated, `bench/run_l1a.sh`) replays
  the in-repo `standard_sequence` cloud through the node frame-by-frame — deterministic, download-free,
  the CI regression baseline. **L1b** (opt-in) replays a real recorded drive for the headline number —
  **measured** on the Autoware urban-environment localization evaluation dataset (İstanbul), whose
  localization-only bag feeds the preprocessed cloud + twist straight into the node graph
  (`launch/ndt_l1b_loc.launch.xml`; replay shims + init in `bench/l1b_*.py`, method in
  `plan/ndt_bench.md`). See the measured results below. Rely on the deterministic **L1a** for
  regression tracking.
- **L2 — kernel micro-benchmark.** A tight per-frame `align` loop on a small synthetic cloud, to
  locate where time goes and bound per-frame WCET. Crate example `examples/wcet_frame.rs`.
- **L3 — offline differential replay.** One executable drives **both** engines on identical inputs
  (`bench/ndt_bench_replay.cpp`); realistic point density, zero live jitter. This is the tier with
  measured results below.

## What L3 measures

`bench/ndt_bench_replay.cpp` builds the target map **and kd-tree once per engine**, then times only
the repeated `align` loop with `steady_clock` — so it measures the align kernel, not map
construction. Both engines read the **same** synthetic geometry (three orthogonal planes at
`interval` spacing) from the same buffers, so the comparison is apples-to-apples by construction. It
emits per-align latency samples as JSON, which `bench/gen_report.py` renders into a self-contained
HTML report.

Fidelity controls: identical inputs and identical convergence (`iteration_num` must match, else the
engines did different work), a warmup phase, align-loop-only timing, and optional core pinning for a
stable tail (`TASKSET="taskset -c 2"`).

## Running

```sh
# L3 offline replay (both engines) — needs the Rust backend + the opt-in bench target.
colcon build --packages-select autoware_ndt_scan_matcher \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DNDT_USE_RUST=ON -DNDT_BUILD_BENCH=ON
bench/run.sh [ITERS] [WARMUP] [INTERVAL]   # -> bench/ndt_bench.json + bench/report.html
#   env: TASKSET="taskset -c 2" (pin a core), OUT_DIR=/path

# L1a node end-to-end, deterministic synthetic cloud, C++ (OFF) vs Rust (ON). The runner does the
# OFF/ON rebuild two-pass itself (opt-in NDT_BUILD_BENCH_L1); no download.
bench/run_l1a.sh [ITERS] [WARMUP]          # -> bench/l1a.json + bench/l1a_report.html

# L1b node end-to-end on the real Autoware sample-rosbag (opt-in; downloads ~193 MB if absent).
bench/run_l1b.sh                           # -> bench/l1b.json + bench/l1b_report.html
#   env: BAG_LIDAR_TOPIC=<bag lidar topic>, AUTOWARE_DATA_DIR=/path

# L2 per-frame WCET micro-benchmark (crate only, no colcon).
cargo run --release --example wcet_frame
```

## Measured results

> Snapshot — regenerate with the commands above; numbers are hardware- and input-dependent.

### L3 — align-loop latency (as of 2026-07-07)

Synthetic 3-plane cloud, **30,603 points**, `resolution = 2.0`, `max_iterations = 30`,
**`num_threads = 1` (serial)**, 200 iterations after 20 warmup, `steady_clock`. **Both engines
converge in `iteration_num = 4`** (equal work). Latency per `align` call, milliseconds:

| Engine | min | median | mean | p95 | max |
|---|--:|--:|--:|--:|--:|
| C++ (`multigrid_ndt_omp`) | 360.61 | 366.64 | 366.44 | 370.68 | 372.33 |
| Rust (`autoware_ndt_scan_matcher_rs`) | 118.19 | 118.67 | 118.71 | 119.06 | 120.50 |

**Rust is ≈ 3.1× faster** at the median (366.64 / 118.67 = 3.09×; ≈ 3.1× at p95 too) on this
workload.

*Environment:* AMD Ryzen 9 5900HX (16 logical cores), governor `powersave`, no CPU pinning,
`rustc 1.96.0`, `g++ 11.4.0`.

### L1b — node end-to-end on real urban data (as of 2026-07-10)

Autoware **urban-environment localization evaluation dataset** (İstanbul; Pandar XT32): the
localization-only bag replays the preprocessed cloud (~1.2–1.5 k pts/frame) + GNSS/INS twist directly
into the node graph (`launch/ndt_l1b_loc.launch.xml`: map loader + NDT + EKF loop, CycloneDDS,
`num_threads = 1`). Same map crop, same `ndt_align`-seeded init, 120 s recorded per engine at 10 Hz;
**both engines converge in `iteration_num = 3`** (equal work). Node `exe_time_ms` per frame,
milliseconds:

| Engine | n | p50 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|
| C++ (`multigrid_ndt_omp`) | 1199 | 4.20 | 6.45 | 7.63 | 11.97 |
| Rust (`autoware_ndt_scan_matcher_rs`) | 1198 | 3.61 | 5.68 | 6.35 | 7.70 |

**Rust is ≈ 1.16× faster at p50 and ≈ 1.56× at max** on this real-world workload. The ratio is
smaller than L1a/L3 because real urban frames converge in only 3 iterations, so the align kernel is a
smaller share of the end-to-end frame (ingest + covariance + publish overheads are largely shared);
the tail improves the most — consistent with the Rust engine's bounded-WCET design.

Replay shims (documented in `plan/ndt_bench.md`): the bag's header stamps lag its `/clock`, so a
re-stamp relay aligns the cloud + twist stamps; the recorded `/clock` is excluded from replay (the
player's own clock is the single time base); the single-file 15 km PCD overflows
`MultiVoxelGridCovariance`'s int32 voxel index at 2 m resolution, so the map is cropped to ±3.5 km
around the route (the production tile pipeline avoids this by construction); init = fresh GNSS seed →
bag paused → `ndt_align_srv` (TPE) → refined pose + triggers → resume (`bench/l1b_ndt_align_init.py`).

### L2 — per-frame WCET (`examples/wcet_frame.rs`)

Rust serial `align`, small 288-point cloud, 5 iterations, 20,000 frames (µs per frame):

| p50 | p99 | p99.9 | max |
|--:|--:|--:|--:|
| 524 | 647 | 859 | 1090 |

(min 513 µs, mean 529 µs.) The bounded, low-spread tail reflects the allocation-free, panic-free
align path (see [The WCET contract](../rt/wcet.md)).

### Caveats

These are a point-in-time snapshot under specific conditions — synthetic plane geometry, the
**serial** backend, **align-loop only** (map construction excluded), and one CPU. The L2 and L3 rows
use different cloud sizes (288 vs 30,603 points), so their absolute times are not comparable. Real
sequences, the parallel backend, or the full node pipeline will shift the ratios; treat the ≈3× as
indicative for this align workload, not a universal figure.

> Source: `bench/` (`ndt_bench_replay.cpp`, `run.sh`, `gen_report.py`, `ndt_bench.json`,
> `report.html`); `examples/wcet_frame.rs`; `CMakeLists.txt` (`NDT_BUILD_BENCH`).
