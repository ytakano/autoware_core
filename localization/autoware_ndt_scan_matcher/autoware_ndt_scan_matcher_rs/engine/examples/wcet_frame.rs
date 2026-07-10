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

//! Frame-time (WCET) benchmark for `align` — a manual diagnostic, NOT a CI gate (timing is
//! environment-dependent). Reports min / mean / p50 / p99 / p99.9 / max over many frames; for
//! WCET, the **max + tail** matter (not the mean).
//!
//! Two modes:
//! - `cargo run --release --example wcet_frame` — fixed synthetic fixture (regression watch).
//! - `cargo run --release --example wcet_frame -- FIXTURE.ndtfix [...]` — replay frozen fixtures
//!   (`bench/fixtures/*.ndtfix` from `wcet_fixtures` / `wcet_search`), one HWM row each. With
//!   `--features wcet-count` each row also reports the deterministic cost counters.
//!
//! `WCET_FRAMES` overrides the per-fixture frame count (default 20000 synthetic / 2000 fixture).
//! `WCET_JSON=path` (fixture mode) additionally writes a machine-readable JSON — per fixture:
//! sizes, `iteration_num`, timing samples (µs), and (under `wcet-count`) the cost counters —
//! consumed by `bench/wcet_report.py` for the unit-cost regression.
//! Record baselines in `porting_notes/ndt_wcet_audit.md`. (Single core, warm cache — a relative
//! regression watch, not a hardware WCET proof.)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::arithmetic_side_effects,
    clippy::let_underscore_must_use,
    clippy::allow_attributes,
    reason = "benchmark example (diagnostic, not a gate)"
)]

use std::time::Instant;

use autoware_ndt_rs::fixture::Fixture;
use autoware_ndt_rs::ndt::{AlignResult, AlignWorkspace, NdtParams, align};
use autoware_ndt_rs::voxel_grid::VoxelGridMap;
use nalgebra::Matrix4;

fn dense_cluster(cx: f32, cy: f32, cz: f32) -> Vec<[f32; 3]> {
    (0..8)
        .map(|i| {
            let f = i as f32 * 0.02;
            [cx + f, cy - f, cz + 0.5 * f]
        })
        .collect()
}

struct Stats {
    nanos: Vec<u128>,
}

impl Stats {
    fn measure(
        frames: usize,
        map: &VoxelGridMap,
        source: &[[f32; 3]],
        guess: &Matrix4<f32>,
        params: &NdtParams,
        ws: &mut AlignWorkspace,
        out: &mut AlignResult,
    ) -> Self {
        // Warmup (JIT-free, but page/cache/branch warmup + buffer growth).
        for _ in 0..(frames / 20).clamp(10, 100) {
            align(map, source, guess, params, ws, out);
        }
        let mut nanos: Vec<u128> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let t = Instant::now();
            align(map, source, guess, params, ws, out);
            nanos.push(t.elapsed().as_nanos());
        }
        nanos.sort_unstable();
        Self { nanos }
    }

    fn pct(&self, p: f64) -> u128 {
        let n = self.nanos.len();
        self.nanos[((p * (n as f64 - 1.0)) as usize).min(n - 1)]
    }

    fn mean(&self) -> u128 {
        self.nanos.iter().sum::<u128>() / (self.nanos.len() as u128)
    }
}

fn us(n: u128) -> f64 {
    (n as f64) / 1000.0
}

fn frames_from_env(default: usize) -> usize {
    std::env::var("WCET_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Synthetic default fixture (the original regression-watch numbers).
fn run_synthetic() {
    let frames = frames_from_env(20_000);
    let mut map_points: Vec<[f32; 3]> = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            map_points.extend(dense_cluster((i as f32) * 3.0, (j as f32) * 3.0, 0.0));
        }
    }
    let source: Vec<[f32; 3]> = map_points
        .iter()
        .map(|p| [p[0] + 0.3, p[1] - 0.2, p[2] + 0.1])
        .collect();

    let mut map = VoxelGridMap::new([2.0, 2.0, 2.0], 6, 0.01);
    map.add_target(&map_points, 0);
    map.create_kdtree();

    let params = NdtParams {
        trans_epsilon: 0.01,
        step_size: 0.1,
        resolution: 2.0,
        max_iterations: 30,
        outlier_ratio: 0.55,
        regularization: None,
        num_threads: 1,
    };
    let guess = Matrix4::identity();
    let mut ws = AlignWorkspace::with_capacity(source.len());
    let mut out = AlignResult::default();

    let s = Stats::measure(frames, &map, &source, &guess, &params, &mut ws, &mut out);
    let n = s.nanos.len();
    println!(
        "align frame time over {frames} frames ({} src pts, {} iters):",
        source.len(),
        out.iteration_num
    );
    println!("  min  {:.2} us", us(s.nanos[0]));
    println!("  mean {:.2} us", us(s.mean()));
    println!("  p50  {:.2} us", us(s.pct(0.50)));
    println!("  p99  {:.2} us", us(s.pct(0.99)));
    println!("  p99.9 {:.2} us", us(s.pct(0.999)));
    println!("  max  {:.2} us", us(s.nanos[n - 1]));
}

/// Replay frozen fixtures: one HWM row per file (+ cost counters under `wcet-count`).
fn run_fixtures(paths: &[String]) {
    let frames = frames_from_env(2_000);
    let mut json = String::new();
    json.push_str("{\n  \"fixtures\": {\n");
    let mut first = true;
    println!("align frame time over {frames} frames/fixture (serial, pre-reserved workspace):");
    println!(
        "  {:16} {:>6} {:>6} {:>5} | {:>9} {:>9} {:>9} {:>9} | extra",
        "fixture", "map", "src", "iter", "p50 us", "p99 us", "p99.9 us", "max us"
    );
    for path in paths {
        let fx = Fixture::read(std::path::Path::new(path)).expect("read fixture");
        let mut map = VoxelGridMap::new([fx.params.resolution; 3], 6, 0.01);
        for (id, tile) in fx.tiles.iter().enumerate() {
            map.add_target(tile, id as u64);
        }
        map.create_kdtree();
        let mut ws = AlignWorkspace::with_capacity(fx.source.len());
        let mut out = AlignResult::default();
        let s = Stats::measure(
            frames, &map, &fx.source, &fx.guess, &fx.params, &mut ws, &mut out,
        );
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let n = s.nanos.len();
        print!(
            "  {name:16} {:>6} {:>6} {:>5} | {:>9.1} {:>9.1} {:>9.1} {:>9.1} |",
            fx.map_len(),
            fx.source.len(),
            out.iteration_num,
            us(s.pct(0.50)),
            us(s.pct(0.99)),
            us(s.pct(0.999)),
            us(s.nanos[n - 1]),
        );
        #[cfg(feature = "wcet-count")]
        {
            let c = out.counters;
            let kbar = if c.points_processed == 0 {
                0.0
            } else {
                c.sum_neighbors as f64 / c.points_processed as f64
            };
            print!(
                " passes={} K̄={kbar:.1} kd/pt={:.1}",
                c.derivative_passes,
                c.kd_nodes_visited as f64 / c.points_processed.max(1) as f64
            );
        }
        println!();

        // Machine-readable entry for bench/wcet_report.py (WCET_JSON mode).
        {
            use std::fmt::Write as _;
            if !first {
                json.push_str(",\n");
            }
            first = false;
            let _ = write!(
                json,
                "    \"{name}\": {{\n      \"n_map\": {},\n      \"n_tiles\": {},\n      \
                 \"n_source\": {},\n      \"iteration_num\": {}",
                fx.map_len(),
                fx.tiles.len(),
                fx.source.len(),
                out.iteration_num
            );
            #[cfg(feature = "wcet-count")]
            {
                let c = out.counters;
                let _ = write!(
                    json,
                    ",\n      \"counters\": {{ \"derivative_passes\": {}, \"points_processed\": \
                     {}, \"sum_neighbors\": {}, \"kd_nodes_visited\": {} }}",
                    c.derivative_passes, c.points_processed, c.sum_neighbors, c.kd_nodes_visited
                );
            }
            json.push_str(",\n      \"samples_us\": [");
            for (i, ns) in s.nanos.iter().enumerate() {
                let _ = write!(json, "{}{:.3}", if i == 0 { "" } else { "," }, us(*ns));
            }
            json.push_str("]\n    }");
        }
    }
    json.push_str("\n  }\n}\n");
    if let Ok(path) = std::env::var("WCET_JSON") {
        std::fs::write(&path, json).expect("write WCET_JSON");
        eprintln!("wcet_frame: wrote {path}");
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        run_synthetic();
    } else {
        run_fixtures(&paths);
    }
}
