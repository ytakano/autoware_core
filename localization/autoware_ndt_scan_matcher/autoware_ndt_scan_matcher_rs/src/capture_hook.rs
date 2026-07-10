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

//! Env-gated real-drive input capture (`NDT_CAPTURE_DIR`) — the WCET operational-envelope hook.
//!
//! When the environment variable is set, the FFI boundary records everything the C++ node feeds
//! the Rust engine into the [`autoware_ndt_rs::capture`] sidecar format: params once, each map
//! tile once (write-once by cell id, from `add_target_str`), and per align a frame file (guess +
//! active tile-id set + source cloud). Offline replayers then re-run the drive with the cost
//! counters (`wcet_frame --capture`) and through both engines (`ndt_bench_replay --capture`).
//!
//! Design constraints: **zero effect when the env is unset** (one `OnceLock` check), and
//! **best-effort when set** — IO errors are swallowed (`.ok()`): this is opt-in diagnostic
//! tooling and must never panic or alter the align path. Tiles registered through the numeric
//! `add_target` (bench-only path) are not captured; the node uses the string-id API.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use autoware_ndt_rs::capture;
use autoware_ndt_rs::engine::NdtEngine;
use autoware_ndt_rs::nalgebra::Matrix4;
use autoware_ndt_rs::ndt::NdtParams;

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);

fn dir() -> Option<&'static PathBuf> {
    DIR.get_or_init(|| {
        let d = std::env::var_os("NDT_CAPTURE_DIR").map(PathBuf::from)?;
        std::fs::create_dir_all(&d).ok()?;
        Some(d)
    })
    .as_ref()
}

/// Record the align parameters (constant per run; harmless overwrite on repeated calls).
pub(crate) fn params(
    trans_epsilon: f64,
    step_size: f64,
    resolution: f64,
    max_iterations: i32,
    outlier_ratio: f64,
) {
    if let Some(d) = dir() {
        let p = NdtParams {
            trans_epsilon,
            step_size,
            resolution,
            max_iterations,
            outlier_ratio,
            regularization: None,
            num_threads: 1,
        };
        capture::write_params(d, &p).ok();
    }
}

/// Record a map tile keyed by its cell-id bytes (write-once per id).
pub(crate) fn tile(id: &[u8], points: &[[f32; 3]]) {
    if let Some(d) = dir() {
        capture::write_tile(d, id, points).ok();
    }
}

/// Record one align input: guess + the engine's current (sorted) tile-id set + source cloud.
pub(crate) fn align(engine: &NdtEngine, guess: &Matrix4<f32>, source: &[[f32; 3]]) {
    if let Some(d) = dir() {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let ids = engine.map_ids();
        capture::write_frame(d, seq, guess, &ids, source).ok();
    }
}
