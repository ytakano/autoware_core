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

//! Deterministic algorithmic-cost counters for the WCET analysis (`plan/ndt_wcet.md`, Layer 2).
//!
//! The engine's frame time decomposes as
//! `T ≤ N_iter × [Σ_p (T_search(p) + K(p)·T_kernel) + T_solve]`, so these counters capture the only
//! data-dependent cost drivers: derivative passes (≤ iterations + 1), points processed, neighbors
//! collected (K ≤ [`crate::ndt::MAX_NEIGHBORS`] per point per pass), and kd-tree nodes visited.
//! Being platform-independent and reproducible, they are the worst-input **search fitness** (far
//! more stable than wall time) and the machine-checked link to the analytic bound
//! (`tests/wcet_bounds.rs`).
//!
//! Only compiled under the `wcet-count` feature; the shipping hot path carries zero instrumentation
//! cost when the feature is off. The counters cover the **serial** backend (the WCET baseline) —
//! the `parallel` reduction does not accumulate them.

/// Algorithmic-cost counters for one [`crate::ndt::align`] call (reset at the start of each align).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WcetCounters {
    /// Derivative computations performed: 1 (initial guess) + 1 per Newton iteration.
    pub derivative_passes: u64,
    /// Source points processed, summed over all derivative passes.
    pub points_processed: u64,
    /// Neighbor leaves collected, summed over all points and passes
    /// (≤ `points_processed × MAX_NEIGHBORS`).
    pub sum_neighbors: u64,
    /// kd-tree nodes visited by the neighbor searches, summed over all points and passes
    /// (≤ `points_processed × tree node count`).
    pub kd_nodes_visited: u64,
}

impl WcetCounters {
    /// All-zero counters (`const`, usable from `AlignWorkspace::new`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            derivative_passes: 0,
            points_processed: 0,
            sum_neighbors: 0,
            kd_nodes_visited: 0,
        }
    }
}
