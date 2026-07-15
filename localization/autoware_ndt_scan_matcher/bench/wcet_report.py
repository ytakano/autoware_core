#!/usr/bin/env python3
# Copyright 2024 Autoware Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WCET comparison report (plan/ndt_wcet.md, M4/M5-EVT). Stdlib-only.

Inputs:
  wcet.json       ndt_bench_replay --fixture   (both engines, samples_ms per fixture)
  wcet_rust.json  engine example wcet_frame    (deterministic cost counters per fixture)
  --alloc a.json  optional second replay JSON from the LD_PRELOAD pass (allocs_per_align)

Produces a markdown report with:
  1. per-fixture tail table (p50 / p99 / p99.9 / max, C++ vs Rust, ratio, equal-work check)
  2. unit-cost regression  T_p50 ~= a*sum_neighbors + b*kd_nodes_visited + c  per engine
     (3-param least squares over the fixture set; a = kernel-eval cost, b = kd-node cost,
     c = fixed per-align overhead) -- ties measured time to the platform-independent counters
  3. moment-based Gumbel (pWCET) tail fit per engine on each fixture: block maxima (n=10),
     mu/beta by moments (beta = s*sqrt(6)/pi, mu = m - gamma*beta), exceedance table.
     This is MBPTA-style *approximation for comparison*, not a certified pWCET: samples are
     from one warm-cache container, block maxima are only asymptotically Gumbel, and the
     moment fit is cruder than PoT/MLE. Documented accordingly.
"""

import argparse
import json
import math
import sys

EULER_GAMMA = 0.5772156649015329


def pct(sorted_xs, p):
    if not sorted_xs:
        return float("nan")
    i = min(int(p * (len(sorted_xs) - 1)), len(sorted_xs) - 1)
    return sorted_xs[i]


def solve3(a, b):
    """Solve a 3x3 linear system (Gaussian elimination, partial pivot)."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = 3
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][3] / m[i][i] for i in range(n)]


def unit_cost_regression(rows):
    """rows: (sum_neighbors, kd_nodes, t_ms). Least squares for t = a*nbr + b*kd + c."""
    if len(rows) < 3:
        return None
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for nbr, kd, t in rows:
        x = (float(nbr), float(kd), 1.0)
        for i in range(3):
            for j in range(3):
                ata[i][j] += x[i] * x[j]
            atb[i] += x[i] * t
    sol = solve3(ata, atb)
    if sol is None:
        return None
    a, b, c = sol
    # R^2 for honesty about the fit quality.
    mean_t = sum(t for _, _, t in rows) / len(rows)
    ss_tot = sum((t - mean_t) ** 2 for _, _, t in rows)
    ss_res = sum((t - (a * nbr + b * kd + c)) ** 2 for nbr, kd, t in rows)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return a, b, c, r2


def gumbel_from_block_maxima(samples_ms, block=10):
    """Moment-fit a Gumbel to block maxima. Returns (mu, beta, n_blocks) or None."""
    if len(samples_ms) < 3 * block:
        return None
    maxima = [max(samples_ms[i : i + block]) for i in range(0, len(samples_ms) - block + 1, block)]
    n = len(maxima)
    m = sum(maxima) / n
    var = sum((x - m) ** 2 for x in maxima) / (n - 1)
    s = math.sqrt(var)
    if s <= 0.0:
        return None
    beta = s * math.sqrt(6.0) / math.pi
    mu = m - EULER_GAMMA * beta
    return mu, beta, n


def gumbel_quantile(mu, beta, exceed):
    """Value with per-block exceedance probability `exceed`."""
    return mu - beta * math.log(-math.log(1.0 - exceed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("timing_json", help="ndt_bench_replay --fixture output")
    ap.add_argument("rust_json", help="wcet_frame WCET_JSON output (counters)")
    ap.add_argument("--alloc", help="replay JSON from the LD_PRELOAD allocation pass")
    ap.add_argument("-o", "--out", help="output markdown path (default: stdout)")
    args = ap.parse_args()

    with open(args.timing_json, encoding="utf-8") as fh:
        timing = json.load(fh)
    with open(args.rust_json, encoding="utf-8") as fh:
        rust = json.load(fh)
    alloc = None
    if args.alloc:
        with open(args.alloc, encoding="utf-8") as fh:
            alloc = json.load(fh)

    fixtures = timing.get("fixtures", {})
    counters = rust.get("fixtures", {})
    lines = []
    w = lines.append

    w("# NDT WCET comparison — C++ vs Rust on frozen adversarial fixtures")
    w("")
    meta = timing.get("meta", {})
    w(
        f"Timing: {meta.get('iters', '?')} aligns/fixture/engine (+{meta.get('warmup', '?')} "
        f"warmup), serial (num_threads=1), steady_clock, align loop only."
    )
    env = timing.get("env", {})
    if env:
        w("")
        for k, v in env.items():
            w(f"- **{k}**: {v}")
    w("")

    # ---- 1. tail table ----
    w("## Frame-time tails (ms)")
    w("")
    w("| fixture | work check | engine | p50 | p99 | p99.9 | max | max ratio (Rust/C++) |")
    w("|---|---|---|---|---|---|---|---|")
    mismatch = False
    for name in sorted(fixtures):
        fx = fixtures[name]
        match = fx.get("iter_match", False)
        mismatch |= not match
        chk = f"iter {fx['cpp']['iteration_num']} == {fx['rust']['iteration_num']} ✓" if match else (
            f"**MISMATCH {fx['cpp']['iteration_num']} vs {fx['rust']['iteration_num']}**"
        )
        cs = sorted(fx["cpp"]["samples_ms"])
        rs = sorted(fx["rust"]["samples_ms"])
        ratio = (rs[-1] / cs[-1]) if cs and cs[-1] > 0 else float("nan")
        w(
            f"| {name} | {chk} | C++ | {pct(cs, 0.5):.2f} | {pct(cs, 0.99):.2f} "
            f"| {pct(cs, 0.999):.2f} | {cs[-1]:.2f} | |"
        )
        w(
            f"| | | Rust | {pct(rs, 0.5):.2f} | {pct(rs, 0.99):.2f} "
            f"| {pct(rs, 0.999):.2f} | {rs[-1]:.2f} | {ratio:.2f} |"
        )
    w("")
    if mismatch:
        w(
            "> **WARNING**: at least one fixture has an iteration mismatch — the engines did "
            "different work there and the timing comparison is NOT algorithm-fair on it."
        )
        w("")

    # ---- 2. allocation counts ----
    if alloc:
        w("## Heap allocations per align (LD_PRELOAD interposer)")
        w("")
        w("Rust is independently verified zero-alloc per frame by `engine/tests/zero_alloc.rs`;")
        w("the interposer measures the whole process, so the Rust column includes any harness-")
        w("side allocation as an upper bound.")
        w("")
        w("| fixture | C++ allocs/align | Rust allocs/align |")
        w("|---|---|---|")
        for name in sorted(alloc.get("fixtures", {})):
            fx = alloc["fixtures"][name]
            w(
                f"| {name} | {fx['cpp'].get('allocs_per_align', -1)} "
                f"| {fx['rust'].get('allocs_per_align', -1)} |"
            )
        w("")

    # ---- 3. unit-cost regression ----
    w("## Unit-cost regression (counters → time)")
    w("")
    w("`T_p50 ≈ a·sum_neighbors + b·kd_nodes_visited + c` over the fixture set — the")
    w("deterministic counters are identical for both engines (bit-exactness ⇒ same work),")
    w("so `a`/`b` compare the per-operation cost of the two implementations directly.")
    w("")
    rows = {"cpp": [], "rust": []}
    for name, fx in fixtures.items():
        c = counters.get(name, {}).get("counters")
        if not c or not fx.get("iter_match", False):
            continue
        for eng in ("cpp", "rust"):
            t50 = pct(sorted(fx[eng]["samples_ms"]), 0.5)
            rows[eng].append((c["sum_neighbors"], c["kd_nodes_visited"], t50))
    w("| engine | a (ns/kernel eval) | b (ns/kd node) | c (ms fixed) | R² | n |")
    w("|---|---|---|---|---|---|")
    for eng in ("cpp", "rust"):
        fit = unit_cost_regression(rows[eng])
        if fit is None:
            w(f"| {eng} | — | — | — | — | {len(rows[eng])} (need ≥ 3) |")
        else:
            a, b, c0, r2 = fit
            w(
                f"| {eng} | {a * 1e6:.1f} | {b * 1e6:.1f} | {c0:.3f} | {r2:.4f} "
                f"| {len(rows[eng])} |"
            )
    w("")

    # ---- 4. Gumbel pWCET ----
    w("## Gumbel tail fit (MBPTA-style, documented approximation)")
    w("")
    w("Block maxima (n=10) per fixture/engine, Gumbel by moments")
    w("(β = s·√6/π, μ = m − γβ). **Comparison aid, not a certified pWCET**: one warm-cache")
    w("container, moment fit, asymptotic block-maxima assumption. Quantiles are per-block")
    w("exceedance probabilities.")
    w("")
    w("| fixture | engine | μ (ms) | β (ms) | p=1e-3 | p=1e-6 | p=1e-9 | measured max |")
    w("|---|---|---|---|---|---|---|---|")
    for name in sorted(fixtures):
        fx = fixtures[name]
        for eng in ("cpp", "rust"):
            xs = fx[eng]["samples_ms"]
            g = gumbel_from_block_maxima(xs)
            if g is None:
                w(f"| {name} | {eng} | — | — | — | — | — | {max(xs):.2f} |")
                continue
            mu, beta, _ = g
            q = [gumbel_quantile(mu, beta, p) for p in (1e-3, 1e-6, 1e-9)]
            w(
                f"| {name} | {eng} | {mu:.2f} | {beta:.3f} | {q[0]:.2f} | {q[1]:.2f} "
                f"| {q[2]:.2f} | {max(xs):.2f} |"
            )
    w("")
    w("Counters per fixture (serial baseline, platform-independent):")
    w("")
    w("| fixture | iterations | derivative passes | Σ neighbors | kd nodes visited |")
    w("|---|---|---|---|---|")
    for name in sorted(counters):
        e = counters[name]
        c = e.get("counters")
        if c:
            w(
                f"| {name} | {e.get('iteration_num', '?')} | {c['derivative_passes']} "
                f"| {c['sum_neighbors']} | {c['kd_nodes_visited']} |"
            )
    w("")

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wcet_report: wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
