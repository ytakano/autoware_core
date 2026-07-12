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

"""Profile-B measurement-campaign orchestrator (stdlib-only).

Implements the controlled-engine profile of plan/ndt_timing_measurement_policy.md around
the existing `ndt_bench_replay --fixture` binary: programmatic environment verification with
abort rules, warm/cold series, one-resource-at-a-time co-runners, per-session fixture-order
randomization + per-fixture engine-order randomization, sample tiers, and the policy's full
reproducibility manifest embedded in every output JSON.

Commands:
  verify-env                  run the environment checks and report (no measurement)
  plan                        print the run matrix + a rough duration estimate
  run --session N [...]       execute one session (all series, or --series to restrict)
  merge SESSION_DIR           pair single-engine cells and emit merged per-series JSONs

Typical flow on the prepared host (see CAMPAIGN.md for the root-side knobs):
  python3 wcet_campaign.py verify-env
  python3 wcet_campaign.py plan
  python3 wcet_campaign.py run --session 1
  python3 wcet_campaign.py merge campaign_runs/session-1

Dev-only: `run --allow-env-mismatch` downgrades environment aborts to warnings so the
plumbing can be smoke-tested inside the (powersave, unisolated) dev container; the override
is recorded in the manifest as env_check="overridden".
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid

BENCH_DIR = pathlib.Path(__file__).resolve().parent
PKG = "autoware_ndt_scan_matcher"
PKG_REL = pathlib.Path("src/core/autoware_core/localization") / PKG / "bench"
# Rough per-align cost priors (ms, C++ warm) for the duration estimate only.
EST_CPP_MS = {
    "search_00": 900, "search_01": 620, "dense_neighbors": 600, "max_iterations": 90,
    "cache_hostile": 20, "subnormal": 55, "legal_worst": 165, "legal_osc": 120,
}
RUST_FACTOR = 0.7  # Rust/C++ median ratio prior
COLD_OVERHEAD_MS = 40  # per-sample eviction cost prior


def ws_root():
    p = BENCH_DIR
    rel = str(PKG_REL)
    if str(p).endswith(rel):
        return pathlib.Path(str(p)[: -len(rel)].rstrip("/"))
    return pathlib.Path.cwd()


def read_sys(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def sh(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Environment verification (policy: Frequency and Thermal Control)
# ---------------------------------------------------------------------------

def cpu_sys(cpu, leaf):
    return f"/sys/devices/system/cpu/cpu{cpu}/{leaf}"


def read_freq_khz(cpu):
    v = read_sys(cpu_sys(cpu, "cpufreq/scaling_cur_freq"))
    return int(v) if v and v.isdigit() else None


def read_max_freq_khz(cpu):
    for leaf in ("cpufreq/cpuinfo_max_freq", "cpufreq/scaling_max_freq"):
        v = read_sys(cpu_sys(cpu, leaf))
        if v and v.isdigit():
            return int(v)
    return None


def nominal_freq_khz(cfg):
    """Reference frequency for the sag guard: config override, else sysfs base_frequency.

    `cpuinfo_max_freq` is the *boost* ceiling on most parts and is the wrong reference —
    sustained load legitimately runs below it. Returns (khz, source) or (None, reason).
    """
    cpu = cfg["benchmark_cpu"]
    override = cfg["abort"].get("nominal_khz")
    if override:
        return int(override), "config abort.nominal_khz"
    v = read_sys(cpu_sys(cpu, "cpufreq/base_frequency"))
    if v and v.isdigit():
        return int(v), "sysfs base_frequency"
    return None, "no abort.nominal_khz in config and no sysfs base_frequency"


def read_temps_c():
    temps = {}
    base = pathlib.Path("/sys/class/thermal")
    if not base.is_dir():
        return temps
    for zone in sorted(base.glob("thermal_zone*")):
        t = read_sys(zone / "temp")
        ty = read_sys(zone / "type") or zone.name
        if t and t.lstrip("-").isdigit():
            temps[ty] = int(t) / 1000.0
    return temps


def irqs_allowed_on(cpu):
    """Count IRQs whose smp_affinity_list includes `cpu` (best-effort)."""
    count = 0
    base = pathlib.Path("/proc/irq")
    if not base.is_dir():
        return None
    for irq in base.iterdir():
        lst = read_sys(irq / "smp_affinity_list")
        if not lst:
            continue
        for part in lst.split(","):
            rng = part.split("-")
            try:
                lo = int(rng[0])
                hi = int(rng[-1])
            except ValueError:
                continue
            if lo <= cpu <= hi:
                count += 1
                break
    return count


def verify_environment(cfg):
    """Return (failures, warnings, snapshot). Failures abort a real run."""
    cpu = cfg["benchmark_cpu"]
    failures, warnings = [], []
    snap = {}

    gov = read_sys(cpu_sys(cpu, "cpufreq/scaling_governor"))
    snap["governor"] = gov
    if gov != "performance":
        failures.append(
            f"governor on cpu{cpu} is '{gov}', need 'performance' "
            f"(fix: echo performance | sudo tee {cpu_sys(cpu, 'cpufreq/scaling_governor')})"
        )

    isolated = read_sys("/sys/devices/system/cpu/isolated") or ""
    snap["isolated_cpus"] = isolated
    iso_set = set()
    for part in isolated.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            if lo.isdigit() and hi.isdigit():
                iso_set.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            iso_set.add(int(part))
    if cpu not in iso_set:
        warnings.append(
            f"cpu{cpu} not in /sys/devices/system/cpu/isolated ('{isolated or 'empty'}'); "
            "use isolcpus= or cset shield, or record the alternative mechanism in notes"
        )

    siblings = read_sys(cpu_sys(cpu, "topology/thread_siblings_list")) or str(cpu)
    snap["smt_siblings"] = siblings
    sib_ids = set()
    for part in siblings.split(","):
        rng = part.split("-")
        try:
            sib_ids.update(range(int(rng[0]), int(rng[-1]) + 1))
        except ValueError:
            pass
    sib_ids.discard(cpu)
    for sib in sorted(sib_ids):
        online = read_sys(cpu_sys(sib, "online"))
        if online == "0":
            snap[f"smt_sibling_{sib}"] = "offline"
        else:
            warnings.append(
                f"SMT sibling cpu{sib} of cpu{cpu} is online "
                f"(fix: echo 0 | sudo tee {cpu_sys(sib, 'online')}; or keep it idle and note it)"
            )
            snap[f"smt_sibling_{sib}"] = "online"

    n_irqs = irqs_allowed_on(cpu)
    snap["irqs_allowed_on_benchmark_cpu"] = n_irqs
    if n_irqs is not None and n_irqs > 8:
        warnings.append(
            f"{n_irqs} IRQs may fire on cpu{cpu} "
            "(fix: move IRQ affinities away via /proc/irq/*/smp_affinity_list)"
        )

    freq = read_freq_khz(cpu)
    fmax = read_max_freq_khz(cpu)
    snap["cur_freq_khz"] = freq
    snap["max_freq_khz"] = fmax
    snap["temps_c"] = read_temps_c()
    return failures, warnings, snap


def series_guard(cfg, during, label):
    """Abort rules over in-series samples: sustained frequency sag + thermal ceiling.

    `during` is the list of (freq_khz, temps_c) samples polled while the replay ran; idle
    (pre-series) frequency readings are C-state artifacts and are deliberately not judged.
    On nohz_full/isolated cores the kernel's frequency reporting goes stale (a constant
    boot/idle value), so a constant reading disables this check — the calibration guard
    (measured sustained speed) is the primary throttling detector.
    """
    problems = []
    freqs = sorted(f for f, _ in during if f)
    nominal, src = nominal_freq_khz(cfg)
    frac = cfg["abort"]["min_freq_frac"]
    if freqs and len(set(freqs)) > 1 and nominal is not None:
        med = freqs[len(freqs) // 2]
        if med < frac * nominal:
            problems.append(
                f"{label}: in-series median frequency {med} kHz < {frac:.0%} of nominal "
                f"{nominal} kHz ({src}) -- throttling?"
            )
    max_temp = cfg["abort"]["max_temp_c"]
    for _, temps in during:
        hot = [f"{k}={v:.0f}C" for k, v in (temps or {}).items() if v > max_temp]
        if hot:
            problems.append(f"{label}: in-series temperature above ceiling: {', '.join(hot)}")
            break
    return problems


def run_calibration(cfg, corunner_exe, n=3):
    """Median of n fixed-work spins pinned to the benchmark CPU, in ns (None on failure).

    This measures sustained compute speed directly, which survives the stale frequency
    reporting of nohz_full/isolated cores (validated on this host: an isolated core
    reported 400 MHz under load while outperforming a non-isolated 3.1 GHz core).
    """
    times = []
    for _ in range(n):
        out = sh(["taskset", "-c", str(cfg["benchmark_cpu"]), str(corunner_exe), "calib"])
        if out and out.strip().lstrip("-").isdigit():
            times.append(int(out.strip()))
    if not times:
        return None
    return sorted(times)[len(times) // 2]


# ---------------------------------------------------------------------------
# Manifest (policy: Reproducibility Manifest)
# ---------------------------------------------------------------------------

def cmake_flags(root):
    cache = root / "build" / PKG / "CMakeCache.txt"
    flags = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            for key in ("CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE",
                        "CMAKE_EXE_LINKER_FLAGS"):
                if line.startswith(key + ":"):
                    flags[key] = line.split("=", 1)[-1]
    return flags


def cargo_profile(root):
    manifest = (root / "src/core/autoware_core/localization" / PKG /
                "autoware_ndt_scan_matcher_rs" / "Cargo.toml")
    if not manifest.is_file():
        return None
    lines = manifest.read_text(encoding="utf-8").splitlines()
    grab, out = False, []
    for line in lines:
        if line.strip().startswith("[profile."):
            grab = True
        elif line.strip().startswith("[") and grab:
            grab = False
        if grab:
            out.append(line)
    return "\n".join(out) or None


def git_commit(path):
    return sh(["git", "-C", str(path), "rev-parse", "HEAD"]) or None


def build_manifest(cfg, root, session_id, series, cell=None, env_snap=None, env_mode="strict"):
    fixtures_dir = BENCH_DIR / cfg["fixtures_dir"]
    exe = find_replay(root)
    m = {
        "experiment_id": f"{session_id}/{series}" + (f"/{cell}" if cell else ""),
        "measurement_profile": cfg["profile"],
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_name": platform.node(),
        "cpu_model": next(
            (ln.split(":", 1)[1].strip() for ln in
             pathlib.Path("/proc/cpuinfo").read_text().splitlines()
             if ln.startswith("model name")), None),
        "cpu_microcode": next(
            (ln.split(":", 1)[1].strip() for ln in
             pathlib.Path("/proc/cpuinfo").read_text().splitlines()
             if ln.startswith("microcode")), None),
        "kernel_version": " ".join(platform.uname()[:3]),
        "governor": (env_snap or {}).get("governor"),
        "isolated_cpus": (env_snap or {}).get("isolated_cpus"),
        "benchmark_cpu": cfg["benchmark_cpu"],
        "affinity_mask": f"taskset -c {cfg['benchmark_cpu']}",
        "irq_configuration": {
            "irqs_allowed_on_benchmark_cpu":
                (env_snap or {}).get("irqs_allowed_on_benchmark_cpu")},
        "smt_configuration": (env_snap or {}).get("smt_siblings"),
        "memory_configuration": next(
            (ln for ln in pathlib.Path("/proc/meminfo").read_text().splitlines()
             if ln.startswith("MemTotal")), None),
        "compiler_versions": {
            "cxx": sh(["c++", "--version"]).splitlines()[0] if sh(["c++", "--version"]) else None,
            "rustc": sh(["rustc", "--version"]) or None,
        },
        "compiler_flags": cmake_flags(root),
        "linker_flags": cmake_flags(root).get("CMAKE_EXE_LINKER_FLAGS"),
        "cargo_profile": cargo_profile(root),
        "allocator": sh(["getconf", "GNU_LIBC_VERSION"]) or None,
        "cpp_commit": git_commit(root / "src/core/autoware_core"),
        "rust_commit": git_commit(root / "src/core/autoware_core"),
        "fixture_hashes": {
            p.stem: sha256_of(p) for p in sorted(fixtures_dir.glob("*.ndtfix"))},
        "binary_hash": sha256_of(exe) if exe else None,
        "cache_condition": "cold" if series == "cold" else "warm",
        "co_runner": series.split(":", 1)[1] if series.startswith("corunner:") else None,
        "env_check": env_mode,
        "dataset_id": None,
        "target_board": None,
        "firmware": None,
        "qemu_version": None,
        "notes": cfg.get("note"),
    }
    return m


# ---------------------------------------------------------------------------
# Planning and execution
# ---------------------------------------------------------------------------

def find_replay(root):
    exe = root / "build" / PKG / "ndt_bench_replay"
    return exe if exe.is_file() else None


def fixture_tier(cfg, name):
    for tier, spec in cfg["tiers"].items():
        if name in spec["fixtures"]:
            return tier, spec["samples"]
    return None, None


def build_plan(cfg, session):
    """List of cells: dicts with series/fixture/engine/samples/warmup/evict/corunner."""
    rng = random.Random(0xC0FFEE ^ session)
    all_fixtures = [f for spec in cfg["tiers"].values() for f in spec["fixtures"]]
    series_list = ["warm", "cold"] + [f"corunner:{m}" for m in cfg["corunner"]["modes"]]
    cells = []
    for series in series_list:
        if series == "warm":
            fixtures = list(all_fixtures)
        elif series == "cold":
            fixtures = list(all_fixtures)
        else:
            fixtures = list(cfg["corunner"]["fixtures"])
        rng.shuffle(fixtures)  # fixture-order randomization (policy amendment)
        for fx in fixtures:
            _, tier_samples = fixture_tier(cfg, fx)
            if series == "warm":
                samples, warmup, evict = tier_samples, cfg["warmup"], 0
            elif series == "cold":
                samples = cfg["cold"]["samples"]
                warmup = cfg["cold"]["warmup"]
                evict = cfg["cold"]["evict_mib"] * 1024 * 1024
            else:
                samples, warmup, evict = cfg["corunner"]["samples"], cfg["warmup"], 0
            engines = ["cpp", "rust"]
            rng.shuffle(engines)  # engine-order randomization per fixture
            for eng in engines:
                cells.append({
                    "series": series, "fixture": fx, "engine": eng,
                    "samples": samples, "warmup": warmup, "evict_bytes": evict,
                    "corunner": series.split(":", 1)[1] if ":" in series else None,
                })
    return cells


def estimate_ms(cell):
    per = EST_CPP_MS.get(cell["fixture"], 200)
    if cell["engine"] == "rust":
        per *= RUST_FACTOR
    per += COLD_OVERHEAD_MS if cell["evict_bytes"] else 0
    return (cell["samples"] + cell["warmup"]) * per


def cmd_plan(cfg, args):
    total = 0.0
    for session in range(1, cfg["sessions"] + 1):
        cells = build_plan(cfg, session)
        dur = sum(estimate_ms(c) for c in cells) / 1000.0
        total += dur
        print(f"session {session}: {len(cells)} cells, ~{dur / 60:.0f} min")
        if session == 1 and args.verbose:
            for c in cells:
                print(f"  {c['series']:14} {c['fixture']:16} {c['engine']:4} "
                      f"n={c['samples']} warmup={c['warmup']} evict={c['evict_bytes']}")
    print(f"total (all {cfg['sessions']} sessions): ~{total / 3600:.1f} h "
          "(priors-based estimate; excludes map builds and env checks)")
    return 0


def cmd_verify_env(cfg, _args):
    failures, warnings, snap = verify_environment(cfg)
    for w in warnings:
        print(f"WARN: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    print(json.dumps(snap, indent=1))
    return 1 if failures else 0


def corunner_exe(out_dir):
    src = BENCH_DIR / "corunner.c"
    exe = out_dir / "corunner"
    if not exe.is_file():
        subprocess.run(["cc", "-O2", "-o", str(exe), str(src)], check=True)
    return exe


def start_corunner(cfg, mode, out_dir):
    exe = corunner_exe(out_dir)
    mib = cfg["corunner"]["mib"].get(mode)
    argv = ["taskset", "-c", str(cfg["corunner_cpu"]), str(exe), mode]
    if mib:
        argv.append(str(mib))
    return subprocess.Popen(argv)


def cmd_run(cfg, args):
    root = ws_root()
    exe = find_replay(root)
    if exe is None:
        print(f"FAIL: ndt_bench_replay not found under {root}/build/{PKG} "
              "(build with -DNDT_BUILD_BENCH=ON)", file=sys.stderr)
        return 1
    failures, warnings, snap0 = verify_environment(cfg)
    env_mode = "strict"
    if failures:
        if not args.allow_env_mismatch:
            for f in failures:
                print(f"FAIL: {f}", file=sys.stderr)
            print("environment not ready; fix the knobs above or use --allow-env-mismatch "
                  "for a dev smoke test", file=sys.stderr)
            return 1
        env_mode = "overridden"
        for f in failures:
            print(f"WARN (overridden): {f}", file=sys.stderr)
    for w in warnings:
        print(f"WARN: {w}", file=sys.stderr)

    session_id = f"session-{args.session}"
    out_root = BENCH_DIR / cfg["output_dir"] / session_id
    out_root.mkdir(parents=True, exist_ok=True)
    fixtures_dir = BENCH_DIR / cfg["fixtures_dir"]
    # Session throttling baseline: measured sustained speed on the benchmark CPU.
    calib_exe = corunner_exe(out_root)
    calib_base_ns = run_calibration(cfg, calib_exe)
    if calib_base_ns is None:
        print("WARN: calibration spin failed; throttling guard limited to sysfs readings",
              file=sys.stderr)
    else:
        print(f"calibration baseline: {calib_base_ns / 1e6:.1f} ms "
              f"(guard: > x{cfg['abort'].get('max_calib_slowdown', 1.1)})")
    cells = build_plan(cfg, args.session)
    if args.series:
        cells = [c for c in cells if c["series"] == args.series]
    if args.max_cells:
        cells = cells[: args.max_cells]
    print(f"{session_id}: {len(cells)} cells")

    rc = 0
    for i, cell in enumerate(cells, 1):
        series_dir = out_root / cell["series"].replace(":", "_")
        series_dir.mkdir(parents=True, exist_ok=True)
        out_json = series_dir / f"{cell['fixture']}__{cell['engine']}.json"
        fixture_path = fixtures_dir / f"{cell['fixture']}.ndtfix"
        if not fixture_path.is_file():
            print(f"FAIL: missing fixture {fixture_path}", file=sys.stderr)
            rc = 1
            continue

        _, _, before = verify_environment(cfg)
        co = None
        if cell["corunner"]:
            co = start_corunner(cfg, cell["corunner"], out_root)
            time.sleep(1.0)  # let the co-runner reach steady state
        env = dict(os.environ)
        env.update({
            "WCET_ITERS": str(cell["samples"]),
            "WCET_WARMUP": str(cell["warmup"]),
            "WCET_ENGINE": cell["engine"],
            "WCET_EVICT_BYTES": str(cell["evict_bytes"]),
        })
        argv = ["taskset", "-c", str(cfg["benchmark_cpu"]), str(exe),
                "--fixture", str(out_json), str(fixture_path)]
        print(f"[{i}/{len(cells)}] {cell['series']} {cell['fixture']} {cell['engine']} "
              f"n={cell['samples']}")
        # Poll the effective frequency + temperatures while the replay runs (policy:
        # "record the effective CPU frequency when possible"); idle readings before the
        # series are C-state artifacts and are not judged.
        during = []
        child = subprocess.Popen(argv, env=env)
        while child.poll() is None:
            during.append((read_freq_khz(cfg["benchmark_cpu"]), read_temps_c()))
            time.sleep(1.0)
        proc_rc = child.returncode
        if co is not None:
            co.send_signal(signal.SIGTERM)
            co.wait(timeout=10)
        _, _, after = verify_environment(cfg)
        problems = series_guard(cfg, during, f"{cell['series']}/{cell['fixture']}")
        nominal, nominal_src = nominal_freq_khz(cfg)
        if nominal is None and i == 1:
            print(f"WARN: frequency-sag guard unavailable ({nominal_src})", file=sys.stderr)
        # Calibration guard: sustained-speed regression vs the session baseline detects
        # throttling even where the kernel's frequency reporting is stale (isolated cores).
        calib_ns = run_calibration(cfg, calib_exe, n=1)
        if calib_base_ns and calib_ns:
            slowdown = calib_ns / calib_base_ns
            if slowdown > cfg["abort"].get("max_calib_slowdown", 1.1):
                problems.append(
                    f"{cell['series']}/{cell['fixture']}: calibration spin x{slowdown:.2f} "
                    "slower than the session baseline -- throttling?")
        if proc_rc != 0:
            problems.append(f"replay exited {proc_rc}")
        freq_samples = [f for f, _ in during if f]
        sidecar = {
            "cell": cell,
            "env_before": before,
            "env_after": after,
            "freq_khz_during": {
                "n": len(freq_samples),
                "median": sorted(freq_samples)[len(freq_samples) // 2] if freq_samples else None,
                "min": min(freq_samples) if freq_samples else None,
                "max": max(freq_samples) if freq_samples else None,
                "nominal": nominal,
                "nominal_source": nominal_src,
                "stale": bool(freq_samples) and len(set(freq_samples)) <= 1,
            },
            "calibration": {
                "baseline_ns": calib_base_ns,
                "after_cell_ns": calib_ns,
                "slowdown": (calib_ns / calib_base_ns)
                if (calib_base_ns and calib_ns) else None,
            },
            "problems": problems,
            "manifest": build_manifest(cfg, root, session_id, cell["series"],
                                       cell=f"{cell['fixture']}__{cell['engine']}",
                                       env_snap=snap0, env_mode=env_mode),
        }
        out_json.with_suffix(".cell.json").write_text(
            json.dumps(sidecar, indent=1), encoding="utf-8")
        if problems:
            for p in problems:
                print(f"{'WARN' if env_mode == 'overridden' else 'ABORT'}: {p}",
                      file=sys.stderr)
            if env_mode != "overridden":
                print("aborting the session per the policy's abort rules; re-run this "
                      "series after fixing the environment", file=sys.stderr)
                return 1
            rc = max(rc, 2)
    print(f"done -> {out_root}")
    return rc


def cmd_merge(cfg, args):
    root = ws_root()
    session_dir = pathlib.Path(args.session_dir)
    rc = 0
    for series_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        merged = {}
        meta = None
        for cell_json in sorted(series_dir.glob("*__*.json")):
            if cell_json.name.endswith(".cell.json"):
                continue
            data = json.loads(cell_json.read_text(encoding="utf-8"))
            meta = meta or data.get("meta")
            for name, fx in data.get("fixtures", {}).items():
                slot = merged.setdefault(name, {
                    k: v for k, v in fx.items() if k not in ("cpp", "rust")})
                for eng in ("cpp", "rust"):
                    if eng in fx:
                        slot[eng] = fx[eng]
        # Re-assert equal work across the engine pair (policy: fairness requirements).
        for name, fx in merged.items():
            if "cpp" in fx and "rust" in fx:
                fx["iter_match"] = (
                    fx["cpp"]["iteration_num"] == fx["rust"]["iteration_num"])
                if not fx["iter_match"]:
                    print(f"MERGE MISMATCH: {series_dir.name}/{name}: "
                          f"cpp={fx['cpp']['iteration_num']} "
                          f"rust={fx['rust']['iteration_num']}", file=sys.stderr)
                    rc = 1
        sidecars = sorted(series_dir.glob("*.cell.json"))
        manifest = (json.loads(sidecars[0].read_text())["manifest"] if sidecars
                    else build_manifest(cfg, root, session_dir.name, series_dir.name))
        out = {
            "benchmark": "WCET fixture replay (campaign merge)",
            "meta": dict(meta or {}, manifest=manifest),
            "fixtures": merged,
        }
        out_path = session_dir / f"{series_dir.name}.json"
        out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {out_path} ({len(merged)} fixtures)")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(BENCH_DIR / "campaign_config.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-env")
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--verbose", action="store_true")
    p_run = sub.add_parser("run")
    p_run.add_argument("--session", type=int, required=True)
    p_run.add_argument("--series", default=None,
                       help="restrict to one series (warm|cold|corunner:MODE)")
    p_run.add_argument("--allow-env-mismatch", action="store_true")
    p_run.add_argument("--max-cells", type=int, default=None,
                       help="dev: truncate the plan (smoke tests)")
    p_merge = sub.add_parser("merge")
    p_merge.add_argument("session_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "verify-env":
        return cmd_verify_env(cfg, args)
    if args.cmd == "plan":
        return cmd_plan(cfg, args)
    if args.cmd == "run":
        return cmd_run(cfg, args)
    if args.cmd == "merge":
        return cmd_merge(cfg, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
