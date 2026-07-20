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
    "pareto_01": 895, "pareto_02": 872,
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

def all_fixture_names(cfg):
    names = [name for spec in cfg["tiers"].values() for name in spec["fixtures"]]
    if len(names) != len(set(names)):
        raise ValueError("fixture names must be unique across tiers")
    return names


def fixture_path(cfg, name):
    override = cfg.get("fixture_paths", {}).get(name)
    if override:
        path = BENCH_DIR / override
    else:
        path = BENCH_DIR / cfg["fixtures_dir"] / f"{name}.ndtfix"
    return path.resolve()


def validate_config(cfg):
    names = all_fixture_names(cfg)
    if not names:
        raise ValueError("at least one fixture is required")
    missing = [str(fixture_path(cfg, name)) for name in names
               if not fixture_path(cfg, name).is_file()]
    if missing:
        raise ValueError("missing fixtures: " + ", ".join(missing))
    unknown_corunner = sorted(set(cfg["corunner"]["fixtures"]) - set(names))
    if unknown_corunner:
        raise ValueError(f"co-runner fixtures are not in the timing set: {unknown_corunner}")
    if cfg.get("sessions", 0) < 1:
        raise ValueError("sessions must be positive")
    for tier, spec in cfg["tiers"].items():
        if spec.get("samples", 0) < 1:
            raise ValueError(f"tier {tier} samples must be positive")
    for section in ("cold", "corunner"):
        if cfg[section].get("samples", 0) < 1:
            raise ValueError(f"{section} samples must be positive")
    enabled_series = cfg.get("series")
    valid_series = {"warm", "cold"} | {
        f"corunner:{mode}" for mode in cfg["corunner"]["modes"]
    }
    if enabled_series is not None:
        if not enabled_series or len(enabled_series) != len(set(enabled_series)):
            raise ValueError("series must be a non-empty list without duplicates")
        unknown_series = sorted(set(enabled_series) - valid_series)
        if unknown_series:
            raise ValueError(f"unknown enabled series: {unknown_series}")
    benchmark_env = cfg.get("benchmark_env", {})
    allowed_env = {"WCET_MAX_SOURCE_POINTS", "WCET_MAX_ACTIVE_LEAVES"}
    if not isinstance(benchmark_env, dict) or set(benchmark_env) - allowed_env:
        raise ValueError(f"benchmark_env may contain only {sorted(allowed_env)}")
    if any(not isinstance(value, str) or not value for value in benchmark_env.values()):
        raise ValueError("benchmark_env values must be non-empty strings")
    p_limit = benchmark_env.get("WCET_MAX_SOURCE_POINTS")
    if (p_limit is not None and p_limit != "source"
            and (not p_limit.isdigit() or int(p_limit) < 1)):
        raise ValueError("WCET_MAX_SOURCE_POINTS must be source or a positive integer")
    leaf_limit = benchmark_env.get("WCET_MAX_ACTIVE_LEAVES")
    if leaf_limit is not None and (not leaf_limit.isdigit() or int(leaf_limit) < 1):
        raise ValueError("WCET_MAX_ACTIVE_LEAVES must be a positive integer")
    minimum = cfg["abort"].get("min_mem_available_mib", 0)
    critical = cfg["abort"].get("critical_mem_available_mib", 0)
    if minimum and critical and critical >= minimum:
        raise ValueError("critical memory threshold must be lower than the pre-cell threshold")


def config_hash(cfg):
    encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def boot_id():
    return read_sys("/proc/sys/kernel/random/boot_id")


def read_memory_snapshot():
    values = {}
    text_value = read_sys("/proc/meminfo") or ""
    for line in text_value.splitlines():
        key, _, value = line.partition(":")
        fields = value.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) // 1024
    vmstat = {}
    text_value = read_sys("/proc/vmstat") or ""
    for line in text_value.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in ("pswpin", "pswpout") and fields[1].isdigit():
            vmstat[fields[0]] = int(fields[1])
    return {
        "mem_available_mib": values.get("MemAvailable"),
        "swap_free_mib": values.get("SwapFree"),
        "pswpin": vmstat.get("pswpin", 0),
        "pswpout": vmstat.get("pswpout", 0),
    }



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
    """Return (failures, warnings, snapshot). Failures abort a real run.

    Profile B (controlled): isolation/SMT-off/IRQ-moved/pin expected.
    Profile A (production-representative): performance governor only; isolation must be
    ABSENT (CFS + normal system interference are the object of measurement, not noise).
    """
    cpu = cfg["benchmark_cpu"]
    profile_a = cfg.get("profile") == "A"
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
    if profile_a:
        if iso_set:
            failures.append(
                f"CPU isolation present ({isolated}) -- Profile A requires the normal CFS "
                "environment (remove isolcpus/nohz_full/rcu_nocbs from GRUB and reboot)"
            )
    elif cpu not in iso_set:
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
            if profile_a:
                warnings.append(
                    f"SMT sibling cpu{sib} is offline under Profile A "
                    f"(production keeps SMT on; fix: echo 1 | sudo tee {cpu_sys(sib, 'online')})"
                )
        else:
            snap[f"smt_sibling_{sib}"] = "online"
            if not profile_a:
                warnings.append(
                    f"SMT sibling cpu{sib} of cpu{cpu} is online "
                    f"(fix: echo 0 | sudo tee {cpu_sys(sib, 'online')}; "
                    "or keep it idle and note it)"
                )

    n_irqs = irqs_allowed_on(cpu)
    snap["irqs_allowed_on_benchmark_cpu"] = n_irqs
    if not profile_a and n_irqs is not None and n_irqs > 8:
        warnings.append(
            f"{n_irqs} IRQs may fire on cpu{cpu} "
            "(fix: move IRQ affinities away via /proc/irq/*/smp_affinity_list)"
        )

    freq = read_freq_khz(cpu)
    fmax = read_max_freq_khz(cpu)
    snap["cur_freq_khz"] = freq
    snap["max_freq_khz"] = fmax
    # Frequency pin (min == max == abort.nominal_khz): keeps every session in one speed
    # regime and shuts out platform boost/EC policy moves; resets on reboot -- warn so the
    # post-reboot checklist is fully automated.
    fmin_s = read_sys(cpu_sys(cpu, "cpufreq/scaling_min_freq"))
    fmax_s = read_sys(cpu_sys(cpu, "cpufreq/scaling_max_freq"))
    snap["scaling_min_khz"] = int(fmin_s) if fmin_s and fmin_s.isdigit() else None
    snap["scaling_max_khz"] = int(fmax_s) if fmax_s and fmax_s.isdigit() else None
    nominal = cfg["abort"].get("nominal_khz")
    if (not profile_a and nominal
            and (snap["scaling_min_khz"] != nominal or snap["scaling_max_khz"] != nominal)):
        warnings.append(
            f"frequency not pinned at nominal {nominal} kHz "
            f"(min={snap['scaling_min_khz']}, max={snap['scaling_max_khz']}; "
            f"fix: echo {nominal} | sudo tee "
            f"{cpu_sys(cpu, 'cpufreq/scaling_min_freq')} "
            f"{cpu_sys(cpu, 'cpufreq/scaling_max_freq')})"
        )
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
                        "CMAKE_EXE_LINKER_FLAGS", "NDT_BUILD_TRACED", "NDT_BUILD_BENCH",
                        "NDT_USE_RUST"):
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
    exe = find_replay(root)
    m = {
        "experiment_id": f"{session_id}/{series}" + (f"/{cell}" if cell else ""),
        "campaign_id": cfg.get("campaign_id"),
        "campaign_config_hash": config_hash(cfg),
        "measurement_profile": cfg["profile"],
        "boot_id": boot_id(),
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
        "kernel_version": f"{platform.uname()[0]} {platform.uname()[2]}",
        "governor": (env_snap or {}).get("governor"),
        "isolated_cpus": (env_snap or {}).get("isolated_cpus"),
        "benchmark_cpu": cfg["benchmark_cpu"],
        "affinity_mask": ("none (CFS placement)" if cfg.get("profile") == "A"
                          else f"taskset -c {cfg['benchmark_cpu']}"),
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
            name: sha256_of(fixture_path(cfg, name)) for name in all_fixture_names(cfg)},
        "binary_hash": sha256_of(exe) if exe else None,
        "cache_condition": "cold" if series == "cold" else "warm",
        "co_runner": series.split(":", 1)[1] if series.startswith("corunner:") else None,
        "env_check": env_mode,
        "dataset_id": None,
        "target_board": None,
        "firmware": None,
        "qemu_version": None,
        "benchmark_env": cfg.get("benchmark_env", {}),
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
    """Return randomized single-engine cells for one session."""
    rng = random.Random(0xC0FFEE ^ session)
    all_fixtures = all_fixture_names(cfg)
    default_series = ["warm", "cold"] + [
        f"corunner:{mode}" for mode in cfg["corunner"]["modes"]
    ]
    series_list = cfg.get("series", default_series)
    cells = []
    for series in series_list:
        fixtures = (list(all_fixtures) if series in ("warm", "cold")
                    else list(cfg["corunner"]["fixtures"]))
        rng.shuffle(fixtures)
        for fixture in fixtures:
            _, tier_samples = fixture_tier(cfg, fixture)
            if series == "warm":
                samples, warmup, evict = tier_samples, cfg["warmup"], 0
            elif series == "cold":
                samples = cfg["cold"]["samples"]
                warmup = cfg["cold"]["warmup"]
                evict = cfg["cold"]["evict_mib"] * 1024 * 1024
            else:
                samples, warmup, evict = cfg["corunner"]["samples"], cfg["warmup"], 0
            engines = ["cpp", "rust"]
            rng.shuffle(engines)
            for engine in engines:
                cells.append({
                    "series": series, "fixture": fixture, "engine": engine,
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


def campaign_root(cfg):
    return BENCH_DIR / cfg["output_dir"]


def require_untraced_timing_build(root):
    flags = cmake_flags(root)
    if flags.get("NDT_BUILD_TRACED") != "OFF":
        value = flags.get("NDT_BUILD_TRACED", "missing")
        raise ValueError(
            f"timing campaign requires NDT_BUILD_TRACED=OFF (CMake cache: {value}); "
            "use ndt_bench_replay_traced only for work-trace conformance")


def campaign_identity(cfg, root):
    require_untraced_timing_build(root)
    exe = find_replay(root)
    if exe is None:
        raise ValueError(
            f"ndt_bench_replay not found under {root}/build/{PKG}; "
            "build with -DNDT_BUILD_BENCH=ON")
    return {
        "campaign_id": cfg.get("campaign_id"),
        "config_hash": config_hash(cfg),
        "binary_hash": sha256_of(exe),
        "cpp_commit": git_commit(root / "src/core/autoware_core"),
        "rust_commit": git_commit(root / "src/core/autoware_core"),
        "fixture_hashes": {
            name: sha256_of(fixture_path(cfg, name)) for name in all_fixture_names(cfg)},
        "sessions": cfg["sessions"],
        "cells_per_session": len(build_plan(cfg, 1)),
    }


def ensure_campaign_lock(cfg, root):
    out = campaign_root(cfg)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "campaign.lock.json"
    expected = campaign_identity(cfg, root)
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if actual.get(key) != value]
        if mismatches:
            raise ValueError(
                "campaign lock mismatch for " + ", ".join(mismatches) +
                "; do not rebuild or change fixtures between sessions")
        return actual
    locked = dict(expected, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.write_text(json.dumps(locked, indent=1) + "\n", encoding="utf-8")
    return locked


def ensure_session_lock(cfg, session):
    directory = campaign_root(cfg) / f"session-{session}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "session.lock.json"
    current = boot_id()
    if not current:
        raise ValueError("kernel boot_id is unavailable")
    expected = {
        "campaign_id": cfg.get("campaign_id"), "config_hash": config_hash(cfg),
        "session": session, "boot_id": current,
    }
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if actual.get(key) != value]
        if mismatches:
            raise ValueError(
                f"session-{session} lock mismatch for {', '.join(mismatches)}; "
                "a session may not span boots")
        return directory
    expected["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(expected, indent=1) + "\n", encoding="utf-8")
    return directory


def series_dir_name(series):
    return series.replace(":", "_")


def cell_paths(session_dir, cell):
    directory = session_dir / series_dir_name(cell["series"])
    stem = f"{cell['fixture']}__{cell['engine']}"
    return directory / f"{stem}.json", directory / f"{stem}.cell.json"


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def measurement_problem(cfg, path, cell):
    doc = load_json(path)
    if doc is None:
        return "measurement JSON is missing or malformed"
    meta = doc.get("meta") or {}
    if meta.get("iters") != cell["samples"]:
        return f"sample metadata is {meta.get('iters')}, expected {cell['samples']}"
    if meta.get("engine") != cell["engine"]:
        return f"engine metadata is {meta.get('engine')!r}, expected {cell['engine']!r}"
    fixtures = doc.get("fixtures") or {}
    if set(fixtures) != {cell["fixture"]}:
        return f"fixture set is {sorted(fixtures)}, expected {[cell['fixture']]}"
    fixture = fixtures[cell["fixture"]]
    samples = (fixture.get(cell["engine"]) or {}).get("samples_ms")
    if not isinstance(samples, list) or len(samples) != cell["samples"]:
        return f"sample array length is {len(samples) if isinstance(samples, list) else None}"
    benchmark_env = cfg.get("benchmark_env", {})
    if benchmark_env:
        limits = fixture.get("rust_limits") or {}
        p_setting = benchmark_env.get("WCET_MAX_SOURCE_POINTS", "2000")
        expected_p = fixture.get("n_source") if p_setting == "source" else int(p_setting)
        expected_l = int(benchmark_env.get("WCET_MAX_ACTIVE_LEAVES", "418000"))
        expected_i = fixture.get("max_iterations")
        expected = {
            "max_source_points": expected_p,
            "max_active_leaves": expected_l,
            "max_iterations": expected_i,
        }
        if limits != expected:
            return f"Rust limits are {limits}, expected {expected}"
    return None


def completed_cell_problem(cfg, out_json, sidecar_json, cell):
    problem = measurement_problem(cfg, out_json, cell)
    if problem:
        return problem
    sidecar = load_json(sidecar_json)
    if sidecar is None:
        return "sidecar is missing or malformed"
    if sidecar.get("cell") != cell:
        return "sidecar cell specification differs from the current plan"
    if sidecar.get("problems"):
        return "sidecar records measurement problems"
    manifest = sidecar.get("manifest") or {}
    if manifest.get("campaign_config_hash") != config_hash(cfg):
        return "sidecar config hash differs from the current campaign"
    expected_hash = sha256_of(fixture_path(cfg, cell["fixture"]))
    if manifest.get("fixture_hashes", {}).get(cell["fixture"]) != expected_hash:
        return "sidecar fixture hash differs from the current input"
    return None


def quarantine_cell(out_json, sidecar_json):
    existing = [path for path in (out_json, sidecar_json) if path.exists()]
    if not existing:
        return
    rejected = out_json.parent.parent / "rejected" / out_json.parent.name
    rejected.mkdir(parents=True, exist_ok=True)
    suffix = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    for path in existing:
        path.rename(rejected / f"{path.name}.{suffix}")


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def cmd_prepare(cfg, _args):
    try:
        locked = ensure_campaign_lock(cfg, ws_root())
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(locked, indent=1))
    return 0


def cmd_plan(cfg, args):
    total = 0.0
    for session in range(1, cfg["sessions"] + 1):
        cells = build_plan(cfg, session)
        duration = sum(estimate_ms(cell) for cell in cells) / 1000.0
        total += duration
        print(f"session {session}: {len(cells)} cells, ~{duration / 60:.0f} min")
        if session == 1 and args.verbose:
            for cell in cells:
                print(
                    f"  {cell['series']:14} {cell['fixture']:16} {cell['engine']:4} "
                    f"n={cell['samples']} warmup={cell['warmup']} "
                    f"evict={cell['evict_bytes']}")
    print(
        f"total (all {cfg['sessions']} sessions): ~{total / 3600:.1f} h "
        "(priors-based estimate; excludes map builds and environment checks)")
    return 0


def cmd_verify_env(cfg, _args):
    failures, warnings, snap = verify_environment(cfg)
    memory = read_memory_snapshot()
    minimum = cfg["abort"].get("min_mem_available_mib", 0)
    if memory["mem_available_mib"] is not None and memory["mem_available_mib"] < minimum:
        failures.append(
            f"MemAvailable {memory['mem_available_mib']} MiB < required {minimum} MiB")
    snap["memory"] = memory
    for warning in warnings:
        print(f"WARN: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}")
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


def cmd_status(cfg, args):
    session_dir = campaign_root(cfg) / f"session-{args.session}"
    done = {}
    invalid = []
    for cell in build_plan(cfg, args.session):
        out_json, sidecar = cell_paths(session_dir, cell)
        problem = completed_cell_problem(cfg, out_json, sidecar, cell)
        done.setdefault(cell["series"], [0, 0])
        done[cell["series"]][1] += 1
        if problem is None:
            done[cell["series"]][0] += 1
        elif out_json.exists() or sidecar.exists():
            invalid.append(f"{cell['series']}/{cell['fixture']}/{cell['engine']}: {problem}")
    for series in sorted(done):
        complete, total = done[series]
        print(f"{series}: {complete}/{total} complete")
    for item in invalid:
        print(f"INVALID: {item}")
    complete = all(finished == total for finished, total in done.values())
    return 0 if complete and not invalid else 1


def cmd_run(cfg, args):
    root = ws_root()
    try:
        locked = ensure_campaign_lock(cfg, root)
        session_dir = ensure_session_lock(cfg, args.session)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    exe = find_replay(root)
    failures, warnings, snap0 = verify_environment(cfg)
    env_mode = "strict"
    if failures:
        if not args.allow_env_mismatch:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        env_mode = "overridden"
        for failure in failures:
            print(f"WARN (overridden): {failure}", file=sys.stderr)
    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)

    cells = build_plan(cfg, args.session)
    if args.series:
        cells = [cell for cell in cells if cell["series"] == args.series]
        if not cells:
            print(f"FAIL: unknown or empty series {args.series!r}", file=sys.stderr)
            return 1
    if args.max_cells:
        cells = cells[:args.max_cells]
    calib_exe = corunner_exe(session_dir)
    calib_base_ns = run_calibration(cfg, calib_exe)
    print(f"session-{args.session}: {len(cells)} planned cells")
    rc = 0
    for index, cell in enumerate(cells, 1):
        out_json, sidecar_json = cell_paths(session_dir, cell)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        existing_problem = completed_cell_problem(cfg, out_json, sidecar_json, cell)
        if existing_problem is None:
            if args.resume:
                print(f"[{index}/{len(cells)}] SKIP {cell['series']} {cell['fixture']} {cell['engine']}")
                continue
            print(f"FAIL: completed output exists at {out_json}; use --resume", file=sys.stderr)
            return 1
        if out_json.exists() or sidecar_json.exists():
            if not args.resume:
                print(
                    f"FAIL: incomplete output at {out_json}: {existing_problem}; use --resume",
                    file=sys.stderr)
                return 1
            quarantine_cell(out_json, sidecar_json)

        memory_before = read_memory_snapshot()
        minimum = cfg["abort"].get("min_mem_available_mib", 0)
        available = memory_before["mem_available_mib"]
        if available is not None and available < minimum:
            print(f"ABORT: MemAvailable {available} MiB < {minimum} MiB", file=sys.stderr)
            return 1
        _, _, before = verify_environment(cfg)
        co = child = None
        memory_abort = None
        during = []
        memory_samples = [memory_before]
        env = dict(os.environ)
        env.update({
            "WCET_ITERS": str(cell["samples"]), "WCET_WARMUP": str(cell["warmup"]),
            "WCET_ENGINE": cell["engine"], "WCET_EVICT_BYTES": str(cell["evict_bytes"]),
        })
        env.update(cfg.get("benchmark_env", {}))
        pin = ([] if cfg.get("profile") == "A"
               else ["taskset", "-c", str(cfg["benchmark_cpu"])])
        argv = pin + [str(exe), "--fixture", str(out_json), str(fixture_path(cfg, cell["fixture"]))]
        print(f"[{index}/{len(cells)}] {cell['series']} {cell['fixture']} {cell['engine']} n={cell['samples']}")
        try:
            if cell["corunner"]:
                co = start_corunner(cfg, cell["corunner"], session_dir)
                time.sleep(1.0)
            child = subprocess.Popen(argv, env=env)
            while child.poll() is None:
                during.append((read_freq_khz(cfg["benchmark_cpu"]), read_temps_c()))
                memory_now = read_memory_snapshot()
                memory_samples.append(memory_now)
                critical = cfg["abort"].get("critical_mem_available_mib", 0)
                current = memory_now["mem_available_mib"]
                if critical and current is not None and current < critical:
                    memory_abort = f"MemAvailable {current} MiB < critical {critical} MiB"
                    stop_process(child)
                    break
                time.sleep(1.0)
        finally:
            stop_process(child)
            stop_process(co)
        proc_rc = child.returncode if child is not None else 1
        memory_after = read_memory_snapshot()
        memory_samples.append(memory_after)
        failures_after, _, after = verify_environment(cfg)
        problems = series_guard(cfg, during, f"{cell['series']}/{cell['fixture']}")
        if env_mode == "strict":
            problems.extend(f"post-cell environment: {item}" for item in failures_after)
        if memory_abort:
            problems.append(memory_abort)
        if cfg["abort"].get("abort_on_swap_io", False):
            if (memory_after["pswpin"] != memory_before["pswpin"]
                    or memory_after["pswpout"] != memory_before["pswpout"]):
                problems.append("swap I/O occurred during the cell")
        if proc_rc != 0:
            problems.append(f"replay exited {proc_rc}")
        output_problem = measurement_problem(cfg, out_json, cell)
        if output_problem:
            problems.append(output_problem)
        nominal, nominal_source = nominal_freq_khz(cfg)
        calib_ns = run_calibration(cfg, calib_exe, n=1)
        if calib_base_ns and calib_ns:
            slowdown = calib_ns / calib_base_ns
            if slowdown > cfg["abort"].get("max_calib_slowdown", 1.1):
                problems.append(f"calibration spin x{slowdown:.2f} slower than session baseline")
        demoted = []
        if cfg.get("profile") == "A" and problems:
            retained = []
            for problem in problems:
                (retained if "temperature" in problem else demoted).append(problem)
            problems = retained
        frequencies = [freq for freq, _ in during if freq]
        available_samples = [
            sample["mem_available_mib"] for sample in memory_samples
            if sample["mem_available_mib"] is not None]
        sidecar = {
            "cell": cell, "env_before": before, "env_after": after,
            "freq_khz_during": {
                "n": len(frequencies),
                "median": sorted(frequencies)[len(frequencies) // 2] if frequencies else None,
                "min": min(frequencies) if frequencies else None,
                "max": max(frequencies) if frequencies else None,
                "nominal": nominal, "nominal_source": nominal_source,
                "stale": bool(frequencies) and len(set(frequencies)) <= 1,
            },
            "memory": {
                "before": memory_before, "after": memory_after,
                "min_available_mib": min(available_samples) if available_samples else None,
            },
            "calibration": {
                "baseline_ns": calib_base_ns, "after_cell_ns": calib_ns,
                "slowdown": calib_ns / calib_base_ns if calib_base_ns and calib_ns else None,
            },
            "problems": problems, "profile_a_recorded_excursions": demoted,
            "campaign_lock": locked,
            "manifest": build_manifest(
                cfg, root, f"session-{args.session}", cell["series"],
                cell=f"{cell['fixture']}__{cell['engine']}", env_snap=snap0,
                env_mode=env_mode),
        }
        sidecar_json.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
        if problems:
            for problem in problems:
                print(f"{'WARN' if env_mode == 'overridden' else 'ABORT'}: {problem}", file=sys.stderr)
            if env_mode != "overridden":
                return 1
            rc = max(rc, 2)
    print(f"done -> {session_dir}")
    return rc


def resolve_session_dir(cfg, args):
    if getattr(args, "session", None) is not None:
        return campaign_root(cfg) / f"session-{args.session}"
    if getattr(args, "session_dir", None):
        return pathlib.Path(args.session_dir)
    raise ValueError("either --session or SESSION_DIR is required")


def cmd_merge(cfg, args):
    try:
        session_dir = resolve_session_dir(cfg, args)
        session_number = int(session_dir.name.split("-", 1)[1])
    except (ValueError, IndexError) as exc:
        print(f"FAIL: invalid session directory: {exc}", file=sys.stderr)
        return 1
    cells = build_plan(cfg, session_number)
    invalid = []
    for cell in cells:
        out_json, sidecar = cell_paths(session_dir, cell)
        problem = completed_cell_problem(cfg, out_json, sidecar, cell)
        if problem:
            invalid.append(f"{cell['series']}/{cell['fixture']}/{cell['engine']}: {problem}")
    if invalid:
        for item in invalid:
            print(f"INCOMPLETE: {item}", file=sys.stderr)
        return 1
    rc = 0
    for series in sorted({cell["series"] for cell in cells}):
        selected = [cell for cell in cells if cell["series"] == series]
        merged = {}
        meta = None
        manifests = []
        for cell in selected:
            cell_json, sidecar_json = cell_paths(session_dir, cell)
            data, sidecar = load_json(cell_json), load_json(sidecar_json)
            meta = meta or data.get("meta")
            manifests.append(sidecar["manifest"])
            for name, fixture in data["fixtures"].items():
                slot = merged.setdefault(
                    name, {key: value for key, value in fixture.items()
                           if key not in ("cpp", "rust")})
                slot[cell["engine"]] = fixture[cell["engine"]]
        for name, fixture in merged.items():
            fixture["iter_match"] = (
                fixture["cpp"]["iteration_num"] == fixture["rust"]["iteration_num"])
            if not fixture["iter_match"]:
                print(f"MERGE MISMATCH: {series}/{name}", file=sys.stderr)
                rc = 1
        representative = dict(manifests[0], cell_manifests=manifests)
        output = {
            "benchmark": "WCET fixture replay (campaign merge)",
            "meta": dict(meta or {}, manifest=representative), "fixtures": merged,
        }
        output_path = session_dir / f"{series_dir_name(series)}.json"
        output_path.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {output_path} ({len(merged)} fixtures)")
    return rc


def cmd_smoke(cfg, _args):
    smoke_cfg = json.loads(json.dumps(cfg))
    token = uuid.uuid4().hex[:8]
    smoke_cfg["campaign_id"] = f"{cfg.get('campaign_id', 'campaign')}-smoke-{token}"
    smoke_cfg["output_dir"] = f"campaign_runs/smoke-{token}"
    smoke_cfg["sessions"] = 1
    smoke_cfg["warmup"] = 0
    for spec in smoke_cfg["tiers"].values():
        spec["samples"] = 1
    smoke_cfg["cold"]["samples"] = 1
    smoke_cfg["cold"]["warmup"] = 0
    smoke_cfg["corunner"]["samples"] = 1
    args = argparse.Namespace(
        session=1, series=None, allow_env_mismatch=True, max_cells=None, resume=False)
    rc = cmd_run(smoke_cfg, args)
    if rc not in (0, 2):
        return rc
    merge_args = argparse.Namespace(
        session=1, session_dir=str(campaign_root(smoke_cfg) / "session-1"))
    return cmd_merge(smoke_cfg, merge_args)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(BENCH_DIR / "campaign_config.json"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("verify-env")
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--verbose", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--session", type=int, required=True)
    run_parser.add_argument("--series", default=None,
                            help="warm|cold|corunner:membw|corunner:llc|corunner:fp")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--allow-env-mismatch", action="store_true")
    run_parser.add_argument("--max-cells", type=int, default=None,
                            help="development-only plan truncation")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--session", type=int, required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("session_dir", nargs="?")
    merge_parser.add_argument("--session", type=int)
    sub.add_parser("smoke")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        validate_config(cfg)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: invalid campaign config: {exc}", file=sys.stderr)
        return 1
    if args.cmd == "prepare":
        return cmd_prepare(cfg, args)
    if args.cmd == "verify-env":
        return cmd_verify_env(cfg, args)
    if args.cmd == "plan":
        return cmd_plan(cfg, args)
    if args.cmd == "run":
        if args.session < 1 or args.session > cfg["sessions"]:
            print(f"FAIL: session must be in 1..{cfg['sessions']}", file=sys.stderr)
            return 1
        return cmd_run(cfg, args)
    if args.cmd == "status":
        return cmd_status(cfg, args)
    if args.cmd == "merge":
        return cmd_merge(cfg, args)
    if args.cmd == "smoke":
        return cmd_smoke(cfg, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
