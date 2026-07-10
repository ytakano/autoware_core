#!/usr/bin/env bash
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

# L1a node-level end-to-end benchmark runner (see plan/ndt_bench.md).
# Builds the opt-in L1a bench gtest for BOTH engines (NDT_USE_RUST OFF then ON), runs each to record
# the node's per-frame exe_time_ms on the deterministic synthetic standard_sequence cloud, merges the
# two per-engine JSONs, and renders a self-contained HTML report via bench/gen_report.py.
#
# Run from anywhere inside the dev container:
#   bash src/core/autoware_core/localization/autoware_ndt_scan_matcher/bench/run_l1a.sh [ITERS] [WARMUP]
#
# Optional env:
#   TASKSET="taskset -c 2"   pin to an isolated core for a stable tail
#   OUT_DIR=/path            where the JSON + HTML land (default: the bench/ dir)
#   WS_ROOT=/path            colcon workspace root (default: derived from this script's path)

set -euo pipefail

ITERS="${1:-200}"
WARMUP="${2:-20}"

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$BENCH_DIR}"
CPP_JSON="$OUT_DIR/l1a_cpp.json"
RUST_JSON="$OUT_DIR/l1a_rust.json"
JSON="$OUT_DIR/l1a.json"
HTML="$OUT_DIR/l1a_report.html"
PKG=autoware_ndt_scan_matcher

# Workspace root: strip the canonical bench path from this script's location (see run.sh for rationale).
PKG_REL="src/core/autoware_core/localization/${PKG}/bench"
if [[ -z "${WS_ROOT:-}" ]]; then
  if [[ "$BENCH_DIR" == */"$PKG_REL" ]]; then
    WS_ROOT="${BENCH_DIR%/"$PKG_REL"}"
  else
    WS_ROOT="$PWD"
    echo "[l1a] WARNING: bench dir is not at the canonical '$PKG_REL' path; assuming WS_ROOT=$WS_ROOT" >&2
  fi
fi
echo "[l1a] workspace: $WS_ROOT"
if [[ ! -d "$WS_ROOT/src" ]]; then
  echo "[l1a] ERROR: $WS_ROOT has no src/ — not a colcon workspace root (set WS_ROOT=...)." >&2
  exit 1
fi
cd "$WS_ROOT"

# ROS setup scripts reference unset vars; disable nounset only while sourcing them.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

# Build + run one engine pass. $1 = engine key (cpp|rust), $2 = NDT_USE_RUST (OFF|ON), $3 = out JSON.
run_pass() {
  local engine="$1" use_rust="$2" out="$3"
  echo "[l1a] building $PKG (Release, NDT_USE_RUST=$use_rust, NDT_BUILD_BENCH_L1=ON) ..."
  colcon build --packages-select "$PKG" --cmake-force-configure \
    --cmake-args -DCMAKE_BUILD_TYPE=Release "-DNDT_USE_RUST=$use_rust" -DNDT_BUILD_BENCH_L1=ON
  # The bench uses ament_index to find the installed config yaml, so the workspace overlay must be
  # sourced (re-source after each build; the share path is stable, only contents change).
  set +u
  # shellcheck disable=SC1091
  source "$WS_ROOT/install/setup.bash"
  set -u
  local exe="$WS_ROOT/build/$PKG/l1a_node_frame_bench"
  if [[ ! -x "$exe" ]]; then
    exe="$(find "$WS_ROOT/build" -type f -name l1a_node_frame_bench -perm -u+x 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "$exe" || ! -x "$exe" ]]; then
    echo "[l1a] ERROR: l1a_node_frame_bench not found (did the build enable -DNDT_BUILD_BENCH_L1=ON?)" >&2
    exit 1
  fi
  echo "[l1a] running $engine pass -> $out"
  L1A_ENGINE="$engine" L1A_OUT="$out" L1A_ITERS="$ITERS" L1A_WARMUP="$WARMUP" \
    ${TASKSET:-} "$exe"
}

run_pass cpp OFF "$CPP_JSON"
run_pass rust ON "$RUST_JSON"

# Merge the two single-engine JSONs into one {engines:{cpp,rust}} doc (meta from the cpp pass). Warn if
# the OFF/ON iteration counts differ — the engines then did different work and the comparison is unfair.
echo "[l1a] merging per-engine JSONs ..."
python3 - "$CPP_JSON" "$RUST_JSON" "$JSON" <<'PYEOF'
import json, sys
cpp_path, rust_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cpp_path, encoding="utf-8") as fh:
    cpp = json.load(fh)
with open(rust_path, encoding="utf-8") as fh:
    rust = json.load(fh)
merged = {
    "benchmark": "L1a node end-to-end (synthetic standard_sequence cloud), C++ vs Rust",
    "meta": cpp.get("meta", {}),
    "engines": {"cpp": cpp["engines"]["cpp"], "rust": rust["engines"]["rust"]},
}
it_cpp = merged["engines"]["cpp"].get("iteration_num")
it_rust = merged["engines"]["rust"].get("iteration_num")
if it_cpp != it_rust:
    print(f"[l1a] WARNING: iteration_num differs (cpp={it_cpp}, rust={it_rust}) — "
          "engines did different work; latency comparison is not apples-to-apples.", file=sys.stderr)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(merged, fh, indent=2)
PYEOF

# Capture the run-time environment and merge it under "env" (same block as run.sh).
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^[[:space:]]*//')"
if [[ -z "$CPU_MODEL" ]]; then
  CPU_MODEL="$(lscpu 2>/dev/null | grep -m1 'Model name' | cut -d: -f2- | sed 's/^[[:space:]]*//')"
fi
CPU_CORES="$(nproc 2>/dev/null || echo '?')"
CPU_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'n/a')"
KERNEL="$(uname -sr 2>/dev/null || echo '?')"
CXX_VER="$(${CXX:-c++} --version 2>/dev/null | head -n1 || echo '?')"
RUSTC_VER="$(rustc --version 2>/dev/null | head -n1 || echo '?')"
STAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo '?')"

echo "[l1a] capturing environment ..."
BENCH_CPU="$CPU_MODEL" BENCH_CORES="$CPU_CORES" BENCH_GOV="$CPU_GOV" BENCH_KERNEL="$KERNEL" \
BENCH_CXX="$CXX_VER" BENCH_RUSTC="$RUSTC_VER" BENCH_STAMP="$STAMP" BENCH_TASKSET="${TASKSET:-none}" \
python3 - "$JSON" <<'PYEOF'
import json, os, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
data["env"] = {
    "CPU": os.environ.get("BENCH_CPU") or "unknown",
    "Logical cores": os.environ.get("BENCH_CORES", "?"),
    "CPU governor": os.environ.get("BENCH_GOV", "n/a"),
    "Kernel": os.environ.get("BENCH_KERNEL", "?"),
    "C++ compiler": os.environ.get("BENCH_CXX", "?"),
    "Rust compiler": os.environ.get("BENCH_RUSTC", "?"),
    "CPU pinning": os.environ.get("BENCH_TASKSET", "none"),
    "Captured (UTC)": os.environ.get("BENCH_STAMP", "?"),
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
PYEOF

echo "[l1a] rendering HTML ..."
python3 "$BENCH_DIR/gen_report.py" "$JSON" "$HTML"

echo "[l1a] done:"
echo "  JSON: $JSON"
echo "  HTML: $HTML"
