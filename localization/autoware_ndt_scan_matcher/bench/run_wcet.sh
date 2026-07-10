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

# WCET C++-vs-Rust comparison runner (plan/ndt_wcet.md, M4). Pipeline:
#   1. build the bench executable (both engines, Release, NDT_USE_RUST=ON, NDT_BUILD_BENCH=ON)
#   2. build the LD_PRELOAD allocation interposer (bench/alloc_count.c)
#   3. Rust counters + engine-level samples: engine example `wcet_frame` (WCET_JSON)
#   4. timing pass: ndt_bench_replay --fixture over bench/fixtures/*.ndtfix (both engines,
#      identical buffers, iteration_num equality asserted per fixture)
#   5. allocation pass: same replay under LD_PRELOAD=alloc_count.so, few iters -> allocs/align
#   6. render: bench/wcet_report.py -> markdown report
#
# Run from the workspace root (/autoware_workspace), inside the dev container:
#   bash src/core/autoware_core/localization/autoware_ndt_scan_matcher/bench/run_wcet.sh
#
# Optional env:
#   TASKSET="taskset -c 2"   pin to an isolated core for a stable tail
#   OUT_DIR=/path            where JSONs + report land (default: the bench/ dir)
#   WCET_ITERS / WCET_WARMUP timing pass (default 100 / 10)
#   RUST_FRAMES              Rust engine-level frames per fixture (default 200)
#   FIXTURES="a.ndtfix ..."  override the fixture set (default: bench/fixtures/*.ndtfix)

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$BENCH_DIR}"
PKG=autoware_ndt_scan_matcher

PKG_REL="src/core/autoware_core/localization/${PKG}/bench"
if [[ -z "${WS_ROOT:-}" ]]; then
  if [[ "$BENCH_DIR" == */"$PKG_REL" ]]; then
    WS_ROOT="${BENCH_DIR%/"$PKG_REL"}"
  else
    WS_ROOT="$PWD"
    echo "[wcet] WARNING: bench dir not at canonical '$PKG_REL'; assuming WS_ROOT=$WS_ROOT" >&2
  fi
fi
echo "[wcet] workspace: $WS_ROOT"
cd "$WS_ROOT"

if [[ -z "${FIXTURES:-}" ]]; then
  FIXTURES="$(ls "$BENCH_DIR"/fixtures/*.ndtfix)"
fi
echo "[wcet] fixtures:"
for fx in $FIXTURES; do echo "  $fx"; done

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

echo "[wcet] building $PKG (Release, NDT_USE_RUST=ON, NDT_BUILD_BENCH=ON) ..."
colcon build --packages-select "$PKG" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DNDT_USE_RUST=ON -DNDT_BUILD_BENCH=ON

EXE="$WS_ROOT/build/$PKG/ndt_bench_replay"
if [[ ! -x "$EXE" ]]; then
  EXE="$(find "$WS_ROOT/build" -type f -name ndt_bench_replay -perm -u+x 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$EXE" || ! -x "$EXE" ]]; then
  echo "[wcet] ERROR: ndt_bench_replay not found (build with -DNDT_BUILD_BENCH=ON)" >&2
  exit 1
fi

echo "[wcet] building allocation interposer ..."
ALLOC_SO="$OUT_DIR/alloc_count.so"
cc -O2 -shared -fPIC -o "$ALLOC_SO" "$BENCH_DIR/alloc_count.c" -ldl

ENGINE_DIR="$WS_ROOT/src/core/autoware_core/localization/$PKG/autoware_ndt_scan_matcher_rs/engine"
RUST_JSON="$OUT_DIR/wcet_rust.json"
echo "[wcet] Rust counters + engine-level samples (wcet_frame, wcet-count) ..."
(
  cd "$ENGINE_DIR"
  # shellcheck disable=SC2086
  WCET_FRAMES="${RUST_FRAMES:-200}" WCET_JSON="$RUST_JSON" \
    ${TASKSET:-} cargo run --release -q --features wcet-count --example wcet_frame -- $FIXTURES
)

JSON="$OUT_DIR/wcet.json"
echo "[wcet] timing pass (both engines) ..."
# shellcheck disable=SC2086
${TASKSET:-} "$EXE" --fixture "$JSON" $FIXTURES

ALLOC_JSON="$OUT_DIR/wcet_alloc.json"
echo "[wcet] allocation pass (LD_PRELOAD interposer, short) ..."
# shellcheck disable=SC2086
WCET_ITERS=20 WCET_WARMUP=5 LD_PRELOAD="$ALLOC_SO" "$EXE" --fixture "$ALLOC_JSON" $FIXTURES

# Merge the run-time environment into the timing JSON (same convention as run.sh).
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^[[:space:]]*//')"
CPU_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'n/a')"
BENCH_CPU="$CPU_MODEL" BENCH_GOV="$CPU_GOV" BENCH_KERNEL="$(uname -sr)" \
BENCH_CXX="$(${CXX:-c++} --version | head -n1)" BENCH_RUSTC="$(rustc --version)" \
BENCH_STAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')" BENCH_TASKSET="${TASKSET:-none}" \
python3 - "$JSON" <<'PYEOF'
import json, os, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
data["env"] = {
    "CPU": os.environ.get("BENCH_CPU") or "unknown",
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

REPORT="$OUT_DIR/wcet_report.md"
echo "[wcet] rendering report ..."
python3 "$BENCH_DIR/wcet_report.py" "$JSON" "$RUST_JSON" --alloc "$ALLOC_JSON" -o "$REPORT"

echo "[wcet] done:"
echo "  timing:   $JSON"
echo "  counters: $RUST_JSON"
echo "  allocs:   $ALLOC_JSON"
echo "  report:   $REPORT"
