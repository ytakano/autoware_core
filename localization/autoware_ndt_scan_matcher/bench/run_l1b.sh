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

# L1b node-level end-to-end benchmark runner on the REAL Autoware sample-rosbag — SCAFFOLD / best-effort
# (see plan/ndt_bench.md). It: provisions the sample data on demand (bench/fetch_sample_data.sh),
# preflights the inputs, then for each engine (NDT_USE_RUST OFF then ON) rebuilds the node, launches
# ndt_l1b_bench.launch.xml, replays the bag, and records exe_time_ms via bench/record_exe_time.py;
# finally it merges + renders like run_l1a.sh.
#
#   bash .../bench/run_l1b.sh
#   env: BAG_LIDAR_TOPIC=/points_raw  (the bag's LiDAR topic; find via `ros2 bag info`)
#        OUT_DIR, TASKSET, WS_ROOT, AUTOWARE_DATA_DIR
#
# BEST-EFFORT: a converged headline additionally needs TF (map->base_link) + an initial-pose stream on
# ekf_pose_with_covariance (autoware_ekf_localizer + autoware_pose_initializer) — supply these from the
# bag or by extending ndt_l1b_bench.launch.xml. This runner produces samples only for frames the node
# actually scan-matches; if the pose/TF are missing it will record zero samples and say so.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$BENCH_DIR}"
PKG=autoware_ndt_scan_matcher
BAG_LIDAR_TOPIC="${BAG_LIDAR_TOPIC:-/points_raw}"

PKG_REL="src/core/autoware_core/localization/${PKG}/bench"
if [[ -z "${WS_ROOT:-}" ]]; then
  if [[ "$BENCH_DIR" == */"$PKG_REL" ]]; then
    WS_ROOT="${BENCH_DIR%/"$PKG_REL"}"
  else
    WS_ROOT="$PWD"
  fi
fi
[[ -d "$WS_ROOT/src" ]] || { echo "[l1b] ERROR: $WS_ROOT is not a colcon workspace (set WS_ROOT=...)." >&2; exit 1; }
cd "$WS_ROOT"

# --- 1. provision sample data (download-if-absent) -----------------------------------------------
echo "[l1b] ensuring sample data ..."
FETCH_OUT="$(bash "$BENCH_DIR/fetch_sample_data.sh")" || {
  echo "[l1b] ERROR: sample data unavailable (see message above)." >&2; exit 1; }
SAMPLE_ROSBAG_DIR="$(echo "$FETCH_OUT" | sed -n 's/^SAMPLE_ROSBAG_DIR=//p')"
SAMPLE_MAP_DIR="$(echo "$FETCH_OUT" | sed -n 's/^SAMPLE_MAP_DIR=//p')"
echo "[l1b]   bag: $SAMPLE_ROSBAG_DIR"
echo "[l1b]   map: $SAMPLE_MAP_DIR"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

# --- 2. preflight ---------------------------------------------------------------------------------
PCD="$(find "$SAMPLE_MAP_DIR" -name '*.pcd' | head -n1 || true)"
META="$(find "$SAMPLE_MAP_DIR" -name '*metadata*.yaml' | head -n1 || true)"
[[ -n "$PCD" ]]  || { echo "[l1b] ERROR: no .pcd under $SAMPLE_MAP_DIR" >&2; exit 1; }
[[ -n "$META" ]] || echo "[l1b] WARNING: no *metadata*.yaml under $SAMPLE_MAP_DIR — map loader may need it." >&2
if ! ros2 bag info "$SAMPLE_ROSBAG_DIR" 2>/dev/null | grep -q "$BAG_LIDAR_TOPIC"; then
  echo "[l1b] WARNING: bag does not advertise BAG_LIDAR_TOPIC='$BAG_LIDAR_TOPIC'." >&2
  echo "[l1b]          Topics in the bag:" >&2
  ros2 bag info "$SAMPLE_ROSBAG_DIR" 2>/dev/null | sed -n 's/^ *Topic: /            /p' >&2 || true
  echo "[l1b]          Re-run with BAG_LIDAR_TOPIC=<the LiDAR topic>." >&2
fi

run_pass() {  # <engine> <OFF|ON> <out.json>
  local engine="$1" use_rust="$2" out="$3"
  echo "[l1b] building $PKG (NDT_USE_RUST=$use_rust) ..."
  colcon build --packages-select "$PKG" --cmake-force-configure \
    --cmake-args -DCMAKE_BUILD_TYPE=Release "-DNDT_USE_RUST=$use_rust"
  set +u
  # shellcheck disable=SC1091
  source "$WS_ROOT/install/setup.bash"
  set -u

  echo "[l1b] launching node graph + recorder + bag replay ($engine) ..."
  python3 "$BENCH_DIR/record_exe_time.py" --engine "$engine" --out "$out" &
  local rec_pid=$!
  ros2 launch "$PKG" ndt_l1b_bench.launch.xml \
    pointcloud_map_path:="$PCD" pointcloud_map_metadata_path:="${META:-}" \
    bag_lidar_topic:="$BAG_LIDAR_TOPIC" &
  local launch_pid=$!
  sleep 8  # let the map load + node activate before replay
  ${TASKSET:-} ros2 bag play "$SAMPLE_ROSBAG_DIR" || true
  sleep 2
  kill -INT "$rec_pid" 2>/dev/null || true; wait "$rec_pid" 2>/dev/null || true
  kill -INT "$launch_pid" 2>/dev/null || true; wait "$launch_pid" 2>/dev/null || true
}

run_pass cpp OFF "$OUT_DIR/l1b_cpp.json"
run_pass rust ON "$OUT_DIR/l1b_rust.json"

# --- 3. merge + report (reuse run_l1a's merge shape) ---------------------------------------------
echo "[l1b] merging + rendering ..."
python3 - "$OUT_DIR/l1b_cpp.json" "$OUT_DIR/l1b_rust.json" "$OUT_DIR/l1b.json" <<'PYEOF'
import json, sys
cpp = json.load(open(sys.argv[1], encoding="utf-8"))
rust = json.load(open(sys.argv[2], encoding="utf-8"))
n_cpp = len(cpp["engines"]["cpp"]["samples_ms"])
n_rust = len(rust["engines"]["rust"]["samples_ms"])
if n_cpp == 0 or n_rust == 0:
    print(f"[l1b] WARNING: recorded {n_cpp} cpp / {n_rust} rust samples — the node likely never "
          "scan-matched (missing TF / initial pose / wrong bag topic). See the launch NOTE.",
          file=sys.stderr)
merged = {
    "benchmark": "L1b node end-to-end (Autoware sample-rosbag), C++ vs Rust",
    "meta": cpp.get("meta", {}),
    "engines": {"cpp": cpp["engines"]["cpp"], "rust": rust["engines"]["rust"]},
}
json.dump(merged, open(sys.argv[3], "w", encoding="utf-8"), indent=2)
PYEOF

python3 "$BENCH_DIR/gen_report.py" "$OUT_DIR/l1b.json" "$OUT_DIR/l1b_report.html"
echo "[l1b] done: $OUT_DIR/l1b.json + $OUT_DIR/l1b_report.html"
