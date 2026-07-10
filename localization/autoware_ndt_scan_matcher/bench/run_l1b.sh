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

# L1b node-level end-to-end benchmark on the REAL Autoware sample-rosbag (see plan/ndt_bench.md).
# For each engine (NDT_USE_RUST OFF then ON) it rebuilds the node, brings up the full NDT localization
# graph (ndt_l1b_bench.launch.xml: vehicle TF + Nebula sensing + map + downsample->NDT->EKF->
# pose_initializer), replays the bag with /clock, initializes localization, and records the node's own
# per-frame exe_time_ms (bench/record_exe_time.py) into the bench/gen_report.py schema; then merges and
# renders the OFF-vs-ON report.
#
#   bash .../bench/run_l1b.sh
#   env: AUTOWARE_DATA_DIR, OUT_DIR, WS_ROOT, INITIAL_POSE="x,y,z,qx,qy,qz,qw" (else derived from GNSS)
#
# PREREQUISITES (this bag is raw sensor data -> needs the full sensing stack, which is NOT in the
# default src/core-only build): build the sensing packages once, e.g.
#   colcon build --symlink-install --packages-up-to sample_sensor_kit_launch \
#     autoware_pointcloud_preprocessor sample_vehicle_description tier4_vehicle_launch \
#     --cmake-args -DCMAKE_BUILD_TYPE=Release
# and have rosbag2 + the ros2 CLI available (ros-humble-ros2bag / ros2cli / ros2launch).
#
# NOTE: getting a *converged* run is environment-sensitive — it needs a clean DDS state (this script
# forces UDP-only FastDDS and clears stale /dev/shm segments), a monotonic bag clock (single pass, not
# --loop), and the localization init handshake to complete. On a fresh full-Autoware container this
# reproduces the headline; in a churned/build-focused container the init handshake may not complete
# (see plan/ndt_bench.md "L1b outcome").

set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$BENCH_DIR}"
PKG=autoware_ndt_scan_matcher
PKG_REL="src/core/autoware_core/localization/${PKG}/bench"
WS_ROOT="${WS_ROOT:-${BENCH_DIR%/"$PKG_REL"}}"
[[ -d "$WS_ROOT/src" ]] || { echo "[l1b] ERROR: $WS_ROOT is not a colcon workspace (set WS_ROOT=)." >&2; exit 1; }
cd "$WS_ROOT"

# Force UDP-only FastDDS (shared-memory transport is fragile after repeated launch/kill churn).
DDS_PROFILE="$OUT_DIR/l1b_udp_only.xml"
cat > "$DDS_PROFILE" <<'XML'
<?xml version="1.0" encoding="UTF-8"?>
<dds xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"><profiles>
  <transport_descriptors><transport_descriptor>
    <transport_id>udp</transport_id><type>UDPv4</type>
  </transport_descriptor></transport_descriptors>
  <participant profile_name="udp_only" is_default_profile="true"><rtps>
    <userTransports><transport_id>udp</transport_id></userTransports>
    <useBuiltinTransports>false</useBuiltinTransports>
  </rtps></participant>
</profiles></dds>
XML
export FASTRTPS_DEFAULT_PROFILES_FILE="$DDS_PROFILE"

# 1. provision sample data (download-if-absent).
echo "[l1b] ensuring sample data ..."
FETCH_OUT="$(bash "$BENCH_DIR/fetch_sample_data.sh")" || { echo "[l1b] ERROR: sample data unavailable." >&2; exit 1; }
BAG_DIR="$(echo "$FETCH_OUT" | sed -n 's/^SAMPLE_ROSBAG_DIR=//p')"
MAP_DIR="$(echo "$FETCH_OUT" | sed -n 's/^SAMPLE_MAP_DIR=//p')"

set +u; source /opt/ros/humble/setup.bash; [[ -f "$WS_ROOT/install/setup.bash" ]] && source "$WS_ROOT/install/setup.bash"; set -u

command -v ros2 >/dev/null && ros2 bag info "$BAG_DIR" >/dev/null 2>&1 || {
  echo "[l1b] ERROR: 'ros2 bag' unavailable — install ros-humble-ros2bag / ros2cli / ros2launch." >&2; exit 1; }
ros2 pkg prefix sample_sensor_kit_launch >/dev/null 2>&1 || {
  echo "[l1b] ERROR: sensing stack not built — see PREREQUISITES in this script's header." >&2; exit 1; }

cleanup_graph() {
  for pid in $(ps -eo pid,args 2>/dev/null | grep -iE "ros2 launch $PKG|component_container|robot_state_publisher|rosbag2_player|autoware_.*_node|ndt_scan_matcher|ekf_localizer|pose_initializer|velodyne_ros|topic_tools" | grep -v grep | awk '{print $1}'); do kill -9 "$pid" 2>/dev/null; done
  sleep 3; rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
}

run_pass() {  # <engine> <OFF|ON> <out.json>
  local engine="$1" use_rust="$2" out="$3"
  echo "[l1b] building $PKG (NDT_USE_RUST=$use_rust) ..."
  colcon build --packages-select "$PKG" --cmake-force-configure \
    --cmake-args -DCMAKE_BUILD_TYPE=Release "-DNDT_USE_RUST=$use_rust" || return 1
  set +u; source "$WS_ROOT/install/setup.bash"; set -u

  cleanup_graph
  echo "[l1b] launching graph ($engine) ..."
  setsid ros2 launch "$PKG" ndt_l1b_bench.launch.xml \
    map_path:="$MAP_DIR" use_sim_time:=true "${INITIAL_POSE:+initial_pose:=[$INITIAL_POSE]}" \
    > "$OUT_DIR/l1b_${engine}_launch.log" 2>&1 &
  # wait for the NDT node to appear
  local n=0; until ros2 node list 2>/dev/null | grep -q pose_estimator/ndt_scan_matcher || [ $n -ge 25 ]; do sleep 3; n=$((n+1)); done

  setsid python3 "$BENCH_DIR/record_exe_time.py" --engine "$engine" --out "$out" \
    > "$OUT_DIR/l1b_${engine}_record.log" 2>&1 &
  local rec_pid=$!
  sleep 3
  # kick localization init (GNSS AUTO); harmless if initial_pose already auto-initialized
  timeout 20 ros2 service call /localization/initialize \
    autoware_internal_localization_msgs/srv/InitializeLocalization "{method: 0, pose_with_covariance: []}" \
    >/dev/null 2>&1 || true
  echo "[l1b] replaying bag ($engine) ..."
  ros2 bag play "$BAG_DIR" --clock 100 || true
  sleep 2
  kill -INT "$rec_pid" 2>/dev/null; wait "$rec_pid" 2>/dev/null || true
  cleanup_graph
}

run_pass cpp OFF "$OUT_DIR/l1b_cpp.json"
run_pass rust ON "$OUT_DIR/l1b_rust.json"

echo "[l1b] merging + rendering ..."
python3 - "$OUT_DIR/l1b_cpp.json" "$OUT_DIR/l1b_rust.json" "$OUT_DIR/l1b.json" <<'PYEOF'
import json, sys
try:
    cpp = json.load(open(sys.argv[1], encoding="utf-8")); rust = json.load(open(sys.argv[2], encoding="utf-8"))
except FileNotFoundError as e:
    print(f"[l1b] no recording ({e}); the init handshake likely did not complete.", file=sys.stderr); sys.exit(0)
nc, nr = len(cpp["engines"]["cpp"]["samples_ms"]), len(rust["engines"]["rust"]["samples_ms"])
if nc == 0 or nr == 0:
    print(f"[l1b] WARNING: {nc} cpp / {nr} rust samples — NDT did not scan-match (init/TF/map).", file=sys.stderr)
json.dump({"benchmark": "L1b node end-to-end (Autoware sample-rosbag), C++ vs Rust",
           "meta": cpp.get("meta", {}),
           "engines": {"cpp": cpp["engines"]["cpp"], "rust": rust["engines"]["rust"]}},
          open(sys.argv[3], "w", encoding="utf-8"), indent=2)
PYEOF
[[ -f "$OUT_DIR/l1b.json" ]] && python3 "$BENCH_DIR/gen_report.py" "$OUT_DIR/l1b.json" "$OUT_DIR/l1b_report.html"
echo "[l1b] done."
