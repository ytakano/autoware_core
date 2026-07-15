#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${WS_ROOT:-/autoware_workspace}"
BAG_DIR="${BAG_DIR:-/home/ytakano/autoware_ista_data/loc_bag}"
MAP_PATH="${MAP_PATH:-/home/ytakano/autoware_ista_map/pointcloud_map_tiles}"
MAP_METADATA="${MAP_METADATA:-$MAP_PATH/metadata.yaml}"
MAP_SOURCE="${MAP_SOURCE:-/home/ytakano/autoware_ista_map/pointcloud_map.pcd}"
OUT_DIR="${OUT_DIR:-$WS_ROOT/stack_replay_results}"
RUN_SECONDS="${RUN_SECONDS:-0}"
RSS_LIMIT_KIB="${RSS_LIMIT_KIB:-16777216}"
MEM_AVAILABLE_LIMIT_KIB="${MEM_AVAILABLE_LIMIT_KIB:-8388608}"
STAMP_OFFSET_NS="${STAMP_OFFSET_NS:-2371531504000000}"
FIXTURE="$OUT_DIR/fixed_initial_pose.json"
PKG=autoware_ndt_scan_matcher
RUN_PIDS=()

mkdir -p "$OUT_DIR"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source_env() {
  local profile="$1"
  set +u
  source /opt/ros/humble/setup.bash
  source "$WS_ROOT/install/setup.bash"
  source "$WS_ROOT/install_stack_${profile}/setup.bash"
  set -u
}

cleanup() {
  local pid
  for pid in "${RUN_PIDS[@]:-}"; do
    kill -INT -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${RUN_PIDS[@]:-}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
  RUN_PIDS=()
}
trap cleanup EXIT INT TERM

check_governor() {
  local file
  for file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ "$(<"$file")" == performance ]] || {
      echo "[stack] ERROR: $file is not performance" >&2
      return 1
    }
  done
}

write_manifest() {
  local dir="$1" engine="$2" repetition="$3"
  {
    printf 'engine=%s\nrepetition=%s\n' "$engine" "$repetition"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'autoware_revision=%s\n' "$(git -C "$WS_ROOT/src/core/autoware_core" rev-parse HEAD)"
    printf 'paper_revision=%s\n' "$(git -C "$WS_ROOT" rev-parse HEAD)"
    printf 'bag_db_sha256=%s\n' "$(sha256sum "$BAG_DIR/loc_bag_0.db3" | awk '{print $1}')"
    printf 'map_source_sha256=%s\n' "$(sha256sum "$MAP_SOURCE" | awk '{print $1}')"
    printf 'map_metadata_sha256=%s\n' "$(sha256sum "$MAP_METADATA" | awk '{print $1}')"
    printf 'fixture_sha256=%s\n' "$(sha256sum "$FIXTURE" | awk '{print $1}')"
    printf 'run_seconds=%s\n' "$RUN_SECONDS"
    printf 'stamp_offset_ns=%s\n' "$STAMP_OFFSET_NS"
    printf 'ndt_package_prefix=%s\n' "$(ros2 pkg prefix "$PKG")"
    free -k
    rg -n "" /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor       /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq       /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
  } > "$dir/manifest.txt"
}

wait_for_graph() {
  local attempt
  for attempt in $(seq 1 60); do
    ros2 node list 2>/dev/null | rg -q '/localization/pose_estimator/ndt_scan_matcher' && return 0
    sleep 1
  done
  echo "[stack] ERROR: NDT node did not appear" >&2
  return 1
}

start_graph_and_relay() {
  local dir="$1"
  AUTOWARE_EKF_POSE_TRACE="$dir/ekf_pose_updates.csv" setsid ros2 launch "$PKG"     ndt_l1b_loc.launch.xml pointcloud_map_path:="$MAP_PATH" \
    pointcloud_map_metadata_path:="$MAP_METADATA" enable_whole_load:=false use_sim_time:=true     > "$dir/launch.log" 2>&1 &
  RUN_PIDS+=("$!")
  setsid python3 "$BENCH_DIR/l1b_restamp_relay.py" --offset-ns "$STAMP_OFFSET_NS" --trace "$dir/cloud_mapping.csv" > "$dir/relay.log" 2>&1 &
  RUN_PIDS+=("$!")
  setsid python3 "$BENCH_DIR/stack_replay_recorder.py"     --out-dir "$dir" > "$dir/recorder.log" 2>&1 &
  RUN_PIDS+=("$!")
  wait_for_graph
}

start_player() {
  local start_paused="${2:-0}"
  local pause_args=()
  [[ "$start_paused" == 1 ]] && pause_args+=(--start-paused)
  local dir="$1"
  local command=(ros2 bag play "$BAG_DIR" "${pause_args[@]}" --clock 100 --rate 1.0
    --disable-keyboard-controls --read-ahead-queue-size 100
    --remap /localization/util/downsample/pointcloud:=/l1b/raw_cloud
    /localization/twist_estimator/twist_with_covariance:=/l1b/raw_twist
    --topics /localization/util/downsample/pointcloud
    /localization/twist_estimator/twist_with_covariance
    /sensing/gnss/pose_with_covariance /tf_static)
  if [[ "$RUN_SECONDS" -gt 0 ]]; then
    setsid timeout --signal=INT "${RUN_SECONDS}s" "${command[@]}" > "$dir/player.log" 2>&1 &
  else
    setsid "${command[@]}" > "$dir/player.log" 2>&1 &
  fi
  PLAYER_PID=$!
  RUN_PIDS+=("$PLAYER_PID")
}

monitor_resources() {
  local dir="$1" player_pid="$2" peak=0
  while kill -0 "$player_pid" 2>/dev/null; do
    local available rss
    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    rss="$(ps -u "$(id -u)" -o rss= | awk '{sum += $1} END {print sum + 0}')"
    (( rss > peak )) && peak="$rss"
    printf '%s,%s,%s\n' "$(date +%s)" "$rss" "$available" >> "$dir/resource_usage.csv"
    if (( rss > RSS_LIMIT_KIB || available < MEM_AVAILABLE_LIMIT_KIB )); then
      printf 'OOM_GUARD rss_kib=%s available_kib=%s\n' "$rss" "$available" |
        tee -a "$dir/resource_guard_failure.txt" >&2
      kill -INT -- "-$player_pid" 2>/dev/null || true
      return 1
    fi
    check_governor || {
      printf 'CPU_CONFIGURATION_CHANGED\n' > "$dir/cpu_guard_failure.txt"
      kill -INT -- "-$player_pid" 2>/dev/null || true
      return 1
    }
    sleep 1
  done
  printf 'peak_user_rss_kib=%s\n' "$peak" >> "$dir/manifest.txt"
}

build_profile() {
  local profile="$1" use_rust="$2"
  echo "[stack] building $profile profile"
  set +u
  source /opt/ros/humble/setup.bash
  source "$WS_ROOT/install/setup.bash"
  set -u
  colcon build --packages-select autoware_ekf_localizer "$PKG"     --build-base "build_stack_$profile" --install-base "install_stack_$profile"     --cmake-force-configure --cmake-args -DCMAKE_BUILD_TYPE=Release "-DNDT_USE_RUST=$use_rust"
}

prepare_fixture() {
  [[ -f "$FIXTURE" ]] && return 0
  local dir="$OUT_DIR/init_prepass"
  mkdir -p "$dir"
  source_env cpp
  start_graph_and_relay "$dir"
  start_player "$dir"
  sleep 3
  python3 "$BENCH_DIR/l1b_ndt_align_init.py" --out "$FIXTURE" > "$dir/init.log" 2>&1
  cleanup
  [[ -s "$FIXTURE" ]]
}

run_pass() {
  local profile="$1" engine="$2" repetition="$3"
  local dir="$OUT_DIR/${engine}${repetition}"
  rm -rf "$dir"
  mkdir -p "$dir"
  source_env "$profile"
  write_manifest "$dir" "$engine" "$repetition"
  printf 'epoch,user_rss_kib,mem_available_kib\n' > "$dir/resource_usage.csv"
  start_graph_and_relay "$dir"
  start_player "$dir" 1
  sleep 2
  python3 "$BENCH_DIR/l1b_init_bypass.py" --fixture "$FIXTURE" --pause-player     > "$dir/init.log" 2>&1
  monitor_resources "$dir" "$PLAYER_PID" &
  local monitor_pid=$!
  wait "$PLAYER_PID" || true
  wait "$monitor_pid"
  sleep 2
  cleanup
  local required
  for required in cloud_mapping.csv ndt_diagnostics.csv ndt_raw_pose.csv     ndt_accepted_pose.csv ekf_output_pose.csv ekf_pose_updates.csv recorder_counts.json; do
    [[ -s "$dir/$required" ]] || {
      echo "[stack] ERROR: missing or empty $dir/$required" >&2
      return 1
    }
  done
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$dir/manifest.txt"
}

check_governor
[[ -f "$BAG_DIR/loc_bag_0.db3" ]]
[[ -d "$MAP_PATH" ]]
[[ -f "$MAP_METADATA" ]]
[[ -f "$MAP_SOURCE" ]]
if [[ -n "${RUN_ONLY:-}" ]]; then
  case "$RUN_ONLY" in
    cpp1) run_pass cpp cpp 1 ;;
    cpp2) run_pass cpp cpp 2 ;;
    rust1) run_pass rust rust 1 ;;
    rust2) run_pass rust rust 2 ;;
    *)
      echo "[stack] ERROR: unknown RUN_ONLY value: $RUN_ONLY" >&2
      exit 2
      ;;
  esac
  exit 0
fi

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  build_profile cpp OFF
  build_profile rust ON
fi
prepare_fixture
for run in cpp1 rust1 rust2 cpp2; do
  SKIP_BUILD=1 RUN_ONLY="$run" bash "$0"
done

python3 "$BENCH_DIR/analyze_stack_replay.py"   --cpp "$OUT_DIR/cpp1" --cpp "$OUT_DIR/cpp2"   --rust "$OUT_DIR/rust1" --rust "$OUT_DIR/rust2"   --out "$OUT_DIR/summary.json" > "$OUT_DIR/analysis.log"
echo "[stack] complete: $OUT_DIR/summary.json"
