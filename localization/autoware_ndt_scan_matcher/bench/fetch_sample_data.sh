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

# Download-if-absent provisioning of the Autoware sample data used by the L1b node benchmark:
# `sample-rosbag` (recorded real LiDAR) + `sample-map-rosbag` (its PCD map). It is INTENTIONALLY only
# invoked by the opt-in L1b runner — never a default `colcon test` gate (which must stay hermetic and
# offline; the bag is ~193 MB).
#
# Cache = the standard Autoware layout under ${AUTOWARE_DATA_DIR:-~/autoware_data}, so a machine already
# provisioned by the `demo_artifacts` ansible role is a cache hit:
#   $AUTOWARE_DATA_DIR/recordings/bags/sample-rosbag/
#   $AUTOWARE_DATA_DIR/maps/sample-map-rosbag/
#
# URLs + SHA256 are the pins from autoware/ansible/roles/demo_artifacts/tasks/main.yaml (source of
# truth). On success it prints two `KEY=path` lines (SAMPLE_ROSBAG_DIR=..., SAMPLE_MAP_DIR=...) that the
# runner reads. Offline + absent ⇒ prints the URL + checksum and exits non-zero (never hangs).

set -euo pipefail

AUTOWARE_DATA_DIR="${AUTOWARE_DATA_DIR:-$HOME/autoware_data}"

BAG_URL="https://autoware-files.s3.us-west-2.amazonaws.com/recordings/bags/demos/sample-rosbag.zip"
BAG_SHA="5f9d36353393b3d249212153c19049822b1298db56512aa045b4f7f6fc37cf88"
MAP_URL="https://autoware-files.s3.us-west-2.amazonaws.com/maps/demos/sample-map-rosbag.zip"
MAP_SHA="07e2da0b0bf12e2324f7083c2ce5556fb8044c50cef1da6428ab9084c3903bc8"

verify_sha() {  # <file> <expected_sha256>  -> 0 if match
  local file="$1" expected="$2" actual
  actual="$(sha256sum "$file" | cut -d' ' -f1)"
  [[ "$actual" == "$expected" ]]
}

# ensure <url> <sha256> <dest_parent> <extracted_subdir>
# Guarantees <dest_parent>/<extracted_subdir> exists, from a checksum-verified archive. Re-verifies the
# archive on every run when it is still present (guards a truncated cached download).
ensure() {
  local url="$1" sha="$2" dest_parent="$3" subdir="$4"
  local archive="$dest_parent/$(basename "$url")"
  local extracted="$dest_parent/$subdir"
  mkdir -p "$dest_parent"

  if [[ -f "$archive" ]]; then
    if verify_sha "$archive" "$sha"; then
      echo "[fetch] cached archive OK: $archive" >&2
    else
      echo "[fetch] cached archive checksum MISMATCH — re-downloading: $archive" >&2
      rm -f "$archive"
    fi
  fi

  if [[ ! -d "$extracted" || -z "$(ls -A "$extracted" 2>/dev/null)" ]]; then
    if [[ ! -f "$archive" ]]; then
      echo "[fetch] downloading $(basename "$url") -> $archive" >&2
      local tmp="$archive.part"
      if ! curl -fL -C - --retry 3 -o "$tmp" "$url"; then
        echo "[fetch] ERROR: download failed (offline?). Fetch it manually:" >&2
        echo "          URL:    $url" >&2
        echo "          SHA256: $sha" >&2
        echo "          into:   $dest_parent/  (then unzip)" >&2
        echo "        or run the Autoware 'demo_artifacts' ansible role." >&2
        rm -f "$tmp"
        return 1
      fi
      if ! verify_sha "$tmp" "$sha"; then
        echo "[fetch] ERROR: checksum mismatch after download of $(basename "$url")" >&2
        rm -f "$tmp"
        return 1
      fi
      mv -f "$tmp" "$archive"  # atomic: only a fully-verified archive lands at the final path
    fi
    echo "[fetch] extracting $(basename "$archive") -> $dest_parent/" >&2
    unzip -q -o "$archive" -d "$dest_parent"
  else
    echo "[fetch] cached extract OK: $extracted" >&2
  fi

  if [[ ! -d "$extracted" ]]; then
    echo "[fetch] ERROR: expected '$extracted' after extraction but it is absent" >&2
    echo "        (the archive layout may differ — inspect $archive)" >&2
    return 1
  fi
}

ensure "$BAG_URL" "$BAG_SHA" "$AUTOWARE_DATA_DIR/recordings/bags" "sample-rosbag"
ensure "$MAP_URL" "$MAP_SHA" "$AUTOWARE_DATA_DIR/maps" "sample-map-rosbag"

echo "SAMPLE_ROSBAG_DIR=$AUTOWARE_DATA_DIR/recordings/bags/sample-rosbag"
echo "SAMPLE_MAP_DIR=$AUTOWARE_DATA_DIR/maps/sample-map-rosbag"
