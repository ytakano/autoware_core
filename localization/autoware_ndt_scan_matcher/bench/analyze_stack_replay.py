#!/usr/bin/env python3
"""Analyze bounded-memory C++/Rust NDT-EKF stack replay traces."""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def distance(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stats(values):
    finite = [v for v in values if math.isfinite(v)]
    return {
        "count": len(finite),
        "p50": quantile(finite, 0.50),
        "p95": quantile(finite, 0.95),
        "p99": quantile(finite, 0.99),
        "max": max(finite) if finite else None,
    }


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def by_stamp(rows):
    return {int(row["stamp_ns"]): row for row in rows}


def interpolate_pose(stamps, table, target, tolerance_ns=40_000_000):
    if len(stamps) < 2:
        return None
    index = bisect.bisect_left(stamps, target)
    if index == 0 or index == len(stamps):
        return None
    before_stamp = stamps[index - 1]
    after_stamp = stamps[index]
    if target - before_stamp > tolerance_ns or after_stamp - target > tolerance_ns:
        return None
    before = table[before_stamp]
    after = table[after_stamp]
    ratio = (target - before_stamp) / (after_stamp - before_stamp)
    before_yaw = float(before["yaw"])
    yaw_delta = math.atan2(
        math.sin(float(after["yaw"]) - before_yaw),
        math.cos(float(after["yaw"]) - before_yaw),
    )
    return {
        "stamp_ns": str(target),
        "x": str(float(before["x"]) + ratio * (float(after["x"]) - float(before["x"]))),
        "y": str(float(before["y"]) + ratio * (float(after["y"]) - float(before["y"]))),
        "z": str(float(before["z"]) + ratio * (float(after["z"]) - float(before["z"]))),
        "yaw": str(before_yaw + ratio * yaw_delta),
    }


def load_run(path):
    relay = read_rows(path / "cloud_mapping.csv")
    raw = by_stamp(read_rows(path / "ndt_raw_pose.csv"))
    diag = by_stamp(read_rows(path / "ndt_diagnostics.csv"))
    accepted = by_stamp(read_rows(path / "ndt_accepted_pose.csv"))
    ekf_output = by_stamp(read_rows(path / "ekf_output_pose.csv"))
    ekf_stamps = sorted(ekf_output)
    ekf_trace = {
        int(row["measurement_ns"]): row
        for row in read_rows(path / "ekf_pose_updates.csv")
    }
    frames = {}
    for row in relay:
        key = (int(row["original_ns"]), row["data_crc32"])
        stamp = int(row["restamped_ns"])
        frames[key] = {
            "sequence": int(row["sequence"]),
            "stamp_ns": stamp,
            "raw": raw.get(stamp),
            "diag": diag.get(stamp),
            "quality_pass": stamp in accepted,
            "accepted": accepted.get(stamp),
            "ekf_update": ekf_trace.get(stamp),
            "ekf_output": interpolate_pose(ekf_stamps, ekf_output, stamp),
        }
    return frames


def recovery(frame_deltas, start_index, translation_threshold, yaw_threshold, consecutive=5):
    streak = 0
    start_stamp = frame_deltas[start_index]["original_ns"]
    for row in frame_deltas[start_index:]:
        if row["ekf_translation"] < translation_threshold and row["ekf_yaw"] < yaw_threshold:
            streak += 1
            if streak == consecutive:
                return {
                    "frames": row["index"] - frame_deltas[start_index]["index"] - consecutive + 2,
                    "seconds": (row["original_ns"] - start_stamp) / 1e9,
                }
        else:
            streak = 0
    return None


def compare(left, right, label):
    common = sorted(set(left) & set(right))
    rows = []
    gate_disagreements = 0
    ekf_gate_comparable = 0
    ekf_gate_disagreements = 0
    innovation_translation = []
    innovation_yaw = []
    for index, key in enumerate(common):
        a = left[key]
        b = right[key]
        if a["quality_pass"] != b["quality_pass"]:
            gate_disagreements += 1
        if not (a["raw"] and b["raw"] and a["ekf_output"] and b["ekf_output"]):
            continue
        row = {
            "index": index,
            "original_ns": key[0],
            "sequence_left": a["sequence"],
            "sequence_right": b["sequence"],
            "ndt_translation": distance(a["raw"], b["raw"]),
            "ndt_yaw": angle_diff(float(a["raw"]["yaw"]), float(b["raw"]["yaw"])),
            "ekf_translation": distance(a["ekf_output"], b["ekf_output"]),
            "ekf_yaw": angle_diff(
                float(a["ekf_output"]["yaw"]), float(b["ekf_output"]["yaw"])
            ),
        }
        rows.append(row)
        ua = a["ekf_update"]
        ub = b["ekf_update"]
        if ua and ub:
            ekf_gate_comparable += 1
            if ua["accepted"] != ub["accepted"]:
                ekf_gate_disagreements += 1
            try:
                innovation_translation.append(
                    math.hypot(
                        float(ua["innovation_x"]) - float(ub["innovation_x"]),
                        float(ua["innovation_y"]) - float(ub["innovation_y"]),
                    )
                )
                innovation_yaw.append(
                    angle_diff(float(ua["innovation_yaw"]), float(ub["innovation_yaw"]))
                )
            except ValueError:
                pass

    ndt_translation = [row["ndt_translation"] for row in rows]
    ndt_yaw = [row["ndt_yaw"] for row in rows]
    ekf_translation = [row["ekf_translation"] for row in rows]
    ekf_yaw = [row["ekf_yaw"] for row in rows]
    result = {
        "label": label,
        "matched_input_frames": len(common),
        "ekf_gate_comparable": ekf_gate_comparable,
        "ekf_gate_disagreements": ekf_gate_disagreements,
        "comparable_frames": len(rows),
        "quality_gate_disagreements": gate_disagreements,
        "ndt_translation_m": stats(ndt_translation),
        "ndt_yaw_rad": stats(ndt_yaw),
        "ekf_translation_m": stats(ekf_translation),
        "ekf_yaw_rad": stats(ekf_yaw),
        "innovation_translation_difference_m": stats(innovation_translation),
        "innovation_yaw_difference_rad": stats(innovation_yaw),
    }
    if rows:
        peak_index = max(range(len(rows)), key=lambda i: rows[i]["ndt_translation"])
        peak = dict(rows[peak_index])
        peak["attenuation_ratio"] = (
            peak["ekf_translation"] / peak["ndt_translation"]
            if peak["ndt_translation"] > 0.0
            else None
        )
        result["max_ndt_event"] = peak
        result["recovery_primary"] = recovery(rows, peak_index, 0.10, 0.01)
        result["recovery_sensitivity"] = {
            "0.20m_0.02rad": recovery(rows, peak_index, 0.20, 0.02),
            "0.05m_0.01rad": recovery(rows, peak_index, 0.05, 0.01),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", action="append", required=True, type=Path)
    parser.add_argument("--rust", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if len(args.cpp) != 2 or len(args.rust) != 2:
        parser.error("exactly two --cpp and two --rust run directories are required")

    cpp = [load_run(path) for path in args.cpp]
    rust = [load_run(path) for path in args.rust]
    comparisons = [
        compare(cpp[0], cpp[1], "cpp-repeat"),
        compare(rust[0], rust[1], "rust-repeat"),
    ]
    for ci, c_run in enumerate(cpp, 1):
        for ri, r_run in enumerate(rust, 1):
            comparisons.append(compare(c_run, r_run, f"cpp{ci}-rust{ri}"))

    summary = {
        "method": "pointcloud original stamp + CRC32 paired stack replay",
        "ground_truth_available": False,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
