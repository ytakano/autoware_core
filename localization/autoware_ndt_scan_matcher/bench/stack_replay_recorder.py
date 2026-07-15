#!/usr/bin/env python3
"""Stream NDT diagnostics/raw poses and EKF outputs to bounded-memory CSV files."""

import argparse
import csv
import json
import math
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class CsvSink:
    def __init__(self, path, fields):
        self.file = path.open("w", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.count = 0

    def write(self, row):
        self.writer.writerow(row)
        self.count += 1

    def close(self):
        self.file.close()


class Recorder(Node):
    def __init__(self, out_dir, ndt_base_frame):
        super().__init__("stack_replay_recorder")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        pose_fields = ["stamp_ns", "x", "y", "z", "yaw"]
        self.sinks = {
            "diag": CsvSink(
                out_dir / "ndt_diagnostics.csv",
                [
                    "stamp_ns", "level", "message", "iteration_num",
                    "local_optimal_solution_oscillation_num",
                    "transform_probability",
                    "nearest_voxel_transformation_likelihood",
                    "execution_time",
                ],
            ),
            "raw": CsvSink(out_dir / "ndt_raw_pose.csv", pose_fields),
            "accepted": CsvSink(out_dir / "ndt_accepted_pose.csv", pose_fields),
            "ekf": CsvSink(out_dir / "ekf_output_pose.csv", pose_fields),
        }
        self.ndt_base_frame = ndt_base_frame
        self.create_subscription(DiagnosticArray, "/diagnostics", self.on_diag, 10)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/localization/pose_estimator/pose_with_covariance",
            self.on_accepted,
            10,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/localization/pose_with_covariance",
            self.on_ekf,
            10,
        )

    @staticmethod
    def pose_row(stamp, pose):
        return {
            "stamp_ns": stamp_ns(stamp),
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "yaw": yaw(pose.orientation),
        }

    def on_diag(self, msg):
        for status in msg.status:
            if not status.name.endswith("scan_matching_status"):
                continue
            values = {item.key: item.value for item in status.values}
            self.sinks["diag"].write(
                {
                    "stamp_ns": stamp_ns(msg.header.stamp),
                    "level": status.level,
                    "message": status.message.replace("\n", " "),
                    "iteration_num": values.get("iteration_num", ""),
                    "local_optimal_solution_oscillation_num": values.get(
                        "local_optimal_solution_oscillation_num", ""
                    ),
                    "transform_probability": values.get("transform_probability", ""),
                    "nearest_voxel_transformation_likelihood": values.get(
                        "nearest_voxel_transformation_likelihood", ""
                    ),
                    "execution_time": values.get("execution_time", ""),
                }
            )

    def on_tf(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id != self.ndt_base_frame:
                continue
            p = transform.transform.translation
            q = transform.transform.rotation
            self.sinks["raw"].write(
                {
                    "stamp_ns": stamp_ns(transform.header.stamp),
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "yaw": yaw(q),
                }
            )

    def on_accepted(self, msg):
        self.sinks["accepted"].write(self.pose_row(msg.header.stamp, msg.pose.pose))

    def on_ekf(self, msg):
        self.sinks["ekf"].write(self.pose_row(msg.header.stamp, msg.pose.pose))

    def finish(self, out_dir):
        counts = {name: sink.count for name, sink in self.sinks.items()}
        for sink in self.sinks.values():
            sink.close()
        (out_dir / "recorder_counts.json").write_text(
            json.dumps(counts, indent=2) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ndt-base-frame", default="ndt_base_link")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Recorder(args.out_dir, args.ndt_base_frame)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.finish(args.out_dir)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
