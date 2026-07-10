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

"""Record the NDT node's per-frame ``exe_time_ms`` for the L1b benchmark.

Subscribes to ``/exe_time_ms`` (autoware_internal_debug_msgs/Float32Stamped) and ``/iteration_num``
(Int32Stamped) while a rosbag is replayed through the node, and on shutdown writes one engine's block
in the schema ``bench/gen_report.py`` consumes:

    {"engines": {"<key>": {"label": ..., "iteration_num": ..., "samples_ms": [...]}},
     "meta": {...}, "benchmark": ...}

``run_l1b.sh`` runs this once per engine (OFF -> cpp, ON -> rust) and merges the two. Stop it with
SIGINT/SIGTERM (the runner sends it after ``ros2 bag play`` finishes); the JSON is written on exit.
"""

import argparse
import json
import signal
import sys

import rclpy
from rclpy.node import Node

from autoware_internal_debug_msgs.msg import Float32Stamped, Int32Stamped


class ExeTimeRecorder(Node):
    def __init__(self) -> None:
        super().__init__("l1b_exe_time_recorder")
        self.samples_ms: list[float] = []
        self.last_iter: int = -1
        self.create_subscription(Float32Stamped, "/exe_time_ms", self._on_exe, 50)
        self.create_subscription(Int32Stamped, "/iteration_num", self._on_iter, 50)

    def _on_exe(self, msg: Float32Stamped) -> None:
        self.samples_ms.append(float(msg.data))

    def _on_iter(self, msg: Int32Stamped) -> None:
        self.last_iter = int(msg.data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="cpp", help="engine key: cpp (OFF) or rust (ON)")
    parser.add_argument("--label", default=None, help="human-readable engine label")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    label = args.label or (
        "Rust (node, autoware_ndt_scan_matcher_rs)"
        if args.engine == "rust"
        else "C++ (node, multigrid_ndt_omp)"
    )

    rclpy.init()
    node = ExeTimeRecorder()

    # Write the JSON and shut down cleanly on SIGINT/SIGTERM (sent by run_l1b.sh after the bag ends).
    def _stop(_signum, _frame):
        rclpy.try_shutdown()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass

    doc = {
        "benchmark": "L1b node end-to-end (Autoware sample-rosbag)",
        "meta": {
            "iters": len(node.samples_ms),
            "num_threads": args.num_threads,
            "clock": "node exe_time_ms (system_clock)",
            "unit": "ms",
            "note": "real sample-rosbag replay; per-frame ingest+align+covariance",
        },
        "engines": {
            args.engine: {
                "label": label,
                "iteration_num": node.last_iter,
                "samples_ms": node.samples_ms,
            }
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    node.get_logger().info(
        f"wrote {len(node.samples_ms)} samples (engine={args.engine}) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
