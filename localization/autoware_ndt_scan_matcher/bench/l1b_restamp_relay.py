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

"""Re-stamp a PointCloud2 to the current (sim) clock — L1b benchmark helper (see plan/ndt_bench.md).

The Autoware urban-localization sample bag stores its `/localization/util/downsample/pointcloud` with a
header.stamp ~27 days behind the bag's /clock (and behind the EKF pose stamps). NDT then can't
interpolate its per-frame initial guess (EKF poses at /clock time) at the cloud's stale stamp
-> "Couldn't interpolate pose", never scan-matches. This relay subscribes to the bag cloud (remapped
to an intermediate topic), rewrites header.stamp = node clock now (sim time, matching the EKF poses),
and republishes to NDT's input topic — a benchmark-only time-alignment shim, not a behavior change to
the cloud data. Run with use_sim_time:=true so `now()` is the sim clock.
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import PointCloud2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-topic", default="/l1b/raw_cloud")
    parser.add_argument("--out-topic", default="/localization/util/downsample/pointcloud")
    parser.add_argument("--in-twist", default="/l1b/raw_twist")
    parser.add_argument("--out-twist", default="/localization/twist_estimator/twist_with_covariance")
    args = parser.parse_args()

    rclpy.init()
    node = Node("l1b_restamp_relay")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])

    # Best-effort sensor QoS on both ends (matches NDT's SensorDataQoS().keep_last(1)).
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    pub = node.create_publisher(PointCloud2, args.out_topic, qos)
    count = {"n": 0}

    def on_cloud(msg: PointCloud2) -> None:
        msg.header.stamp = node.get_clock().now().to_msg()  # align to sim-now (== EKF pose time base)
        pub.publish(msg)
        count["n"] += 1
        if count["n"] % 100 == 1:
            node.get_logger().info(f"re-stamped {count['n']} clouds -> {args.out_topic}")

    node.create_subscription(PointCloud2, args.in_topic, on_cloud, qos)

    # Twist also carries the stale recorded time base; the EKF's delay compensation rejects it,
    # freezing the EKF at the init pose. Re-stamp it identically (reliable QoS: EKF expects it).
    tw_pub = node.create_publisher(TwistWithCovarianceStamped, args.out_twist, 50)

    def on_twist(msg: TwistWithCovarianceStamped) -> None:
        msg.header.stamp = node.get_clock().now().to_msg()
        tw_pub.publish(msg)

    node.create_subscription(TwistWithCovarianceStamped, args.in_twist, on_twist, 50)
    node.get_logger().info(f"re-stamp relay: {args.in_topic} -> {args.out_topic} (header.stamp := sim now)")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
