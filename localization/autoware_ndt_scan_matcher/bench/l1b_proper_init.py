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

"""Call the Autoware pose_initializer's /localization/initialize via rclpy (the ros2 CLI service call
is broken in this minimal image). method=AUTO(0) with an empty pose -> GNSS-seeded init, which runs
NDT's ndt_align (TPE) to REFINE the seed until it locks onto the map — unlike the raw-GNSS trigger
bypass, this gives NDT a converged initial pose. If --pose x,y,z,qx,qy,qz,qw is given, uses DIRECT(1)
with that pose. Reports the response so we can see whether the pose_initializer succeeds on this bag."""

import argparse
import sys

import rclpy
from rclpy.node import Node

from autoware_internal_localization_msgs.srv import InitializeLocalization
from geometry_msgs.msg import PoseWithCovarianceStamped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=None, help="DIRECT init pose x,y,z,qx,qy,qz,qw (map frame)")
    ap.add_argument("--service", default="/localization/initialize")
    args = ap.parse_args()

    rclpy.init()
    node = Node("l1b_proper_init")
    cli = node.create_client(InitializeLocalization, args.service)
    if not cli.wait_for_service(timeout_sec=15.0):
        node.get_logger().error(f"service not available: {args.service}")
        return 1

    req = InitializeLocalization.Request()
    if args.pose:
        x, y, z, qx, qy, qz, qw = (float(v) for v in args.pose.split(","))
        ps = PoseWithCovarianceStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = node.get_clock().now().to_msg()
        ps.pose.pose.position.x, ps.pose.pose.position.y, ps.pose.pose.position.z = x, y, z
        (ps.pose.pose.orientation.x, ps.pose.pose.orientation.y,
         ps.pose.pose.orientation.z, ps.pose.pose.orientation.w) = qx, qy, qz, qw
        cov = [0.0] * 36
        cov[0] = cov[7] = 1.0
        cov[35] = 0.2
        ps.pose.covariance = cov
        req.pose_with_covariance = [ps]
        req.method = 1  # DIRECT
        node.get_logger().info(f"DIRECT init at ({x:.1f}, {y:.1f}, {z:.1f})")
    else:
        req.method = 0  # AUTO (GNSS)
        node.get_logger().info("AUTO init (GNSS-seeded + ndt_align refine)")

    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
    if fut.done() and fut.result() is not None:
        node.get_logger().info(f"initialize response: {fut.result()}")
        rc = 0
    else:
        node.get_logger().error("initialize call timed out / no response")
        rc = 1
    node.destroy_node()
    rclpy.try_shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
