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

"""L1b localization init BYPASS (benchmark-only) — see plan/ndt_bench.md.

The Autoware pose_initializer can't init on the 2021 sample-rosbag under sim-time (its TF lookups fail
with TF_OLD_DATA), so it never resets the EKF pose or fires the NDT/EKF activation triggers. For the
L1b *timing* benchmark we don't need production-grade initialization — only that NDT scan-matches each
frame. This helper drives the two things NDT needs directly:

  1. publish an initial pose (map frame) on /initialpose3d  -> EKF resets/initializes,
  2. call SetBool(true) on the NDT and EKF `*/trigger_node` services -> both become activated.

The initial pose is taken from --pose or, by default, from one /sensing/gnss/pose_with_covariance
message (already map-frame, via gnss_poser). This is NOT a faithful production init (skips the TPE
ndt_align refinement + map-height fit); it is a documented benchmark-only simplification, valid because
it only sets the *first* pose — the per-frame align cost being measured is unaffected once the EKF loop
stabilizes (discard warmup frames; verify iteration_num settles and matches OFF vs ON).

Uses only std_srvs + geometry_msgs (no autoware srv type / ros2interface needed).
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import SetBool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose",
        default=None,
        help="init pose 'x,y,z,qx,qy,qz,qw' in map frame; default = one GNSS pose",
    )
    parser.add_argument("--initialpose-topic", default="/initialpose3d")
    parser.add_argument("--gnss-topic", default="/sensing/gnss/pose_with_covariance")
    parser.add_argument("--publishes", type=int, default=5, help="times to (re)publish /initialpose3d")
    args = parser.parse_args()

    rclpy.init()
    node = Node("l1b_init_bypass")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    log = node.get_logger()

    # 1. resolve the init pose
    pose_msg = PoseWithCovarianceStamped()
    pose_msg.header.frame_id = "map"
    if args.pose:
        x, y, z, qx, qy, qz, qw = (float(v) for v in args.pose.split(","))
        p = pose_msg.pose.pose
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
        log.info(f"using --pose ({x:.2f}, {y:.2f}, {z:.2f})")
    else:
        got = {}

        def on_gnss(msg: PoseWithCovarianceStamped) -> None:
            got["msg"] = msg

        sub = node.create_subscription(PoseWithCovarianceStamped, args.gnss_topic, on_gnss, 10)
        deadline = time.monotonic() + 20.0  # wall-clock deadline (sim clock starts at 0 until /clock arrives)
        while "msg" not in got and rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_subscription(sub)
        if "msg" not in got:
            log.error(f"no GNSS pose on {args.gnss_topic}; pass --pose x,y,z,qx,qy,qz,qw")
            return 1
        pose_msg.pose.pose = got["msg"].pose.pose
        pp = pose_msg.pose.pose.position
        log.info(f"using GNSS pose ({pp.x:.2f}, {pp.y:.2f}, {pp.z:.2f})")

    # a plausible init covariance (x,y ~0.25, yaw ~0.07) so the EKF accepts it
    cov = [0.0] * 36
    cov[0] = cov[7] = 0.25
    cov[14] = 0.01
    cov[21] = cov[28] = 0.01
    cov[35] = 0.2
    pose_msg.pose.covariance = cov

    # 2. publish /initialpose3d (transient-local so a late EKF still gets it), a few times
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(PoseWithCovarianceStamped, args.initialpose_topic, qos)
    for _ in range(max(1, args.publishes)):
        pose_msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(pose_msg)
        rclpy.spin_once(node, timeout_sec=0.5)
    log.info(f"published init pose on {args.initialpose_topic}")

    # 3. activate NDT + EKF via their SetBool trigger services (auto-discovered by name)
    triggers = [
        name
        for name, types in node.get_service_names_and_types()
        if name.endswith("/trigger_node") and "std_srvs/srv/SetBool" in types
    ]
    if not triggers:
        log.error("no */trigger_node SetBool services found — is the graph up?")
        return 1
    ok = True
    for name in sorted(triggers):
        client = node.create_client(SetBool, name)
        if not client.wait_for_service(timeout_sec=10.0):
            log.error(f"trigger service not available: {name}")
            ok = False
            continue
        req = SetBool.Request()
        req.data = True
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
        if fut.done() and fut.result() is not None:
            log.info(f"triggered {name}: success={fut.result().success}")
        else:
            log.error(f"trigger call timed out: {name}")
            ok = False

    node.destroy_node()
    rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
