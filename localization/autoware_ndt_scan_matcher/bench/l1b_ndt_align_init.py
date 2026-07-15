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

"""L1b init via NDT's own ndt_align_srv (the TPE multi-start alignment), bypassing the unresponsive
pose_initializer. Calls /localization/pose_estimator/ndt_align_srv with the GNSS seed; the response's
success/reliable flags + returned pose are a direct diagnostic: if success/reliable are True the
map/cloud DO overlap (score-0 was just a rough seed) and we publish the REFINED pose to /initialpose3d
+ activate NDT/EKF; if False, the map and cloud don't match (a coordinate/map issue ndt_align can't
fix). Uses rclpy (the ros2 CLI is broken here)."""

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from autoware_internal_localization_msgs.srv import PoseWithCovarianceStamped as NdtAlign
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import SetBool

try:
    from rosbag2_interfaces.srv import Pause, Resume
except ImportError:  # rosbag2 not installed; pause/resume becomes a no-op
    Pause = Resume = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=None, help="seed x,y,z,qx,qy,qz,qw (map); default = GNSS")
    ap.add_argument("--align-srv", default="/localization/pose_estimator/ndt_align_srv")
    ap.add_argument("--gnss-topic", default="/sensing/gnss/pose_with_covariance")
    ap.add_argument("--cloud-topic", default="/localization/util/downsample/pointcloud")
    ap.add_argument("--yaw-var", type=float, default=0.1, help="seed yaw variance (rad^2); ~10 -> full-circle TPE search")
    ap.add_argument("--xy-var", type=float, default=4.0)
    ap.add_argument("--out", default=None, help="write refined pose and covariance fixture as JSON")
    args = ap.parse_args()

    rclpy.init()
    node = Node("l1b_ndt_align_init")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    log = node.get_logger()

    def call_player(srv_type, name):
        if srv_type is None:
            return False
        c = node.create_client(srv_type, name)
        if not c.wait_for_service(timeout_sec=3.0):
            return False
        f = c.call_async(srv_type.Request())
        rclpy.spin_until_future_complete(node, f, timeout_sec=5.0)
        return f.done()

    seed = PoseWithCovarianceStamped()
    seed.header.frame_id = "map"
    if args.pose:
        x, y, z, qx, qy, qz, qw = (float(v) for v in args.pose.split(","))
        p = seed.pose.pose
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
    else:
        got = {}
        sub = node.create_subscription(
            PoseWithCovarianceStamped, args.gnss_topic, lambda m: got.setdefault("m", m), 10)
        dl = time.monotonic() + 20.0  # wall-clock deadline (sim clock starts at 0 until /clock arrives)
        while "m" not in got and rclpy.ok() and time.monotonic() < dl:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_subscription(sub)
        if "m" not in got:
            log.error(f"no GNSS pose on {args.gnss_topic}")
            return 1
        seed.pose.pose = got["m"].pose.pose
    seed.pose.covariance = [0.0] * 36
    seed.pose.covariance[0] = seed.pose.covariance[7] = args.xy_var
    seed.pose.covariance[35] = args.yaw_var
    pp = seed.pose.pose.position
    log.info(f"seed ({pp.x:.2f}, {pp.y:.2f}, {pp.z:.2f}) -> ndt_align")

    got_cloud = {"value": False}
    cloud_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    cloud_sub = node.create_subscription(
        PointCloud2, args.cloud_topic, lambda _msg: got_cloud.__setitem__("value", True), cloud_qos)
    cloud_deadline = time.monotonic() + 20.0
    while not got_cloud["value"] and rclpy.ok() and time.monotonic() < cloud_deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(cloud_sub)
    if not got_cloud["value"]:
        log.error(f"no pointcloud on {args.cloud_topic}")
        return 1
    log.info("received an NDT input cloud before pausing the bag")

    # Freeze the bag while we align: the vehicle moves ~13 m/s, so the multi-second TPE align would
    # otherwise leave the refined pose ~50+ m stale. Seed captured just above -> pause now, so the
    # frozen vehicle is at most a few m from the seed (covered by the TPE search covariance).
    if call_player(Pause, "/rosbag2_player/pause"):
        log.info("bag paused for init")

    cli = node.create_client(NdtAlign, args.align_srv)
    if not cli.wait_for_service(timeout_sec=15.0):
        log.error(f"ndt_align service unavailable: {args.align_srv}")
        return 1
    req = NdtAlign.Request()
    req.pose_with_covariance = seed
    seed.header.stamp = node.get_clock().now().to_msg()
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
    if not (fut.done() and fut.result() is not None):
        log.error("ndt_align call timed out")
        call_player(Resume, "/rosbag2_player/resume")  # never leave the bag paused
        return 1
    res = fut.result()
    rp = res.pose_with_covariance.pose.pose.position
    log.info(f"ndt_align: success={res.success} reliable={res.reliable} "
             f"refined=({rp.x:.2f}, {rp.y:.2f}, {rp.z:.2f})")
    if not res.success:
        log.error("ndt_align did NOT succeed -> map/cloud do not match (coordinate/map issue).")
        call_player(Resume, "/rosbag2_player/resume")  # never leave the bag paused
        return 2

    # publish the REFINED pose as the EKF init + activate NDT/EKF
    refined = res.pose_with_covariance
    refined.header.frame_id = "map"
    if args.out:
        p = refined.pose.pose
        with open(args.out, "w", encoding="utf-8") as fixture_file:
            json.dump({
                "pose": [p.position.x, p.position.y, p.position.z,
                         p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
                "covariance": list(refined.pose.covariance),
            }, fixture_file, indent=2)
            fixture_file.write("\n")
        log.info(f"wrote fixed initialization fixture to {args.out}")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose3d", qos)
    for _ in range(5):
        refined.header.stamp = node.get_clock().now().to_msg()
        pub.publish(refined)
        rclpy.spin_once(node, timeout_sec=0.4)
    log.info("published refined pose on /initialpose3d")

    for name in ("/localization/pose_estimator/trigger_node",
                 "/localization/pose_twist_fusion_filter/trigger_node"):
        c = node.create_client(SetBool, name)
        if c.wait_for_service(timeout_sec=10.0):
            f = c.call_async(SetBool.Request(data=True))
            rclpy.spin_until_future_complete(node, f, timeout_sec=10.0)
            log.info(f"triggered {name}: {getattr(f.result(), 'success', None)}")
        else:
            log.error(f"trigger unavailable: {name}")

    if call_player(Resume, "/rosbag2_player/resume"):
        log.info("bag resumed")

    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
