// Copyright 2024 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STUB_EKF_POSE_PUBLISHER_HPP_
#define STUB_EKF_POSE_PUBLISHER_HPP_

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

// Publishes `ekf_pose_with_covariance` (the topic the NDT node's `initial_pose_buffer_` interpolates
// and from which the 1 Hz map-update timer reads its query position). The standard integration tests
// drive the align-service path instead, so this stub is only needed by the L1a per-frame benchmark
// to reach `callback_sensor_points` (which publishes `exe_time_ms`).
class StubEkfPosePublisher : public rclcpp::Node
{
public:
  StubEkfPosePublisher() : Node("stub_ekf_pose_publisher")
  {
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/ekf_pose_with_covariance", 10);
  }

  // Publish an identity-orientation pose at (x, y) in the map frame, stamped `stamp`. The covariance
  // mirrors `make_pose()` in test_util.hpp (a plausible EKF diagonal).
  void publish_pose(const double x, const double y, const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped pose{};
    pose.header.frame_id = "map";
    pose.header.stamp = stamp;
    pose.pose.pose.position.x = x;
    pose.pose.pose.position.y = y;
    pose.pose.pose.position.z = 0.0;
    pose.pose.pose.orientation.w = 1.0;
    pose.pose.covariance[0] = 0.25;
    pose.pose.covariance[7] = 0.25;
    pose.pose.covariance[14] = 0.0025;
    pose.pose.covariance[21] = 0.0006853891909122467;
    pose.pose.covariance[28] = 0.0006853891909122467;
    pose.pose.covariance[35] = 0.06853891909122467;
    pose_publisher_->publish(pose);
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
};

#endif  // STUB_EKF_POSE_PUBLISHER_HPP_
