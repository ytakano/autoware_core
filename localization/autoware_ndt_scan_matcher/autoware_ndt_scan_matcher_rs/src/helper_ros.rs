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

//! ROS-gated helper glue that bridges `geometry_msgs::msg::Pose` to the pure
//! [`realtime_ndt_scan_matcher::helper`] kernels. Only built under the `ros` feature (the pure numeric path
//! lives in the engine crate).

/// Count the maximum consecutive direction inversions over a `geometry_msgs::Pose` trajectory.
///
/// Reads only `position.{x,y,z}` from each pose and delegates to the pure
/// [`realtime_ndt_scan_matcher::helper::count_oscillation`] (the private step-classification kernel lives in
/// the engine crate, so this extracts the positions first).
#[must_use]
pub fn count_oscillation_poses(poses: &[crate::ros_msgs::geometry_msgs__msg__Pose]) -> i32 {
    let positions: alloc::vec::Vec<[f64; 3]> = poses
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    realtime_ndt_scan_matcher::helper::count_oscillation(&positions)
}

#[cfg(test)]
#[allow(clippy::allow_attributes, reason = "test code")]
mod tests {
    use super::*;

    fn xyz(x: f64) -> [f64; 3] {
        [x, 0.0, 0.0]
    }

    // The Pose path must agree with the pure path on the same positions.
    #[test]
    fn count_oscillation_poses_matches_pure_path() {
        let pose = |x: f64| {
            let mut p = crate::ros_msgs::geometry_msgs__msg__Pose::default();
            p.position.x = x;
            p
        };
        for xs in [
            vec![0.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 0.0, 1.0, 2.0, 1.0],
        ] {
            let positions: Vec<[f64; 3]> = xs.iter().map(|&x| xyz(x)).collect();
            let poses: Vec<_> = xs.iter().map(|&x| pose(x)).collect();
            assert_eq!(
                count_oscillation_poses(&poses),
                realtime_ndt_scan_matcher::helper::count_oscillation(&positions)
            );
        }
    }
}
