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

// L1a — node / end-to-end benchmark (see plan/ndt_bench.md, doc/book/src/quality/benchmarks.md).
//
// Drives the FULL NDT node in-process (reusing the integration-test fixture + stubs), replaying the
// synthetic `standard_sequence` cloud frame-by-frame through the real sensor callback, and records the
// node's own per-frame `exe_time_ms` (ingest + align + covariance + convergence). The engine is
// selected at BUILD time by `NDT_USE_RUST` (OFF = C++ / multigrid_ndt_omp, ON = the Rust port); the
// `bench/run_l1a.sh` runner does the OFF/ON two-pass and merges the two per-engine JSONs into one
// report via `bench/gen_report.py`. This binary emits ONE engine's samples; the engine key/label and
// the output path come from the environment set by the runner.
//
// Env knobs: L1A_ENGINE (cpp|rust, default cpp), L1A_LABEL (default derived), L1A_OUT (JSON path,
// default /tmp/l1a_<engine>.json), L1A_ITERS (default 200), L1A_WARMUP (default 20).

#include "../stub_ekf_pose_publisher.hpp"
#include "../test_fixture.hpp"
#include "../test_util.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_internal_debug_msgs/msg/float32_stamped.hpp>
#include <autoware_internal_debug_msgs/msg/int32_stamped.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace
{
using autoware_internal_debug_msgs::msg::Float32Stamped;
using autoware_internal_debug_msgs::msg::Int32Stamped;
using namespace std::chrono_literals;

std::string env_or(const char * key, const std::string & fallback)
{
  const char * v = std::getenv(key);
  return (v != nullptr && v[0] != '\0') ? std::string(v) : fallback;
}

int env_int(const char * key, const int fallback)
{
  const char * v = std::getenv(key);
  if (v == nullptr || v[0] == '\0') {
    return fallback;
  }
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

// Subscribes to the node's per-frame debug outputs. `count` increments once per `exe_time_ms`, which
// the node publishes ONLY on a fully-successful scan-match — so it doubles as the "frame processed"
// signal that serializes the driver (the sensor QoS is keep_last(1), so one frame in flight at a time).
class DebugRecorder : public rclcpp::Node
{
public:
  DebugRecorder() : Node("l1a_debug_recorder")
  {
    exe_sub_ = create_subscription<Float32Stamped>(
      "/exe_time_ms", 10, [this](const Float32Stamped::ConstSharedPtr msg) {
        last_exe_ms_.store(static_cast<double>(msg->data));
        count_.fetch_add(1);
      });
    iter_sub_ = create_subscription<Int32Stamped>(
      "/iteration_num", 10,
      [this](const Int32Stamped::ConstSharedPtr msg) { last_iter_.store(msg->data); });
  }

  std::atomic<uint64_t> count_{0};
  std::atomic<double> last_exe_ms_{0.0};
  std::atomic<int> last_iter_{-1};

private:
  rclcpp::Subscription<Float32Stamped>::SharedPtr exe_sub_;
  rclcpp::Subscription<Int32Stamped>::SharedPtr iter_sub_;
};

class L1aNodeFrameBench : public TestNDTScanMatcher
{
protected:
  int64_t num_threads_for_test() const override { return 1; }  // serial = the fair WCET baseline
};
}  // namespace

TEST_F(L1aNodeFrameBench, replay_frames_record_exe_time)  // NOLINT
{
  const std::string engine = env_or("L1A_ENGINE", "cpp");
  const std::string label = env_or(
    "L1A_LABEL", engine == "rust" ? "Rust (node, autoware_ndt_scan_matcher_rs)"
                                   : "C++ (node, multigrid_ndt_omp)");
  const std::string out_path = env_or("L1A_OUT", "/tmp/l1a_" + engine + ".json");
  const int iters = env_int("L1A_ITERS", 200);
  const int warmup = env_int("L1A_WARMUP", 20);

  auto recorder = std::make_shared<DebugRecorder>();
  auto ekf_pub = std::make_shared<StubEkfPosePublisher>();

  std::thread spin_thread([&]() {
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node_);
    exec.add_node(recorder);
    exec.spin();
  });
  std::thread map_thread([&]() { rclcpp::spin(pcd_loader_); });

  // Activate the node so the sensor callback runs the align (rather than early-returning).
  ASSERT_TRUE(trigger_node_client_->send_trigger_node(true));

  const sensor_msgs::msg::PointCloud2 cloud = make_default_sensor_pcd();
  const uint32_t n_points = cloud.width;

  // A synthetic, strictly-increasing stamp clock (0.2 s/frame) decoupled from wall time: it keeps the
  // EKF-pose pushes monotonic (SmartPoseBuffer clears on a backward stamp) and each sensor stamp
  // bracketed by two EKF poses. The node's delay check only warns (never early-returns), so old
  // synthetic stamps are fine.
  rclcpp::Time stamp(1000, 0, RCL_ROS_TIME);
  const rclcpp::Duration frame_dt = rclcpp::Duration::from_seconds(0.2);
  const rclcpp::Duration bracket = rclcpp::Duration::from_seconds(0.05);

  // The stub map tile sits at (100, 100); offset the initial guess by 0.5 m so every frame does
  // representative align work (a stable handful of iterations) instead of converging in one step —
  // otherwise the per-frame cost is unrealistically low and the OFF/ON comparison is uninformative.
  constexpr double kGuessX = 100.5;
  constexpr double kGuessY = 99.5;

  // Publish one frame: bracket the sensor stamp with two EKF poses at the offset guess, publish the
  // cloud, and block until the node reports a new exe_time_ms (or timeout).
  auto run_frame = [&](std::chrono::milliseconds timeout) -> bool {
    ekf_pub->publish_pose(kGuessX, kGuessY, stamp - bracket);
    ekf_pub->publish_pose(kGuessX, kGuessY, stamp + bracket);
    std::this_thread::sleep_for(20ms);  // let the initial-pose callbacks ingest before the sensor cb
    const uint64_t before = recorder->count_.load();
    sensor_msgs::msg::PointCloud2 framed = cloud;
    framed.header.stamp = stamp;
    sensor_pcd_publisher_->publish_pcd(framed);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (recorder->count_.load() == before) {
      if (std::chrono::steady_clock::now() > deadline || !rclcpp::ok()) {
        stamp = stamp + frame_dt;
        return false;
      }
      std::this_thread::sleep_for(2ms);
    }
    stamp = stamp + frame_dt;
    return true;
  };

  // Prime the map: publish an EKF pose and give the 1 Hz map-update timer time to fetch the tile from
  // the stub loader (hasTarget() must be true before any frame succeeds). Then retry frames until the
  // first success, bounded.
  ekf_pub->publish_pose(kGuessX, kGuessY, stamp);
  std::this_thread::sleep_for(2500ms);

  bool ready = false;
  for (int attempt = 0; attempt < 100 && rclcpp::ok() && !ready; ++attempt) {
    ready = run_frame(500ms);
  }
  ASSERT_TRUE(ready) << "node never produced exe_time_ms — map/activation/interpolation gate not met";

  for (int i = 0; i < warmup && rclcpp::ok(); ++i) {
    run_frame(3000ms);
  }

  std::vector<double> samples_ms;
  samples_ms.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters && rclcpp::ok();) {
    const uint64_t before = recorder->count_.load();
    if (run_frame(3000ms) && recorder->count_.load() > before) {
      samples_ms.push_back(recorder->last_exe_ms_.load());
      ++i;
    }
    // a timed-out frame is retried (not counted) — no measurement bias
  }
  const int iteration_num = recorder->last_iter_.load();

  rclcpp::shutdown();
  spin_thread.join();
  map_thread.join();

  EXPECT_EQ(samples_ms.size(), static_cast<size_t>(iters));
  for (const double s : samples_ms) {
    EXPECT_TRUE(std::isfinite(s) && s > 0.0);
  }

  // Emit one engine's block in the schema bench/gen_report.py consumes (the runner merges cpp+rust).
  FILE * f = std::fopen(out_path.c_str(), "w");
  ASSERT_NE(f, nullptr) << "cannot open " << out_path;
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"benchmark\": \"L1a node end-to-end (synthetic standard_sequence cloud)\",\n");
  std::fprintf(
    f,
    "  \"meta\": {\"n_points\": %u, \"iters\": %d, \"warmup\": %d, \"resolution\": 2.0, "
    "\"max_iterations\": 30, \"num_threads\": 1, \"clock\": \"node exe_time_ms (system_clock)\", "
    "\"unit\": \"ms\"},\n",
    n_points, iters, warmup);
  std::fprintf(f, "  \"engines\": {\n");
  std::fprintf(f, "    \"%s\": {\n", engine.c_str());
  std::fprintf(f, "      \"label\": \"%s\",\n", label.c_str());
  std::fprintf(f, "      \"iteration_num\": %d,\n", iteration_num);
  std::fprintf(f, "      \"samples_ms\": [");
  for (size_t i = 0; i < samples_ms.size(); ++i) {
    std::fprintf(f, "%s%.6f", (i == 0 ? "" : ", "), samples_ms[i]);
  }
  std::fprintf(f, "]\n    }\n  }\n}\n");
  std::fclose(f);
  RCLCPP_INFO_STREAM(
    rclcpp::get_logger("l1a_bench"),
    "wrote " << samples_ms.size() << " samples (engine=" << engine << ", iter=" << iteration_num
             << ") to " << out_path);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return result;
}
