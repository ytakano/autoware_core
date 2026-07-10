cmake_minimum_required(VERSION 3.14)
project(autoware_ndt_scan_matcher)

# cspell:ignore multigrid

find_package(autoware_cmake REQUIRED)
autoware_package()

# Compile flags for SIMD instructions
# Be careful to change these options, especially when `ndt_omp` implementation is used.
# All packages linked to `ndt_omp` should use the same SIMD instruction set.
# In case mismatched instruction set are used, program causes a crash at its initialization
# because of a misaligned access to the `Eigen` libraries' data structure.
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  # For x86_64 architecture, SIMD instruction set is fixed below versions,
  # because the `ndt_omp` is optimized to these versions.
  add_compile_options(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
else()
  # For other architecture, like arm64, compile flags are generally prepared by compiler
  # march=native is disabled as default for specific depending pcl libraries
  # or pre-building packages for other computers.
  if(BUILD_WITH_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

find_package(OpenMP)

find_package(PCL REQUIRED COMPONENTS common io registration)
include_directories(${PCL_INCLUDE_DIRS})

# When OFF (default), the package builds the original C++ node unchanged and has no dependency on
# Rust or Corrosion. When ON, the node is built from the self-contained `rust/` tree, which drives
# its compute through the `autoware_ndt_scan_matcher_rs` crate over a C ABI; the original `src/` and
# `include/` files are not compiled. Both source sets implement the same
# `autoware::ndt_scan_matcher::NDTScanMatcher` node; exactly one is selected here, so there is no
# symbol clash.
option(NDT_USE_RUST "Build the node from the Rust port instead of the original C++" OFF)

if(NDT_USE_RUST)
  # Corrosion is fetched at configure time; for offline builds switch GIT_* to a local SOURCE_DIR.
  include(FetchContent)
  FetchContent_Declare(Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
  )
  FetchContent_MakeAvailable(Corrosion)
  # The `ros` feature builds the bindgen ROS-message bindings (geometry_msgs Pose) used by the
  # zero-copy FFI shims; bindgen needs the rosidl C headers, so pass geometry_msgs' include dir.
  corrosion_import_crate(MANIFEST_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/autoware_ndt_scan_matcher_rs/Cargo.toml
    FEATURES ros
  )
  find_package(geometry_msgs REQUIRED)
  string(JOIN ":" NDT_ROS_INCLUDE_DIRS ${geometry_msgs_INCLUDE_DIRS})
  corrosion_set_env_vars(autoware_ndt_scan_matcher_rs
    "ROS_INCLUDE_DIRS=${NDT_ROS_INCLUDE_DIRS}")
  # Generate the C FFI header (autoware_ndt_scan_matcher_rs.h) from the crate's `#[repr(C)]` structs
  # and `extern "C"` fns via cbindgen — a single source of truth, so the C++ side never hand-syncs
  # against the Rust FFI. Corrosion attaches the generated header's directory to the crate target,
  # so every target that links the crate sees it transitively.
  corrosion_experimental_cbindgen(
    TARGET autoware_ndt_scan_matcher_rs
    HEADER_NAME autoware_ndt_scan_matcher_rs.h)
endif()

ament_auto_add_library(multigrid_ndt_omp SHARED
  src/ndt_omp/multi_voxel_grid_covariance_omp.cpp
  src/ndt_omp/multigrid_ndt_omp.cpp
  src/ndt_omp/estimate_covariance.cpp
)
target_link_libraries(multigrid_ndt_omp ${PCL_LIBRARIES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(multigrid_ndt_omp OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found")
endif()

# The node library is built from either the original C++ or the Rust-port tree. `helper` and
# `particle` are shared, unchanged sources used by both.
set(NDT_NODE_SOURCES
  src/ndt_scan_matcher_helper.cpp
  src/particle.cpp
)
if(NDT_USE_RUST)
  list(APPEND NDT_NODE_SOURCES
    rust/src/ndt_scan_matcher_core.cpp
    rust/src/map_update_module.cpp
    rust/src/map_update_module_rust.cpp
    rust/src/ndt_scan_matcher_rust_shell.cpp
    rust/src/ndt_scan_matcher_sensor_rust.cpp
    rust/src/ndt_scan_matcher_rust_host.cpp
    rust/src/ndt_scan_matcher_align_rust.cpp
  )
else()
  list(APPEND NDT_NODE_SOURCES
    src/ndt_scan_matcher_core.cpp
    src/map_update_module.cpp
  )
endif()

ament_auto_add_library(${PROJECT_NAME} SHARED ${NDT_NODE_SOURCES})

link_directories(${PCL_LIBRARY_DIRS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} multigrid_ndt_omp)

if(NDT_USE_RUST)
  # The Rust-port headers mirror the original include path, so they must be searched before the
  # original include/ directory. BEFORE guarantees that; it is safe because this branch never
  # compiles the original node sources.
  target_include_directories(${PROJECT_NAME} BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rust/include>)
  # The Rust-port headers/sources reuse a few unchanged originals by their unqualified (same-folder)
  # name: the private helper in src/, and the sibling headers (guarded.hpp, hyper_parameters.hpp,
  # ndt_omp/*) in include/autoware/ndt_scan_matcher/. The forked rust/ headers still win over the
  # originals because a quoted include searches the including file's own directory first.
  target_include_directories(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/autoware/ndt_scan_matcher>)
  target_link_libraries(${PROJECT_NAME} autoware_ndt_scan_matcher_rs)
  # PUBLIC so the generated node executable and any test linking this library also compile with the
  # Rust-port declarations.
  target_compile_definitions(${PROJECT_NAME} PUBLIC NDT_USE_RUST)
endif()

autoware_agnocast_wrapper_register_node(${PROJECT_NAME}
  PLUGIN "autoware::ndt_scan_matcher::NDTScanMatcher"
  EXECUTABLE ${PROJECT_NAME}_node
  ROS2_EXECUTOR MultiThreadedExecutor
  AGNOCAST_EXECUTOR CallbackIsolatedAgnocastExecutor
)

# Offline replay benchmark comparing the two engines in one executable. Opt-in and off by default so
# it never slows the normal build; it drives both the C++ multigrid_ndt_omp engine and the Rust port,
# so it requires NDT_USE_RUST=ON. Built standalone and run directly (bench/run.sh orchestrates it).
option(NDT_BUILD_BENCH "Build the offline replay benchmark (requires NDT_USE_RUST)" OFF)
if(NDT_BUILD_BENCH)
  if(NOT NDT_USE_RUST)
    message(FATAL_ERROR "NDT_BUILD_BENCH requires NDT_USE_RUST=ON (the benchmark drives both engines)")
  endif()
  add_executable(ndt_bench_replay bench/ndt_bench_replay.cpp)
  target_include_directories(ndt_bench_replay PRIVATE include ${PCL_INCLUDE_DIRS})
  target_link_libraries(ndt_bench_replay
    multigrid_ndt_omp ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)
endif()

# L1 node-level end-to-end benchmark (bench/l1a_node_frame_bench.cpp). Opt-in and off by default so
# the default `colcon test` / `./test.sh` never build or run it. Unlike NDT_BUILD_BENCH it drives the
# node (not the engine directly), so it builds under BOTH engine modes — bench/run_l1a.sh does the
# NDT_USE_RUST OFF/ON two-pass and merges the two per-engine JSONs. The target is registered below,
# inside BUILD_TESTING (it needs ament_cmake_gtest + the test fixture/stubs).
option(NDT_BUILD_BENCH_L1 "Build the L1 node-level end-to-end benchmark" OFF)

if(BUILD_TESTING)
  add_launch_test(
    test/test_ndt_scan_matcher_launch.py
    TIMEOUT "30"
  )

  find_package(ament_cmake_gtest REQUIRED)

  # Node-level integration tests construct NDTScanMatcher directly through test_fixture.hpp (an
  # original header). They characterize the original C++ node, so they build only when it is the one
  # compiled (OFF). The Rust port's behavior is covered by the differential unit tests below.
  if(NOT NDT_USE_RUST)
    ament_auto_add_gtest(standard_sequence_for_initial_pose_estimation
      test/test_cases/standard_sequence_for_initial_pose_estimation.cpp
      TIMEOUT "300"
    )
    ament_auto_add_gtest(once_initialize_at_out_of_map_then_initialize_correctly
      test/test_cases/once_initialize_at_out_of_map_then_initialize_correctly.cpp
      TIMEOUT "300"
    )
    ament_auto_add_gtest(particles_num_less_than_publish_num
      test/test_cases/particles_num_less_than_publish_num.cpp
      TIMEOUT "300"
    )
    ament_auto_add_gtest(missing_sensor_for_initial_pose_estimation
      test/test_cases/missing_sensor_for_initial_pose_estimation.cpp
      TIMEOUT "300"
    )
    ament_auto_add_gtest(align_service_tpe_baseline
      test/test_cases/align_service_tpe_baseline.cpp
      TIMEOUT "300"
    )
  endif()

  # Fast, dependency-light unit tests for the pure covariance / oscillation math.
  ament_auto_add_gtest(test_estimate_covariance
    test/test_estimate_covariance.cpp
  )
  target_link_libraries(test_estimate_covariance multigrid_ndt_omp ${PCL_LIBRARIES})

  ament_auto_add_gtest(test_ndt_scan_matcher_helper
    test/test_ndt_scan_matcher_helper.cpp
  )
  target_link_libraries(test_ndt_scan_matcher_helper ${PROJECT_NAME})

  # L1a node-level benchmark (opt-in via NDT_BUILD_BENCH_L1). Registered outside the NDT_USE_RUST
  # split so it builds for BOTH engines; it drives the full node through the integration-test fixture
  # and records the node's own exe_time_ms. Run directly by bench/run_l1a.sh (env-configured), not as
  # a default ctest. Links ${PROJECT_NAME} (which carries the selected engine + the NDT_USE_RUST
  # define transitively); the Rust crate is linked explicitly only in the ON build.
  if(NDT_BUILD_BENCH_L1)
    ament_auto_add_gtest(l1a_node_frame_bench
      test/bench/l1a_node_frame_bench.cpp
      TIMEOUT "600"
    )
    target_link_libraries(l1a_node_frame_bench ${PROJECT_NAME} multigrid_ndt_omp ${PCL_LIBRARIES})
    if(NDT_USE_RUST)
      target_link_libraries(l1a_node_frame_bench autoware_ndt_scan_matcher_rs)
    endif()
  endif()

  if(NDT_USE_RUST)
    # Differential and FFI tests for the Rust port. Each proves equivalence against the original C++
    # engine (linked as multigrid_ndt_omp) or exercises the crate's C ABI directly. Tests that use
    # the C++ RAII wrapper header (ndt_scan_matcher_rs.hpp) link the node library, which exports the
    # rust/include directory.

    # Run the crate's own unit tests through CTest so `colcon test` / `./test.sh` pick them up. Cargo
    # writes its artifacts under the CMake build dir, not the source tree.
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    add_test(
      NAME autoware_ndt_scan_matcher_rs_cargo_test
      COMMAND ${CARGO_EXECUTABLE} test --features ros
              --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/autoware_ndt_scan_matcher_rs/Cargo.toml
              --target-dir ${CMAKE_CURRENT_BINARY_DIR}/rs_test_target
    )
    set_tests_properties(autoware_ndt_scan_matcher_rs_cargo_test PROPERTIES
      ENVIRONMENT "ROS_INCLUDE_DIRS=${NDT_ROS_INCLUDE_DIRS}")

    # Differential vs the C++ engine (link multigrid_ndt_omp as the reference).
    ament_auto_add_gtest(test_voxel_grid test/test_voxel_grid.cpp)
    target_link_libraries(test_voxel_grid
      multigrid_ndt_omp ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_align test/test_align.cpp)
    target_link_libraries(test_align
      multigrid_ndt_omp ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_estimate_covariance_multi test/test_estimate_covariance_multi.cpp)
    target_link_libraries(test_estimate_covariance_multi
      multigrid_ndt_omp ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_ndt_engine test/test_ndt_engine.cpp)
    target_link_libraries(test_ndt_engine
      multigrid_ndt_omp ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_node_run_align test/test_node_run_align.cpp)
    target_link_libraries(test_node_run_align
      multigrid_ndt_omp ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_estimate_pose_covariance test/test_estimate_pose_covariance.cpp)
    target_link_libraries(test_estimate_pose_covariance
      multigrid_ndt_omp ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)

    # Exercise the crate's C ABI directly (no C++ engine reference needed).
    ament_auto_add_gtest(test_tpe_ffi test/test_tpe_ffi.cpp)
    target_link_libraries(test_tpe_ffi autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_convergence_verdict test/test_convergence_verdict.cpp)
    target_link_libraries(test_convergence_verdict autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_node_pose_callbacks test/test_node_pose_callbacks.cpp)
    target_link_libraries(test_node_pose_callbacks autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_map_update_verdict test/test_map_update_verdict.cpp)
    target_link_libraries(test_map_update_verdict autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_ndt_align_service_decision test/test_ndt_align_service_decision.cpp)
    target_link_libraries(test_ndt_align_service_decision autoware_ndt_scan_matcher_rs)

    # Use the C++ RAII wrapper (ndt_scan_matcher_rs.hpp) — link the node library.
    ament_auto_add_gtest(test_ndt_scan_matcher_rs_handle test/test_ndt_scan_matcher_rs_handle.cpp)
    target_link_libraries(test_ndt_scan_matcher_rs_handle
      ${PROJECT_NAME} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_regularization_buffer test/test_regularization_buffer.cpp)
    target_link_libraries(test_regularization_buffer
      ${PROJECT_NAME} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_initial_pose_buffer test/test_initial_pose_buffer.cpp)
    target_link_libraries(test_initial_pose_buffer
      ${PROJECT_NAME} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_map_update_state test/test_map_update_state.cpp)
    target_link_libraries(test_map_update_state
      ${PROJECT_NAME} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_sensor_points_prepare test/test_sensor_points_prepare.cpp)
    target_link_libraries(test_sensor_points_prepare
      ${PROJECT_NAME} ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)

    ament_auto_add_gtest(test_sensor_points_match test/test_sensor_points_match.cpp)
    target_link_libraries(test_sensor_points_match
      ${PROJECT_NAME} ${PCL_LIBRARIES} autoware_ndt_scan_matcher_rs)
  endif()
endif()

autoware_ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  launch
  config
)
