#!/usr/bin/env python

cmake_minimum_required(VERSION 3.0.2)
project(mask_svd)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# C++ノードの設定
add_executable(point_cloud_processor src/noise_filter.cpp)
target_link_libraries(point_cloud_processor
  ${catkin_LIBRARIES}
)

# Pythonスクリプトのインストール設定
install(PROGRAMS
  scripts/kalman_estimate.py
  scripts/camera_to_xarm6eef_broadcaster.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)