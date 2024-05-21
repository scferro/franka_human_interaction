cmake_minimum_required(VERSION 3.22)
project(franka_control)

# Specify C++ Standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(franka_hri_interfaces REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Declare a C++ executable
add_executable(manipulate_blocks src/manipulate_blocks.cpp)
ament_target_dependencies(manipulate_blocks
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  std_srvs
  tf2_ros
  visualization_msgs
  rclcpp_action
  franka_hri_interfaces
  franka_msgs
  std_msgs
)

# Install the executable
install(TARGETS
  manipulate_blocks
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
