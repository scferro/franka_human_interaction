cmake_minimum_required(VERSION 3.8)
project(franka_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_servo REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(franka_hri_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Include directories
include_directories(
  include
  ${moveit_core_INCLUDE_DIRS}
  ${moveit_ros_planning_interface_INCLUDE_DIRS}
  ${moveit_servo_INCLUDE_DIRS}
)

# Create the franka_control library
add_library(franka_control
  src/franka_control/robot_control.cpp
  src/franka_control/collision_management.cpp
  src/franka_control/gripper_control.cpp
  src/franka_control/marker_handling.cpp
  src/franka_control/block_sorting.cpp
  src/franka_control/utils.cpp
)

ament_target_dependencies(franka_control
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  moveit_core
  moveit_ros_planning
  moveit_servo
  franka_msgs
  geometry_msgs
  visualization_msgs
  franka_hri_interfaces
)

# Create the manipulate_blocks executable
add_executable(manipulate_blocks src/manipulate_blocks.cpp)
target_link_libraries(manipulate_blocks franka_control)
ament_target_dependencies(manipulate_blocks
  rclcpp
  rclcpp_action
  franka_hri_interfaces
)

# Create the servo_control_node executable
add_executable(servo_control_node src/servo_control_node.cpp)
target_link_libraries(servo_control_node
  franka_control
  ${tf2_LIBRARIES}
  ${tf2_eigen_LIBRARIES}
  ${tf2_geometry_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}
)
ament_target_dependencies(servo_control_node
  rclcpp
  moveit_servo
  franka_hri_interfaces
  std_srvs
  tf2
  tf2_eigen
  tf2_geometry_msgs
  tf2_ros
  trajectory_msgs
)

# Teleop control node
add_executable(teleop_control_node src/teleop_control_node.cpp)
target_link_libraries(teleop_control_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${franka_hri_interfaces_LIBRARIES}
  ${std_srvs_LIBRARIES}
)
ament_target_dependencies(teleop_control_node
  rclcpp
  sensor_msgs
  franka_hri_interfaces
  std_srvs
)

# Install targets
install(TARGETS
franka_control
manipulate_blocks
servo_control_node
teleop_control_node 
ARCHIVE DESTINATION lib
LIBRARY DESTINATION lib
RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()