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
find_package(franka_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(franka_hri_interfaces REQUIRED)

# Include directories
include_directories(include)

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

# Install targets
install(TARGETS
  franka_control
  manipulate_blocks
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()