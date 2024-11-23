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
# find_package(moveit_servo REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(object_recognition_msgs REQUIRED)
find_package(franka_hri_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${moveit_core_INCLUDE_DIRS}
  ${moveit_ros_planning_interface_INCLUDE_DIRS}
  ${moveit_servo_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Create the manipulate_blocks executable
add_executable(manipulate_blocks src/manipulate_blocks.cpp)
ament_target_dependencies(manipulate_blocks
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  moveit_core
  moveit_ros_planning
  franka_msgs
  geometry_msgs
  visualization_msgs
  franka_hri_interfaces
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  trajectory_msgs
  tf2
  tf2_eigen
  std_srvs
)

# # Create the servo_control_node executable
# add_executable(servo_control_node src/servo_control_node.cpp)
# target_link_libraries(servo_control_node
#   ${tf2_LIBRARIES}
#   ${tf2_eigen_LIBRARIES}
#   ${tf2_geometry_msgs_LIBRARIES}
#   ${tf2_ros_LIBRARIES}
# )
# ament_target_dependencies(servo_control_node
#   rclcpp
#   moveit_servo
#   franka_hri_interfaces
#   std_srvs
#   tf2
#   tf2_eigen
#   tf2_geometry_msgs
#   tf2_ros
#   trajectory_msgs
#   Eigen3
# )

# # Teleop control node
# add_executable(teleop_control_node src/teleop_control_node.cpp)
# target_link_libraries(teleop_control_node
#   ${rclcpp_LIBRARIES}
#   ${sensor_msgs_LIBRARIES}
#   ${franka_hri_interfaces_LIBRARIES}
#   ${std_srvs_LIBRARIES}
# )
# ament_target_dependencies(teleop_control_node
#   rclcpp
#   sensor_msgs
#   franka_hri_interfaces
#   std_srvs
#   franka_msgs
#   rclcpp_action
# )

# # VLA control node
# add_executable(vla_control_node src/vla_control_node.cpp)
# target_link_libraries(vla_control_node
#   ${rclcpp_LIBRARIES}
#   ${sensor_msgs_LIBRARIES}
#   ${franka_hri_interfaces_LIBRARIES}
#   ${std_srvs_LIBRARIES}
#   ${OpenCV_LIBS}
# )
# ament_target_dependencies(vla_control_node
#   rclcpp
#   sensor_msgs
#   franka_hri_interfaces
#   std_srvs
#   franka_msgs
#   rclcpp_action
#   geometry_msgs
#   OpenCV
#   cv_bridge
#   message_filters
# )

# Install targets
install(TARGETS
  manipulate_blocks
  # servo_control_node
  # teleop_control_node
  # vla_control_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files and config
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()