cmake_minimum_required(VERSION 3.22)
project(franka_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(franka_control_lib REQUIRED)
find_package(nuturtlebot_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(nav_msgs REQUIRED)

add_executable(plates src/plates.cpp)
ament_target_dependencies(plates rclcpp std_msgs geometry_msgs sensor_msgs std_srvs tf2 tf2_ros visualization_msgs nuturtlebot_msgs nav_msgs)
target_link_libraries(plates franka_control_lib::franka_control_lib "${cpp_typesupport_target}")

install(TARGETS
  plates
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  srv
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
