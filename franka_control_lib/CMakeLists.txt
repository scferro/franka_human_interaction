cmake_minimum_required(VERSION 3.22)
project(franka_control_lib)

# Create libraries
add_library(franka_control_lib src/franka_controller.cpp src/motion_generator.cpp)

# find dependencies
find_package(ament_cmake REQUIRED)

# Use target_include_directories so that #include"mylibrary/header.hpp" works
target_include_directories(franka_control_lib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)

# specify additional compilation flags for the library
target_compile_options(franka_control_lib PUBLIC -Wall -Wextra -pedantic)

# Enable c++17 support.
target_compile_features(franka_control_lib PUBLIC cxx_std_17)

# install the include files by copying the whole include directory
install(DIRECTORY include/franka_control_lib DESTINATION include)

# Create a CMake Exported Target containing the lib and exe.
# Also create CMake Export called projet_name-targets
# The CMake Export contains files that allow other CMake projects
# to find this project. It must be installed separately.
install(TARGETS franka_control_lib EXPORT franka_control_lib-targets)

# The project_name-targets created by install(TARGETS) needs to be installed.
# install(EXPORT ...) will generate a file called project_name-config.cmake
# that contains the exported targets.
# After installation this file will then be found when calling
# find_package(project_name) from another cmake project
# A user can then target_link_libraries(target project_name::library)
# to use the libraries installed here
install(EXPORT franka_control_lib-targets
        FILE franka_control_lib-config.cmake
        NAMESPACE franka_control_lib::
        DESTINATION lib/cmake/${PROJECT_NAME})