#pragma once

#include <geometry_msgs/msg/pose.hpp>
#include <string>

namespace franka_control {
namespace utils {

/**
 * @brief Create a Pose message from position and orientation components.
 * 
 * @param x X-coordinate of the position.
 * @param y Y-coordinate of the position.
 * @param z Z-coordinate of the position.
 * @param qx X-component of the quaternion orientation.
 * @param qy Y-component of the quaternion orientation.
 * @param qz Z-component of the quaternion orientation.
 * @param qw W-component of the quaternion orientation.
 * @return geometry_msgs::msg::Pose The created Pose message.
 */
geometry_msgs::msg::Pose createPose(double x, double y, double z, double qx, double qy, double qz, double qw);

/**
 * @brief Generate a unique identifier with a given prefix.
 * 
 * @param prefix The prefix to use for the unique identifier.
 * @return std::string The generated unique identifier.
 */
std::string generateUniqueId(const std::string& prefix);

}  // namespace utils
}  // namespace franka_control