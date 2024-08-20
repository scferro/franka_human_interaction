#pragma once

#include <geometry_msgs/msg/pose.hpp>
#include <string>

namespace franka_control {
namespace utils {

geometry_msgs::msg::Pose createPose(double x, double y, double z, double qx, double qy, double qz, double qw);
std::string generateUniqueId(const std::string& prefix);

}
}