#include "franka_control/utils.hpp"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace franka_control {
namespace utils {

geometry_msgs::msg::Pose createPose(double x, double y, double z, double qx, double qy, double qz, double qw)
{
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    pose.orientation.x = qx;
    pose.orientation.y = qy;
    pose.orientation.z = qz;
    pose.orientation.w = qw;
    return pose;
}

std::string generateUniqueId(const std::string& prefix)
{
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    long duration = value.count();

    std::stringstream ss;
    ss << prefix << "_" << std::setfill('0') << std::setw(13) << duration;
    return ss.str();
}

}
}