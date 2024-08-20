#pragma once

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace franka_control {

class RobotControl {
public:
    RobotControl(const rclcpp::Node::SharedPtr& node);
    void moveToPose(const geometry_msgs::msg::PoseStamped& target_pose, double planning_time, double vel_factor, double accel_factor);

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
};

}