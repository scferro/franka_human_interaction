#include "franka_control/robot_control.hpp"

namespace franka_control {

RobotControl::RobotControl(const rclcpp::Node::SharedPtr& node)
    : node_(node)
{
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "panda_manipulator");
}

void RobotControl::moveToPose(const geometry_msgs::msg::PoseStamped& target_pose, double planning_time, double vel_factor, double accel_factor)
{
    RCLCPP_INFO(node_->get_logger(), "Planning movement to goal...");

    move_group_->setPoseTarget(target_pose);
    move_group_->setPlanningTime(planning_time);
    move_group_->setMaxVelocityScalingFactor(vel_factor);
    move_group_->setMaxAccelerationScalingFactor(accel_factor);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
        RCLCPP_INFO(node_->get_logger(), "Planning succeeded! Moving to goal.");
        move_group_->execute(plan);
        RCLCPP_INFO(node_->get_logger(), "Moved to goal pose.");
    } else {
        RCLCPP_INFO(node_->get_logger(), "Planning failed");
    }
}

}