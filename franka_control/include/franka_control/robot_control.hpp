#pragma once

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace franka_control {

/**
 * @brief Class for controlling the Franka robot's movements.
 */
class RobotControl {
public:
    /**
     * @brief Construct a new Robot Control object.
     * 
     * @param node Shared pointer to the ROS2 node.
     */
    RobotControl(const rclcpp::Node::SharedPtr& node);

    /**
     * @brief Move the robot to a target pose.
     * 
     * @param target_pose The target pose for the robot.
     * @param planning_time Maximum time allowed for motion planning.
     * @param vel_factor Velocity scaling factor.
     * @param accel_factor Acceleration scaling factor.
     */
    void moveToPose(const geometry_msgs::msg::PoseStamped& target_pose, double planning_time, double vel_factor, double accel_factor);

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
};

}