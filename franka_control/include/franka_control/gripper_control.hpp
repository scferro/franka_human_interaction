#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/homing.hpp>

namespace franka_control {

class GripperControl {
public:
    GripperControl(const rclcpp::Node::SharedPtr& node);
    void sendHomingGoal();
    void sendGraspGoal(double width, double speed, double force);

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp_action::Client<franka_msgs::action::Homing>::SharedPtr gripper_homing_client_;
    rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_grasping_client_;
};

}