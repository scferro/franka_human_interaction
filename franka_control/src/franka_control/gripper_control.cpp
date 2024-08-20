#include "franka_control/gripper_control.hpp"

namespace franka_control {

GripperControl::GripperControl(const rclcpp::Node::SharedPtr& node)
    : node_(node)
{
    gripper_homing_client_ = rclcpp_action::create_client<franka_msgs::action::Homing>(node, "panda_gripper/homing");
    gripper_grasping_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(node, "panda_gripper/grasp");
}

void GripperControl::sendHomingGoal()
{
    if (!gripper_homing_client_->wait_for_action_server(std::chrono::seconds(1))) {
        RCLCPP_ERROR(node_->get_logger(), "Homing action server not available");
        return;
    }

    auto homing_goal = franka_msgs::action::Homing::Goal();
    auto homing_future = gripper_homing_client_->async_send_goal(homing_goal);

    if (rclcpp::spin_until_future_complete(node_, homing_future) != rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to send homing goal");
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Homing completed");
}

void GripperControl::sendGraspGoal(double width, double speed, double force)
{
    if (!gripper_grasping_client_->wait_for_action_server(std::chrono::seconds(1))) {
        RCLCPP_ERROR(node_->get_logger(), "Grasp action server not available");
        return;
    }

    auto grasp_goal = franka_msgs::action::Grasp::Goal();
    grasp_goal.width = width;
    grasp_goal.speed = speed;
    grasp_goal.force = force;
    grasp_goal.epsilon.inner = 0.2;
    grasp_goal.epsilon.outer = 0.2;

    auto grasp_future = gripper_grasping_client_->async_send_goal(grasp_goal);

    if (rclcpp::spin_until_future_complete(node_, grasp_future) != rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to send grasp goal");
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Grasp completed");
}

}