#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/homing.hpp>

namespace franka_control {

/**
 * @brief Class for controlling the Franka robot's gripper.
 */
class GripperControl {
public:
    /**
     * @brief Construct a new Gripper Control object.
     * 
     * @param node Shared pointer to the ROS2 node.
     */
    GripperControl(const rclcpp::Node::SharedPtr& node);

    /**
     * @brief Send a homing goal to the gripper.
     */
    void sendHomingGoal();

    /**
     * @brief Send a grasp goal to the gripper.
     * 
     * @param width Width of the grasp.
     * @param speed Speed of the grasp motion.
     * @param force Force to apply during the grasp.
     */
    void sendGraspGoal(double width, double speed, double force);

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp_action::Client<franka_msgs::action::Homing>::SharedPtr gripper_homing_client_;
    rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_grasping_client_;
};

}