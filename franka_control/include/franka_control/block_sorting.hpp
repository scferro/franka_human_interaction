#pragma once

#include <rclcpp/rclcpp.hpp>
#include <franka_control/robot_control.hpp>
#include <franka_control/collision_management.hpp>
#include <franka_control/gripper_control.hpp>
#include <franka_control/marker_handling.hpp>

namespace franka_control {

class BlockSorting {
public:
    BlockSorting(const rclcpp::Node::SharedPtr& node);
    void sortBlocks();

private:
    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<RobotControl> robot_control_;
    std::unique_ptr<CollisionManagement> collision_management_;
    std::unique_ptr<GripperControl> gripper_control_;
    std::unique_ptr<MarkerHandling> marker_handling_;

    void scanBlock(int index, bool update_scale);
    void grabBlock(int index);
    void placeInStack(int index, int stack_id);
    double getNetworkPrediction(int index);
    void trainNetwork(int index, int label);
};

}