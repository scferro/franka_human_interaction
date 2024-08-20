#pragma once

#include <rclcpp/rclcpp.hpp>
#include <franka_control/robot_control.hpp>
#include <franka_control/collision_management.hpp>
#include <franka_control/gripper_control.hpp>
#include <franka_control/marker_handling.hpp>

namespace franka_control {

/**
 * @brief Class for sorting blocks using a Franka robot arm.
 */
class BlockSorting {
public:
    /**
     * @brief Construct a new Block Sorting object.
     * 
     * @param node Shared pointer to the ROS2 node.
     */
    BlockSorting(const rclcpp::Node::SharedPtr& node);

    /**
     * @brief Main function to perform block sorting.
     */
    void sortBlocks();

private:
    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<RobotControl> robot_control_;
    std::unique_ptr<CollisionManagement> collision_management_;
    std::unique_ptr<GripperControl> gripper_control_;
    std::unique_ptr<MarkerHandling> marker_handling_;

    /**
     * @brief Scan a block to update its position and optionally its scale.
     * 
     * @param index Index of the block to scan.
     * @param update_scale Whether to update the scale of the block.
     */
    void scanBlock(int index, bool update_scale);

    /**
     * @brief Grab a block with the robot's gripper.
     * 
     * @param index Index of the block to grab.
     */
    void grabBlock(int index);

    /**
     * @brief Place a block in a specific stack.
     * 
     * @param index Index of the block to place.
     * @param stack_id ID of the stack to place the block in.
     */
    void placeInStack(int index, int stack_id);

    /**
     * @brief Get a network prediction for a block.
     * 
     * @param index Index of the block to predict.
     * @return double The prediction result.
     */
    double getNetworkPrediction(int index);

    /**
     * @brief Train the network with a labeled block.
     * 
     * @param index Index of the block used for training.
     * @param label Label assigned to the block.
     */
    void trainNetwork(int index, int label);
};

}