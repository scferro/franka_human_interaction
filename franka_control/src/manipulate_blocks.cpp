/**
 * @file manipulate_blocks.cpp
 * @brief ROS2 node for manipulating blocks using a Franka robot arm.
 *
 * This node provides an action server for sorting blocks and uses the BlockSorting class
 * to perform the actual sorting operation.
 *
 * PARAMETERS:
 *     None
 *
 * PUBLISHES:
 *     None
 *
 * SUBSCRIBES:
 *     None
 *
 * SERVERS:
 *     + sort_blocks (franka_hri_interfaces/action/EmptyAction) - Action server for initiating block sorting
 *
 * CLIENTS:
 *     None
 */

#include <rclcpp/rclcpp.hpp>
#include <franka_control/block_sorting.hpp>
#include <franka_hri_interfaces/action/empty_action.hpp>

/**
 * @brief Main node class for block manipulation.
 */
class ManipulateBlocks : public rclcpp::Node
{
public:
    /**
     * @brief Construct a new ManipulateBlocks object.
     */
    ManipulateBlocks()
        : Node("manipulate_blocks")
    {
        block_sorting_ = std::make_unique<franka_control::BlockSorting>(shared_from_this());
        
        // Create the action server
        action_server_ = rclcpp_action::create_server<franka_hri_interfaces::action::EmptyAction>(
            this,
            "sort_blocks",
            std::bind(&ManipulateBlocks::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&ManipulateBlocks::handleCancel, this, std::placeholders::_1),
            std::bind(&ManipulateBlocks::handleAccepted, this, std::placeholders::_1)
        );
    }

private:
    std::unique_ptr<franka_control::BlockSorting> block_sorting_;
    rclcpp_action::Server<franka_hri_interfaces::action::EmptyAction>::SharedPtr action_server_;

    /**
     * @brief Handle incoming goal requests.
     * 
     * @param uuid The goal UUID
     * @param goal The goal request
     * @return rclcpp_action::GoalResponse The response to the goal request
     */
    rclcpp_action::GoalResponse handleGoal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const franka_hri_interfaces::action::EmptyAction::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received goal request");
        (void)uuid;
        (void)goal;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    /**
     * @brief Handle cancellation requests.
     * 
     * @param goal_handle The goal handle of the action being cancelled
     * @return rclcpp_action::CancelResponse The response to the cancellation request
     */
    rclcpp_action::CancelResponse handleCancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received cancel request");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    /**
     * @brief Handle accepted goals.
     * 
     * This method is called when a goal is accepted. It starts a new thread to perform
     * the block sorting operation.
     * 
     * @param goal_handle The goal handle of the accepted action
     */
    void handleAccepted(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
    {
        std::thread{[this, goal_handle]() {
            block_sorting_->sortBlocks();
            goal_handle->succeed(std::make_shared<franka_hri_interfaces::action::EmptyAction::Result>());
        }}.detach();
    }
};

/**
 * @brief Main function to initialize and run the ManipulateBlocks node.
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return int Exit status
 */
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulateBlocks>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}