#include <rclcpp/rclcpp.hpp>
#include <franka_control/block_sorting.hpp>
#include <franka_hri_interfaces/action/empty_action.hpp>

class ManipulateBlocks : public rclcpp::Node
{
public:
    ManipulateBlocks()
        : Node("manipulate_blocks")
    {
        block_sorting_ = std::make_unique<franka_control::BlockSorting>(shared_from_this());

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

    rclcpp_action::GoalResponse handleGoal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const franka_hri_interfaces::action::EmptyAction::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received goal request");
        (void)uuid;
        (void)goal;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleCancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received cancel request");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handleAccepted(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
    {
        std::thread{[this, goal_handle]() {
            block_sorting_->sortBlocks();
            goal_handle->succeed(std::make_shared<franka_hri_interfaces::action::EmptyAction::Result>());
        }}.detach();
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulateBlocks>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}