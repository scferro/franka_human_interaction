#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include "franka_hri_interfaces/action/empty_action.hpp"
#include <rclcpp_action/rclcpp_action.hpp>
#include <franka_msgs/action/grasp.hpp>

using namespace std::chrono_literals;

class Sorting : public rclcpp::Node
{
public:
  using EmptyAction = franka_hri_interfaces::action::EmptyAction;
  using GoalHandleEmptyAction = rclcpp_action::ServerGoalHandle<EmptyAction>;

  using Grasp = franka_msgs::action::Grasp;
  using GoalHandleGrasp = rclcpp_action::ClientGoalHandle<Grasp>;

  Sorting()
  : Node("sorting")
  {
    declare_parameter("rate", 200.);
    loop_rate = get_parameter("rate").as_double();

    // Gripper action client
    this->gripper_grasping_client = rclcpp_action::create_client<Grasp>(
      this,
      "panda_gripper/grasp");

    action_server_scan = rclcpp_action::create_server<EmptyAction>(
      this,
      "move_to_scan_pose",
      std::bind(&Sorting::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Sorting::handle_cancel, this, std::placeholders::_1),
      std::bind(&Sorting::handle_accepted_scan, this, std::placeholders::_1)
    );

    action_server_grab = rclcpp_action::create_server<EmptyAction>(
      this,
      "grab_block",
      std::bind(&Sorting::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Sorting::handle_cancel, this, std::placeholders::_1),
      std::bind(&Sorting::handle_accepted_grab, this, std::placeholders::_1)
    );

    action_server_left_wait = rclcpp_action::create_server<EmptyAction>(
      this,
      "wait_left",
      std::bind(&Sorting::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Sorting::handle_cancel, this, std::placeholders::_1),
      std::bind(&Sorting::handle_accepted_left_wait, this, std::placeholders::_1)
    );

    action_server_left_drop = rclcpp_action::create_server<EmptyAction>(
      this,
      "drop_left",
      std::bind(&Sorting::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Sorting::handle_cancel, this, std::placeholders::_1),
      std::bind(&Sorting::handle_accepted_left_drop, this, std::placeholders::_1)
    );

    action_server_right_wait = rclcpp_action::create_server<EmptyAction>(
      this,
      "wait_right",
      std::bind(&Sorting::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Sorting::handle_cancel, this, std::placeholders::_1),
      std::bind(&Sorting::handle_accepted_right_wait, this, std::placeholders::_1)
    );

    action_server_right_drop = rclcpp_action::create_server<EmptyAction>(
      this,
      "drop_right",
      std::bind(&Sorting::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Sorting::handle_cancel, this, std::placeholders::_1),
      std::bind(&Sorting::handle_accepted_right_drop, this, std::placeholders::_1)
    );

    int cycle_time = 1000.0 / loop_rate;
    main_timer = this->create_wall_timer(
      std::chrono::milliseconds(cycle_time),
      std::bind(&Sorting::timer_callback, this));
  }

  void initialize()
  {
    // Now safe to use shared_from_this() because the object is fully constructed
    move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "panda_manipulator");
    move_group_hand = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "hand");
  }

private:
  double loop_rate;
  rclcpp_action::Server<EmptyAction>::SharedPtr action_server_scan;
  rclcpp_action::Server<EmptyAction>::SharedPtr action_server_grab;
  rclcpp_action::Server<EmptyAction>::SharedPtr action_server_left_wait;
  rclcpp_action::Server<EmptyAction>::SharedPtr action_server_left_drop;
  rclcpp_action::Server<EmptyAction>::SharedPtr action_server_right_wait;
  rclcpp_action::Server<EmptyAction>::SharedPtr action_server_right_drop;
  rclcpp::TimerBase::SharedPtr main_timer;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_hand;

  rclcpp_action::Client<Grasp>::SharedPtr gripper_grasping_client;

  void timer_callback()
  {
    int something = 0;
  }

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const EmptyAction::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request");
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received cancel request");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted_scan(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&Sorting::move_to_scan, this, _1), goal_handle}.detach();
  }

  void handle_accepted_grab(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&Sorting::grab_block, this, _1), goal_handle}.detach();
  }

  void handle_accepted_left_wait(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&Sorting::wait_left, this, _1), goal_handle}.detach();
  }

  void handle_accepted_left_drop(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&Sorting::drop_left, this, _1), goal_handle}.detach();
  }

  void handle_accepted_right_wait(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&Sorting::wait_right, this, _1), goal_handle}.detach();
  }

  void handle_accepted_right_drop(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&Sorting::drop_right, this, _1), goal_handle}.detach();
  }

  void move_to_scan(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<EmptyAction::Feedback>();
    auto result = std::make_shared<EmptyAction::Result>();

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.3;
    target_pose.position.y = 0.0;
    target_pose.position.z = 0.25;
    target_pose.orientation.x = 1.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 0.0;

    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(5.0);
    move_group->setMaxVelocityScalingFactor(0.8);
    move_group->setMaxAccelerationScalingFactor(0.2);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      open_gripper();
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
      result->success = false;
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void grab_block(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<EmptyAction::Feedback>();
    auto result = std::make_shared<EmptyAction::Result>();

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.3;
    target_pose.position.y = 0.0;
    target_pose.position.z = 0.07;
    target_pose.orientation.x = 1.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 0.0;
    
    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(5.0);
    move_group->setMaxVelocityScalingFactor(0.5);
    move_group->setMaxAccelerationScalingFactor(0.2);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      close_gripper(0.03);
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
      result->success = false;
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void wait_left(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<EmptyAction::Feedback>();
    auto result = std::make_shared<EmptyAction::Result>();

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.3;
    target_pose.position.y = 0.2;
    target_pose.position.z = 0.4;
    target_pose.orientation.x = 1.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 0.0;
    
    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(5.0);
    move_group->setMaxVelocityScalingFactor(0.5);
    move_group->setMaxAccelerationScalingFactor(0.2);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
      result->success = false;
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void drop_left(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<EmptyAction::Feedback>();
    auto result = std::make_shared<EmptyAction::Result>();

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.3;
    target_pose.position.y = 0.3;
    target_pose.position.z = 0.15;
    target_pose.orientation.x = 1.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 0.0;

    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(5.0);
    move_group->setMaxVelocityScalingFactor(0.5);
    move_group->setMaxAccelerationScalingFactor(0.2);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      open_gripper();
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
      result->success = false;
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void wait_right(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<EmptyAction::Feedback>();
    auto result = std::make_shared<EmptyAction::Result>();

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.3;
    target_pose.position.y = -0.2;
    target_pose.position.z = 0.4;
    target_pose.orientation.x = 1.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 0.0;

    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(5.0);
    move_group->setMaxVelocityScalingFactor(0.5);
    move_group->setMaxAccelerationScalingFactor(0.2);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
            result->success = false;
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void drop_right(const std::shared_ptr<GoalHandleEmptyAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<EmptyAction::Feedback>();
    auto result = std::make_shared<EmptyAction::Result>();

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.3;
    target_pose.position.y = -0.3;
    target_pose.position.z = 0.15;
    target_pose.orientation.x = 1.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 0.0;

    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(5.0);
    move_group->setMaxVelocityScalingFactor(0.5);
    move_group->setMaxAccelerationScalingFactor(0.2);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      open_gripper();
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
      result->success = false;
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void close_gripper(double position)
  {
    send_grasp_goal(position, 0.1, 0.05, 0.05, 0.001);
    RCLCPP_INFO(this->get_logger(), "Gripper moved to position: %f", position);
  }

  void open_gripper()
  {
    send_grasp_goal(0.075, 0.2, 0.001, 0.001, 3.0);
    RCLCPP_INFO(this->get_logger(), "Gripper opened");
  }

  void send_grasp_goal(double width, double speed, double epsilon_inner, double epsilon_outer, double force)
  {
    auto goal_msg = Grasp::Goal();
    goal_msg.width = width;
    goal_msg.speed = speed;
    goal_msg.epsilon.inner = epsilon_inner;
    goal_msg.epsilon.outer = epsilon_outer;
    goal_msg.force = force;

    RCLCPP_INFO(this->get_logger(), "Sending Grasp goal...");

    auto goal_options = rclcpp_action::Client<Grasp>::SendGoalOptions();
    goal_options.goal_response_callback = 
      std::bind(&Sorting::goal_response_callback, this, std::placeholders::_1);
    goal_options.feedback_callback = 
      std::bind(&Sorting::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    goal_options.result_callback = 
      std::bind(&Sorting::result_callback, this, std::placeholders::_1);

    // auto goal_handle_future = gripper_grasping_client->async_send_goal(goal_msg, goal_options);
    this->gripper_grasping_client->async_send_goal(goal_msg, goal_options);
  }

  void goal_response_callback(std::shared_future<GoalHandleGrasp::SharedPtr> future)
  {
    auto goal_handle = future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
    } else {
      RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
    }
  }

  void feedback_callback(const std::shared_ptr<const Grasp::Feedback> feedback)
  {
    RCLCPP_INFO(this->get_logger(), "Received feedback: %f", feedback->current_width);
  }

  void result_callback(const GoalHandleGrasp::WrappedResult & result)
  {
    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(this->get_logger(), "Goal was successful");
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_WARN(this->get_logger(), "Goal was canceled");
        break;
      default:
        RCLCPP_ERROR(this->get_logger(), "Unknown result code");
        break;
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Sorting>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
