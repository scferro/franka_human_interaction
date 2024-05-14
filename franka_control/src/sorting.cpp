#include <chrono>
#include <memory>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <rclcpp_action/rclcpp_action.hpp>
#include <franka_msgs/action/grasp.hpp>

#include "franka_hri_interfaces/action/pose_action.hpp"
#include "franka_hri_interfaces/srv/create_box.hpp"

using namespace std::chrono_literals;

class Sorting : public rclcpp::Node
{
public:
  using PoseAction = franka_hri_interfaces::action::PoseAction;
  using GoalHandlePoseAction = rclcpp_action::ServerGoalHandle<PoseAction>;
  using CreateBox = franka_hri_interfaces::srv::CreateBox;

  Sorting()
  : Node("sorting")
  {
    declare_parameter("rate", 200.);
    loop_rate = get_parameter("rate").as_double();

    action_server_move_to_pose = rclcpp_action::create_server<PoseAction>(
      this,
      "move_to_pose",
      std::bind(&Sorting::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Sorting::handle_cancel, this, std::placeholders::_1),
      std::bind(&Sorting::handle_accepted, this, std::placeholders::_1)
    );

    create_box_service = create_service<CreateBox>(
      "create_box",
      std::bind(&Sorting::create_box_callback, this, std::placeholders::_1, std::placeholders::_2)
    );

    int cycle_time = 1000.0 / loop_rate;
    main_timer = this->create_wall_timer(
      std::chrono::milliseconds(cycle_time),
      std::bind(&Sorting::timer_callback, this)
    );
  }

  void initialize()
  {
    // Now safe to use shared_from_this() because the object is fully constructed
    move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "panda_manipulator");
    move_group_hand = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "hand");
    planning_scene_interface = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
  }

private:
  double loop_rate;
  rclcpp_action::Server<PoseAction>::SharedPtr action_server_move_to_pose;
  rclcpp::TimerBase::SharedPtr main_timer;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_hand;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface;
  rclcpp::Service<CreateBox>::SharedPtr create_box_service;

  rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_grasping_client;

  void timer_callback()
  {
    int something = 0;
  }

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const PoseAction::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request");
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandlePoseAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received cancel request");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandlePoseAction> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&Sorting::move_to_pose, this, _1), goal_handle}.detach();
  }

  void move_to_pose(const std::shared_ptr<GoalHandlePoseAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<PoseAction::Feedback>();
    auto result = std::make_shared<PoseAction::Result>();

    // Use goal's target pose
    geometry_msgs::msg::Pose target_pose = goal->goal_pose;

    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(5.0);
    move_group->setMaxVelocityScalingFactor(goal->vel_factor);
    move_group->setMaxAccelerationScalingFactor(goal->accel_factor);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);

      // Return final pose in result
      result->success = true;
      result->true_pose = move_group->getCurrentPose().pose;
      goal_handle->succeed(result);

      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
      result->success = false;
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void create_box_callback(
    const std::shared_ptr<CreateBox::Request> request,
    std::shared_ptr<CreateBox::Response> response)
  {
    // Create collision object
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = move_group->getPlanningFrame();
    collision_object.id = request->box_id.data;

    // Define the box's shape and pose
    shape_msgs::msg::SolidPrimitive box_primitive;
    box_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    box_primitive.dimensions = {request->size.x, request->size.y, request->size.z};

    // Add the shape and pose to the collision object
    collision_object.primitives.push_back(box_primitive);
    collision_object.primitive_poses.push_back(request->pose);
    collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Apply the collision object to the planning scene
    planning_scene_interface->applyCollisionObject(collision_object);

    // Send success response
    RCLCPP_INFO(this->get_logger(), "Created box collision object at pose (%.2f, %.2f, %.2f)", 
                request->pose.position.x, request->pose.position.y, request->pose.position.z);
    response->success = true;
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
