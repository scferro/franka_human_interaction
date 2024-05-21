#include <chrono>
#include <memory>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <rclcpp_action/rclcpp_action.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/homing.hpp>

#include "franka_hri_interfaces/action/pose_action.hpp"
#include "franka_hri_interfaces/action/empty_action.hpp"
#include "franka_hri_interfaces/srv/create_box.hpp"
#include "franka_hri_interfaces/srv/update_markers.hpp"

using namespace std::chrono_literals;

class ManipulateBlocks : public rclcpp::Node
{
public:
  ManipulateBlocks()
  : Node("manipulate_blocks")
  {
    declare_parameter("rate", 10.);
    loop_rate = get_parameter("rate").as_double();

    // Create an action server for the sort_blocks service
    action_server_sort_blocks = rclcpp_action::create_server<franka_hri_interfaces::action::EmptyAction>(
      this,
      "sort_blocks",
      std::bind(&ManipulateBlocks::handle_goal, this, std::placeholders::_1),
      std::bind(&ManipulateBlocks::handle_cancel, this, std::placeholders::_1),
      std::bind(&ManipulateBlocks::handle_accepted, this, std::placeholders::_1)
    );

    // Create a client for the overhead scanning service
    scan_overhead_cli = this->create_client<franka_hri_interfaces::srv::UpdateMarkers>("scan_overhead");

    // Create a publisher for block markers
    block_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("blocks", 10);

    // Create action clients for the gripper
    gripper_homing_client = rclcpp_action::create_client<franka_msgs::action::Homing>(this, "panda_gripper/homing");
    gripper_grasping_client = rclcpp_action::create_client<franka_msgs::action::Grasp>(this, "panda_gripper/grasp");

    block_markers = visualization_msgs::msg::MarkerArray();

    // Initialize the locations to make stacks
    // Set the box pose
    pile_0_start.position.x = 0.5;
    pile_0_start.position.y = 0.25;
    pile_0_start.position.z = 0.0;
    pile_0_start.orientation.x = 1.0;
    pile_0_start.orientation.y = 0.0;
    pile_0_start.orientation.z = 0.0;
    pile_0_start.orientation.w = 0.0;

    pile_1_start = pile_0_start;
    pile_1_start.position.y = -pile_0_start.position.y;
  }

  void initialize()
  {
    // Now safe to use shared_from_this() because the object is fully constructed
    planning_scene_interface = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
    move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "panda_manipulator");

    // Create collision object to represent table
    // Define the box dimensions
    double box_width = 2.;
    double box_length = box_width;
    double box_height = 0.3;

    // Create the box ID
    std_msgs::msg::String box_id;
    box_id.data = "table";

    // Set the box pose
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = box_width / 2 + 0.1;
    target_pose.position.y = 0.0;
    target_pose.position.z = -(box_height / 2) - 0.001;
    target_pose.orientation.x = 0.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 1.0;

    // Set the box size
    geometry_msgs::msg::Vector3 box_size;
    box_size.x = box_width;
    box_size.y = box_length;
    box_size.z = box_height;

    // Call the create_collision_box function
    // create_collision_box(target_pose, box_size, box_id);

    // Create a subscriber for the blocks
    block_markers_sub = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        "blocks", 10, std::bind(&ManipulateBlocks::marker_array_callback, this, std::placeholders::_1));
  }

private:
  double loop_rate;
  visualization_msgs::msg::MarkerArray block_markers;
  geometry_msgs::msg::Pose pile_0_start, pile_1_start;
  std::vector<int> pile_0_index, pile_1_index;

  rclcpp_action::Server<franka_hri_interfaces::action::EmptyAction>::SharedPtr action_server_sort_blocks;
  rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_grasping_client;
  rclcpp_action::Client<franka_msgs::action::Homing>::SharedPtr gripper_homing_client;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr block_markers_sub;
  rclcpp::TimerBase::SharedPtr main_timer;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface;
  rclcpp::Client<franka_hri_interfaces::srv::UpdateMarkers>::SharedPtr scan_overhead_cli;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr block_pub;

  void sort_blocks(const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
  {
    auto overhead_scan_pose = geometry_msgs::msg::PoseStamped();

    // Fill out the header
    overhead_scan_pose.header.stamp = this->get_clock()->now();
    overhead_scan_pose.header.frame_id = "world";

    // Fill out the pose
    overhead_scan_pose.pose.position.x = 0.3;
    overhead_scan_pose.pose.position.y = 0.0;
    overhead_scan_pose.pose.position.z = 0.25;

    overhead_scan_pose.pose.orientation.x = 1.0;
    overhead_scan_pose.pose.orientation.y = 0.0;
    overhead_scan_pose.pose.orientation.z = 0.0;
    overhead_scan_pose.pose.orientation.w = 0.0;

    double planning_time = 10.;
    double vel_factor = 0.8;
    double accel_factor = 0.2;

    move_to_pose(overhead_scan_pose, planning_time, vel_factor, accel_factor);

    auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
    request->input_markers = block_markers;

    int count = 0;
    while ((!scan_overhead_cli->wait_for_service(1s)) && (count < 5)) {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Waiting for scan_overhead service...");
      count++;
    }

    auto future = scan_overhead_cli->async_send_request(request);
    auto result = future.get();
    block_markers = result->output_markers;

    for (int i = 0; i < (block_markers.markers.size()); i++) {
      // pick up block
      grab_block(i);

      // Calculate top of stack
      double top_z = 0.0;
      for (int j = 0; j < (pile_0_index.size()); j++) {
        auto block = block_markers.markers[j];
        top_z += block.scale.z;
      }

      // Calculate place height
      double z_place = top_z + (block_markers.markers[i].scale.z / 2);
      RCLCPP_INFO(this->get_logger(), "Z-Place: %s", std::to_string(z_place).c_str());

      // Move above top of stack
      auto hover_pose = overhead_scan_pose;
      hover_pose.header.stamp = this->get_clock()->now();
      hover_pose.pose.position.x = pile_0_start.position.x;
      hover_pose.pose.position.y = pile_0_start.position.y;
      hover_pose.pose.position.z = z_place + 0.1;
      move_to_pose(hover_pose, planning_time, vel_factor, accel_factor);

      // Move down to drop brick
      auto drop_pose = hover_pose;
      drop_pose.pose.position.z = z_place + 0.03;
      move_to_pose(drop_pose, planning_time, vel_factor, accel_factor);

      // Open gripper
      double width = 0.04;
      double speed = 0.2;
      double force = 1.;
      send_grasp_goal(width, speed, force);

      // Add block index to pile vector
      pile_0_index.push_back(i);

      // Move back to hover pose
      move_to_pose(hover_pose, planning_time, vel_factor, accel_factor);
    }
  }

  void grab_block(int i)
  {
    // Get current marker
    auto marker = block_markers.markers[i];

    // Move above the block
    auto grab_pose_1 = geometry_msgs::msg::PoseStamped();

    // Fill out the header
    grab_pose_1.header.stamp = this->get_clock()->now();
    grab_pose_1.header.frame_id = "world";

    // Fill out the pose
    grab_pose_1.pose.position.x = marker.pose.position.x;
    grab_pose_1.pose.position.y = marker.pose.position.y;
    grab_pose_1.pose.position.z = 0.15;

    // Set the orientation of the grab pose
    grab_pose_1.pose.orientation.x = 1.0;
    grab_pose_1.pose.orientation.y = 0.0;
    grab_pose_1.pose.orientation.z = 0.0;
    grab_pose_1.pose.orientation.w = 0.0;

    double planning_time = 20.;
    double vel_factor = 0.4;
    double accel_factor = 0.2;

    move_to_pose(grab_pose_1, planning_time, vel_factor, accel_factor);

    // Perform a refinement scan
    auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
    request->input_markers = block_markers;
    std::vector<int> update = {i};
    request->markers_to_update = update;

    int count = 0;
    while ((!scan_overhead_cli->wait_for_service(1s)) && (count < 5)) {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Waiting for scan_overhead service...");
      count++;
    }

    auto future = scan_overhead_cli->async_send_request(request);
    auto result = future.get();
    block_markers = result->output_markers;

    // Rotate gripper above the block
    auto grab_pose_2 = grab_pose_1;

    // Set the orientation of the grab pose
    grab_pose_2.pose.position.z = 0.12;
    grab_pose_2.pose.orientation = marker.pose.orientation;

    move_to_pose(grab_pose_2, planning_time, vel_factor, accel_factor);

    // Grab the block
    auto grab_pose_3 = grab_pose_2;

    // Fill out the changes to the pose and move to pose
    grab_pose_3.header.stamp = this->get_clock()->now();
    grab_pose_3.pose.position.z = marker.pose.position.z + 0.03;
    move_to_pose(grab_pose_3, planning_time, vel_factor, accel_factor);

    // Close gripper
    double width = 0.02;
    double speed = 0.2;
    double force = 0.001;
    send_grasp_goal(width, speed, force);

    // Retreat after grabbing block
    auto retreat_pose = grab_pose_1;

    // Fill out the changes to the pose and move to pose
    retreat_pose.header.stamp = this->get_clock()->now();
    retreat_pose.pose.position.z = 0.2;
    retreat_pose.pose.orientation = marker.pose.orientation;
    move_to_pose(retreat_pose, planning_time, vel_factor, accel_factor);
  }

  void move_to_pose(const geometry_msgs::msg::PoseStamped target_pose, const double planning_time,
    const double vel_factor, const double accel_factor)
  {
    RCLCPP_INFO(this->get_logger(), "Planning movement to goal...");

    // Use goal's target pose
    move_group->setPoseTarget(target_pose);

    // Set the planning time (in seconds)
    move_group->setPlanningTime(planning_time);
    move_group->setMaxVelocityScalingFactor(vel_factor);
    move_group->setMaxAccelerationScalingFactor(accel_factor);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      RCLCPP_INFO(this->get_logger(), "Planning succeeded! Moving to goal.");
      move_group->execute(plan);
      RCLCPP_INFO(this->get_logger(), "Moved to goal pose.");
    } else {
      RCLCPP_INFO(this->get_logger(), "Planning failed");
    }
  }

  void create_collision_box(
    const geometry_msgs::msg::Pose target_pose, const geometry_msgs::msg::Vector3 size,
    std_msgs::msg::String box_id)
  {
    // Create collision object
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = move_group->getPlanningFrame();;
    collision_object.id = box_id.data;

    // Define the box's shape and pose
    shape_msgs::msg::SolidPrimitive box_primitive;
    box_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    box_primitive.dimensions = {size.x, size.y, size.z};

    // Add the shape and pose to the collision object
    collision_object.primitives.push_back(box_primitive);
    collision_object.primitive_poses.push_back(target_pose);
    collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Apply the collision object to the planning scene
    planning_scene_interface->applyCollisionObject(collision_object);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Created collision box.");
  }

  void send_homing_goal()
  {
      if (!gripper_homing_client->wait_for_action_server(std::chrono::seconds(1))) {
          RCLCPP_ERROR(get_logger(), "Homing action server not available");
          return;
      }

      auto homing_goal = franka_msgs::action::Homing::Goal();
      auto homing_future = gripper_homing_client->async_send_goal(homing_goal);

      // if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), homing_future) !=
      //     rclcpp::FutureReturnCode::SUCCESS) {
      //     RCLCPP_ERROR(get_logger(), "Failed to send homing goal");
      //     return;
      // }

      auto homing_result = homing_future.get();
  }

  void send_grasp_goal(double width, double speed, double force)
  {
    if (!gripper_grasping_client->wait_for_action_server(std::chrono::seconds(1))) {
        RCLCPP_ERROR(get_logger(), "Grasp action server not available");
        return;
    }

    auto grasp_goal = franka_msgs::action::Grasp::Goal();
    grasp_goal.width = width;
    grasp_goal.speed = speed;
    grasp_goal.force = force;
    grasp_goal.epsilon.inner = 0.1;
    grasp_goal.epsilon.outer = 0.1;

    auto grasp_future = gripper_grasping_client->async_send_goal(grasp_goal);

    // Create a new node for spinning
    auto spin_node = std::make_shared<rclcpp::Node>("spin_node");

    // Wait for the result
    auto result_future = gripper_grasping_client->async_get_result(grasp_future.get());
    RCLCPP_INFO(get_logger(), "Grasping!");

    auto timeout = std::chrono::seconds(1); // Set the desired timeout duration
    if (rclcpp::spin_until_future_complete(spin_node, result_future, timeout) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
        return;
    }

    // Check the result
    auto result = result_future.get();
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO(get_logger(), "Grasp succeeded.");
    } else {
        RCLCPP_INFO(get_logger(), "Grasp failed.");
    }
  }

  void marker_array_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
      block_markers = *msg;
  }

  void timer_callback()
  {
    int something = 0;
  }

  rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID & uuid)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request");
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received cancel request");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{std::bind(&ManipulateBlocks::sort_blocks, this, _1), goal_handle}.detach();
  }

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ManipulateBlocks>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
