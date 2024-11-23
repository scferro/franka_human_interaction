#include <chrono>
#include <memory>
#include <random>
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
#include <std_msgs/msg/int8.hpp>
#include <std_srvs/srv/empty.hpp>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "franka_hri_interfaces/action/pose_action.hpp"
#include "franka_hri_interfaces/action/empty_action.hpp"
#include "franka_hri_interfaces/srv/create_box.hpp"
#include "franka_hri_interfaces/srv/update_markers.hpp"
#include "franka_hri_interfaces/srv/sort_net.hpp"

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

using namespace std::chrono_literals;

class ManipulateBlocks : public rclcpp::Node
{
public:
  ManipulateBlocks()
  : Node("manipulate_blocks")
  {
    declare_parameter("rate", 10.);
    loop_rate = get_parameter("rate").as_double();

    action_server_sort_blocks = rclcpp_action::create_server<franka_hri_interfaces::action::EmptyAction>(
      this,
      "sort_blocks",
      std::bind(&ManipulateBlocks::handle_goal, this, std::placeholders::_1),
      std::bind(&ManipulateBlocks::handle_cancel, this, std::placeholders::_1),
      std::bind(&ManipulateBlocks::handle_accepted, this, std::placeholders::_1)
    );

    pretrain_franka_srv = this->create_service<std_srvs::srv::Empty>("pretrain_franka",
            std::bind(&ManipulateBlocks::pretrain_franka_callback, this, std::placeholders::_1, std::placeholders::_2));

    scan_overhead_cli = this->create_client<franka_hri_interfaces::srv::UpdateMarkers>("scan_overhead");
    update_markers_cli = this->create_client<franka_hri_interfaces::srv::UpdateMarkers>("update_markers");
    block_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("blocks", 10);

    block_markers_sub = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        "blocks", 10, std::bind(&ManipulateBlocks::marker_array_callback, this, std::placeholders::_1));

    human_sorting_sub = this->create_subscription<std_msgs::msg::Int8>(
        "human_sorting",
        10,
        std::bind(&ManipulateBlocks::human_sorting_callback, this, std::placeholders::_1)
    );

    train_network_client = create_client<franka_hri_interfaces::srv::SortNet>("train_network");
    get_network_prediction_client = create_client<franka_hri_interfaces::srv::SortNet>("get_network_prediction");
    pretrain_network_client = create_client<std_srvs::srv::Empty>("pretrain_network");

    gripper_homing_client = rclcpp_action::create_client<franka_msgs::action::Homing>(this, "panda_gripper/homing");
    gripper_grasping_client = rclcpp_action::create_client<franka_msgs::action::Grasp>(this, "panda_gripper/grasp");
    
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    block_markers = visualization_msgs::msg::MarkerArray();

    pile_0_start.position.x = 0.6;
    pile_0_start.position.y = 0.35;
    pile_0_start.position.z = 0.0;
    pile_0_start.orientation.x = 1.0;
    pile_0_start.orientation.y = 0.0;
    pile_0_start.orientation.z = 0.0;
    pile_0_start.orientation.w = 0.0;

    pile_1_start = pile_0_start;
    pile_1_start.position.y = -pile_0_start.position.y;

    human_sort_input = -1;
  }

  void initialize()
  {
    planning_scene_interface = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
    move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "panda_manipulator");

    double box_width = 2.;
    double box_length = box_width;
    double box_height = 0.3;

    std_msgs::msg::String box_id;
    box_id.data = "table";

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = box_width / 2 + 0.1;
    target_pose.position.y = 0.0;
    target_pose.position.z = -(box_height / 2) - 0.001;
    target_pose.orientation.x = 0.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 1.0;

    geometry_msgs::msg::Vector3 box_size;
    box_size.x = box_width;
    box_size.y = box_length;
    box_size.z = box_height;

    create_collision_box(target_pose, box_size, box_id);
  }

private:
  double loop_rate;
  visualization_msgs::msg::MarkerArray block_markers;
  geometry_msgs::msg::Pose pile_0_start, pile_1_start;
  std::vector<int> pile_0_index, pile_1_index, sorted_index;
  int human_sort_input;

  rclcpp_action::Server<franka_hri_interfaces::action::EmptyAction>::SharedPtr action_server_sort_blocks;
  rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_grasping_client;
  rclcpp_action::Client<franka_msgs::action::Homing>::SharedPtr gripper_homing_client;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr block_markers_sub;
  rclcpp::TimerBase::SharedPtr main_timer;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface;
  rclcpp::Client<franka_hri_interfaces::srv::UpdateMarkers>::SharedPtr scan_overhead_cli;
  rclcpp::Client<franka_hri_interfaces::srv::UpdateMarkers>::SharedPtr update_markers_cli;
  rclcpp::Client<franka_hri_interfaces::srv::SortNet>::SharedPtr train_network_client;
  rclcpp::Client<franka_hri_interfaces::srv::SortNet>::SharedPtr get_network_prediction_client;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr pretrain_network_client;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr human_sorting_sub;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr pretrain_franka_srv;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr block_pub;
  std::shared_ptr<tf2_ros::Buffer> tfBuffer;
  std::shared_ptr<tf2_ros::TransformListener> tfListener;
  std::shared_ptr<moveit::core::RobotModel> kinematic_model;
  std::shared_ptr<moveit::core::RobotState> kinematic_state;

  void pretrain_franka_callback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                        std::shared_ptr<std_srvs::srv::Empty::Response> response)
  {
    // Start in ready position 
    std::vector<double> joint_positions = {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
    move_to_joints(joint_positions);

    auto overhead_scan_pose = geometry_msgs::msg::PoseStamped();
    overhead_scan_pose.header.stamp = this->get_clock()->now();
    overhead_scan_pose.header.frame_id = "world";
    overhead_scan_pose.pose.position.x = 0.35;
    overhead_scan_pose.pose.position.y = 0.0;
    overhead_scan_pose.pose.position.z = 0.3;
    overhead_scan_pose.pose.orientation.x = 1.0;
    overhead_scan_pose.pose.orientation.y = 0.0;
    overhead_scan_pose.pose.orientation.z = 0.0;
    overhead_scan_pose.pose.orientation.w = 0.0;

    double planning_time = 20.;
    double vel_factor = 0.3;
    double accel_factor = 0.1;

    move_to_pose(overhead_scan_pose, planning_time, vel_factor, accel_factor);

    auto request_out = std::make_shared<std_srvs::srv::Empty::Request>();
    if (!pretrain_network_client->wait_for_service(1s)) {
        RCLCPP_WARN(this->get_logger(), "Pretrain network service not available");
    } else {
        auto future = pretrain_network_client->async_send_request(request_out);
    }
  }

  void sort_blocks(const std::shared_ptr<rclcpp_action::ServerGoalHandle<franka_hri_interfaces::action::EmptyAction>> goal_handle)
  {
    // Start in ready position 
    std::vector<double> joint_positions = {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
    move_to_joints(joint_positions);

    auto overhead_scan_pose = geometry_msgs::msg::PoseStamped();
    overhead_scan_pose.header.stamp = this->get_clock()->now();
    overhead_scan_pose.header.frame_id = "world";
    overhead_scan_pose.pose.position.x = 0.45;
    overhead_scan_pose.pose.position.y = 0.0;
    overhead_scan_pose.pose.position.z = 0.2;
    overhead_scan_pose.pose.orientation.x = 1.0;
    overhead_scan_pose.pose.orientation.y = 0.0;
    overhead_scan_pose.pose.orientation.z = 0.0;
    overhead_scan_pose.pose.orientation.w = 0.0;

    double planning_time = 20.;
    double vel_factor = 0.3;
    double accel_factor = 0.1;

    move_to_pose(overhead_scan_pose, planning_time, vel_factor, accel_factor);

    auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
    request->input_markers = block_markers;
    request->update_scale = true;
    int count = 0;
    while ((!scan_overhead_cli->wait_for_service(1s)) && (count < 5)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for scan_overhead service");
      count++;
    }
    auto future = scan_overhead_cli->async_send_request(request);
    auto result = future.get();
    block_markers = result->output_markers;

    for (std::vector<visualization_msgs::msg::Marker>::size_type i = 0; i < (block_markers.markers.size()); i++) {
      auto it = std::find(sorted_index.begin(), sorted_index.end(), i);
      bool sorted = false;

      if (it != sorted_index.end()) {
          sorted = true;
      }

      if (!sorted) { 
        bool update_scale = true;
        scan_block(i, update_scale);

        double pred = get_network_prediction(i);
        RCLCPP_INFO(this->get_logger(), "Prediction: %f", pred);
        int stack_id;
        auto wait_pose = overhead_scan_pose;

        grab_block(i);

        human_sort_input = -1;

        if (pred >= 0.5) {
          stack_id = 1;
          wait_pose.pose.position.y = -0.2;
        } else {
          stack_id = 0;
          wait_pose.pose.position.y = 0.2;
        }
        vel_factor = 0.6;
        move_to_pose(wait_pose, planning_time, vel_factor, accel_factor);

        double hi_conf_thresh = 0.15;
        
        if ((pred > hi_conf_thresh) && (pred < (1. - hi_conf_thresh))) {
          auto start_time = std::chrono::steady_clock::now();
          int timeout = 5;
          while (human_sort_input == -1)
          {
              auto current_time = std::chrono::steady_clock::now();
              auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
              
              if (elapsed_time.count() >= timeout) 
              {
                  RCLCPP_INFO(this->get_logger(), "Timeout waiting for input");
                  break;
              }          
          }
          RCLCPP_INFO(this->get_logger(), "Human input: %d", human_sort_input);
          if (human_sort_input == 0) {
            if (stack_id==1) {
              stack_id = 0;
            } else if (stack_id==0) {
              stack_id = 1;
            }
          }
        } else {
          if (pred < 0.5) {
            stack_id = 0;
          } else {
            stack_id = 1;
          }
        }

        place_in_stack(i, stack_id);
        train_network(i, stack_id);
        sorted_index.push_back(i);

        // Move back to ready position 
        std::vector<double> joint_positions = {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
        move_to_joints(joint_positions);  
      }
    }
    move_to_pose(overhead_scan_pose, planning_time, vel_factor, accel_factor);
  }

  void human_sorting_callback(const std_msgs::msg::Int8::SharedPtr msg)
  {
      RCLCPP_INFO(this->get_logger(), "Human sorting input: %d", msg->data);
      human_sort_input = msg->data;
  }

  void train_network(int index, int label)
  {
      auto request = std::make_shared<franka_hri_interfaces::srv::SortNet::Request>();
      request->index = index;
      request->label = label;

      while (!train_network_client->wait_for_service(std::chrono::seconds(1)))
      {
          if (!rclcpp::ok())
          {
              RCLCPP_ERROR(this->get_logger(), "Training service interrupted");
              return;
          }
          RCLCPP_INFO(this->get_logger(), "Waiting for training service");
      }

      auto result_future = train_network_client->async_send_request(request);
  }

  double get_network_prediction(int index)
  {
      auto request = std::make_shared<franka_hri_interfaces::srv::SortNet::Request>();
      request->index = index;

      while (!get_network_prediction_client->wait_for_service(std::chrono::seconds(1)))
      {
          if (!rclcpp::ok())
          {
              RCLCPP_ERROR(this->get_logger(), "Prediction service interrupted");
              return -1;
          }
          RCLCPP_INFO(this->get_logger(), "Waiting for prediction service");
      }

      auto result_future = get_network_prediction_client->async_send_request(request);
      if (rclcpp::spin_until_future_complete(this->shared_from_this(), result_future, std::chrono::seconds(1)) != 
          rclcpp::FutureReturnCode::SUCCESS)
      {
          RCLCPP_ERROR(this->get_logger(), "Failed to get prediction");
          return -1;
      }

      auto result = result_future.get();
      return result->prediction;
  }

  void place_in_stack(int i, int stack_id)
  {
    if (stack_id==0 || stack_id==1) {
      double top_z = pile_0_start.position.z;
      auto place_pose = geometry_msgs::msg::Pose();
      
      int max_blocks = 3;
      if (stack_id==0) {
        if (pile_0_index.size() >= static_cast<std::vector<int>::size_type>(max_blocks))
        {
          pile_0_index = {};
          pile_0_start.position.x += -0.15;
        }
        if (pile_0_index.size() > 0){ 
          auto last_marker_index = pile_0_index.back();
          auto last_marker = block_markers.markers[last_marker_index];
          top_z = last_marker.pose.position.z + (last_marker.scale.z / 2);
        }
        place_pose = pile_0_start;
      } else if (stack_id==1) {
        if (pile_1_index.size() >= static_cast<std::vector<int>::size_type>(max_blocks)){
          pile_1_index = {};
          pile_1_start.position.x += -0.15;
        }
        if (pile_1_index.size() > 0){ 
          auto last_marker_index = pile_1_index.back();
          auto last_marker = block_markers.markers[last_marker_index];
          top_z = last_marker.pose.position.z + (last_marker.scale.z / 2);
        }
        place_pose = pile_1_start;
      }

      double z_add = (block_markers.markers[i].scale.z / 2);
      if (z_add < 0.02) {
        z_add = 0.02;
      }
      place_pose.position.z = top_z + z_add;

      auto hover_pose = geometry_msgs::msg::PoseStamped();
      hover_pose.header.stamp = this->get_clock()->now();
      hover_pose.header.frame_id = "world";
      hover_pose.pose.position.x = place_pose.position.x;
      hover_pose.pose.position.y = place_pose.position.y;
      hover_pose.pose.position.z = place_pose.position.z + 0.15;
      hover_pose.pose.orientation.x = 1.0;
      hover_pose.pose.orientation.y = 0.0;
      hover_pose.pose.orientation.z = 0.0;
      hover_pose.pose.orientation.w = 0.0;
      double planning_time = 10.;
      double vel_factor = 0.4;
      double accel_factor = 0.1;
      move_to_pose(hover_pose, planning_time, vel_factor, accel_factor);

      if (stack_id==0) {
        for (int j = 0; j < pile_0_index.size(); ++j) {
            int id = pile_0_index[j];
            std::string id_string = std::to_string(id);
            auto block_id = std_msgs::msg::String();
            block_id.data = id_string;
            remove_collision_box(block_id);
        }
        pile_0_index.push_back(i);
      } else if (stack_id==1) {
        for (int j = 0; j < pile_1_index.size(); ++j) {
            int id = pile_1_index[j];
            std::string id_string = std::to_string(id);
            auto block_id = std_msgs::msg::String();
            block_id.data = id_string;
            remove_collision_box(block_id);
        }
        pile_1_index.push_back(i);
      }

      place_block(place_pose, i);

      if (stack_id==0) {
        for (int j = 0; j < pile_0_index.size(); ++j) {
            int id = pile_0_index[j];
            std::string id_string = std::to_string(id);
            auto block_id = std_msgs::msg::String();
            block_id.data = id_string;
            create_collision_box(block_markers.markers[id].pose, block_markers.markers[id].scale, block_id);
        }
      } else if (stack_id==1) {
        for (int j = 0; j < pile_1_index.size(); ++j) {
            int id = pile_1_index[j];
            std::string id_string = std::to_string(id);
            auto block_id = std_msgs::msg::String();
            block_id.data = id_string;
            create_collision_box(block_markers.markers[id].pose, block_markers.markers[id].scale, block_id);
        }
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "Invalid stack ID");
    }
  }

  void place_block(geometry_msgs::msg::Pose place_pose, int marker_index)
 {
   double planning_time = 20.;
   double vel_factor = 0.1;
   double accel_factor = 0.1;

   auto drop_pose = geometry_msgs::msg::PoseStamped();
   drop_pose.header.stamp = this->get_clock()->now();
   drop_pose.header.frame_id = "world";
   drop_pose.pose.position.x = place_pose.position.x;
   drop_pose.pose.position.y = place_pose.position.y;
   drop_pose.pose.position.z = place_pose.position.z + 0.03;
   drop_pose.pose.orientation.x = 1.0;
   drop_pose.pose.orientation.y = 0.0;
   drop_pose.pose.orientation.z = 0.0;
   drop_pose.pose.orientation.w = 0.0;

   move_to_pose(drop_pose, planning_time, vel_factor, accel_factor);

   double width = 0.04;
   double speed = 0.2;
   double force = 1.;
   send_grasp_goal(width, speed, force);

   auto marker = block_markers.markers[marker_index];
   marker.pose = place_pose;
   auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
   marker.action = 0;
   request->input_markers.markers.push_back(marker);
   std::vector<int> update = {marker_index};
   request->markers_to_update = update;

   auto retreat_pose = drop_pose;
   retreat_pose.pose.position.z += 0.15;
   move_to_pose(retreat_pose, planning_time, vel_factor, accel_factor);

   int count = 0;
   while ((!update_markers_cli->wait_for_service(1s)) && (count < 5)) {
     RCLCPP_INFO(this->get_logger(), "Waiting for update_markers service");
     count++;
   }
   auto future = update_markers_cli->async_send_request(request);
   auto result = future.get();
   block_markers = result->output_markers;
 }

 void scan_block(int i, bool update_scale)
 {
   std::this_thread::sleep_for(std::chrono::milliseconds(500));

   auto marker = block_markers.markers[i];
   double x_trans = 0.065;

   auto scan_pose = geometry_msgs::msg::PoseStamped();
   scan_pose.header.stamp = this->get_clock()->now();
   scan_pose.header.frame_id = "world";
   scan_pose.pose.position.x = marker.pose.position.x - x_trans;
   scan_pose.pose.position.y = marker.pose.position.y;
   scan_pose.pose.position.z = marker.pose.position.z + 0.15;
   scan_pose.pose.orientation.x = 1.0;
   scan_pose.pose.orientation.y = 0.0;
   scan_pose.pose.orientation.z = 0.0;
   scan_pose.pose.orientation.w = 0.0;

   double planning_time = 20.;
   double vel_factor = 0.4;
   double accel_factor = 0.1;

   move_to_pose(scan_pose, planning_time, vel_factor, accel_factor);

   auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
   request->input_markers = block_markers;
   std::vector<int> update = {i};
   request->markers_to_update = update;
   request->update_scale = update_scale;

   int count = 0;
   while ((!scan_overhead_cli->wait_for_service(1s)) && (count < 5)) {
     RCLCPP_INFO(this->get_logger(), "Waiting for scan_overhead service");
     count++;
   }

   auto future = scan_overhead_cli->async_send_request(request);
   auto result = future.get();
   block_markers = result->output_markers;
 }

 void grab_block(int i)
 {
   auto marker = block_markers.markers[i];
   auto grab_pose_1 = geometry_msgs::msg::PoseStamped();

   grab_pose_1.header.stamp = this->get_clock()->now();
   grab_pose_1.header.frame_id = "world";
   grab_pose_1.pose.position.x = marker.pose.position.x;
   grab_pose_1.pose.position.y = marker.pose.position.y;
   grab_pose_1.pose.position.z = marker.pose.position.z + 0.08;
   grab_pose_1.pose.orientation = marker.pose.orientation;

   double planning_time = 20.;
   double vel_factor = 0.4;
   double accel_factor = 0.1;

   move_to_pose(grab_pose_1, planning_time, vel_factor, accel_factor);

   auto grab_pose_2 = grab_pose_1;
   grab_pose_2.header.stamp = this->get_clock()->now();
   grab_pose_2.pose.position.z = marker.pose.position.z + 0.03;
   if (grab_pose_2.pose.position.z < 0.05) {
     grab_pose_2.pose.position.z = 0.06;
   }
   move_to_pose(grab_pose_2, planning_time, vel_factor, accel_factor);

   int count = 0;
   while ((!update_markers_cli->wait_for_service(1s)) && (count < 5)) {
     RCLCPP_INFO(this->get_logger(), "Waiting for update_markers service");
     count++;
   }

   auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
   marker.action = 2;
   request->input_markers.markers.push_back(marker);
   std::vector<int> update = {i};
   request->markers_to_update = update;

   auto future = update_markers_cli->async_send_request(request);
   auto result = future.get();
   block_markers = result->output_markers;

   double width = 0.02;
   double speed = 0.2;
   double force = 0.001;
   send_grasp_goal(width, speed, force);

   auto retreat_pose = grab_pose_1;
   retreat_pose.header.stamp = this->get_clock()->now();
   retreat_pose.pose.position.z = 0.2;
   retreat_pose.pose.orientation = marker.pose.orientation;
   move_to_pose(retreat_pose, planning_time, vel_factor, accel_factor);
 }

 void move_to_pose(const geometry_msgs::msg::PoseStamped target_pose, const double planning_time,
   const double vel_factor, const double accel_factor)
 {
   RCLCPP_INFO(this->get_logger(), "Planning movement");

   move_group->setPoseTarget(target_pose);
   move_group->setPlanningTime(planning_time);
   move_group->setMaxVelocityScalingFactor(vel_factor);
   move_group->setMaxAccelerationScalingFactor(accel_factor);

   moveit::planning_interface::MoveGroupInterface::Plan plan;
   bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

   if (success) {
     RCLCPP_INFO(this->get_logger(), "Executing movement");
     move_group->execute(plan);
   } else {
     RCLCPP_ERROR(this->get_logger(), "Planning failed");
   }
 }

void move_to_joints(const std::vector<double>& joint_positions)
{
    RCLCPP_INFO(this->get_logger(), "Planning joint movement");

    // Set the joint value target
    move_group->setJointValueTarget(joint_positions);

    // Plan and execute
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
        RCLCPP_INFO(this->get_logger(), "Executing joint movement");
        move_group->execute(plan);
        RCLCPP_INFO(this->get_logger(), "Joint movement complete");
    } else {
        RCLCPP_ERROR(this->get_logger(), "Joint movement planning failed");
    }
}

 void create_collision_box(
   const geometry_msgs::msg::Pose target_pose, const geometry_msgs::msg::Vector3 size,
   const std_msgs::msg::String box_id)
 {
   moveit_msgs::msg::CollisionObject collision_object;
   collision_object.header.frame_id = move_group->getPlanningFrame();;
   collision_object.id = box_id.data;

   shape_msgs::msg::SolidPrimitive box_primitive;
   box_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
   box_primitive.dimensions = {size.x, size.y, size.z};

   collision_object.primitives.push_back(box_primitive);
   collision_object.primitive_poses.push_back(target_pose);
   collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;

   planning_scene_interface->applyCollisionObject(collision_object);
   RCLCPP_INFO(this->get_logger(), "Created collision box");
 }

 void remove_collision_box(
   const std_msgs::msg::String box_id)
 {
   moveit_msgs::msg::CollisionObject collision_object;
   collision_object.header.frame_id = move_group->getPlanningFrame();;
   collision_object.id = box_id.data;

   shape_msgs::msg::SolidPrimitive box_primitive;
   box_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;

   collision_object.primitives.push_back(box_primitive);
   collision_object.operation = moveit_msgs::msg::CollisionObject::REMOVE;

   planning_scene_interface->applyCollisionObject(collision_object);
   RCLCPP_INFO(this->get_logger(), "Removed collision box");
 }

 void attach_collision_object_to_gripper(const geometry_msgs::msg::Vector3& size)
 {
     moveit_msgs::msg::CollisionObject collision_object;
     collision_object.header.frame_id = move_group->getEndEffectorLink();
     collision_object.id = "Attached";

     shape_msgs::msg::SolidPrimitive object_primitive;
     object_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
     object_primitive.dimensions = {size.x, size.y, size.z};

     auto target_pose = geometry_msgs::msg::Pose();
     target_pose.position.x = 0.;
     target_pose.position.y = 0.;
     target_pose.position.z = 0.02;
     target_pose.orientation.x = 0.0;
     target_pose.orientation.y = 0.0;
     target_pose.orientation.z = 0.0;
     target_pose.orientation.w = 1.0;

     collision_object.primitives.push_back(object_primitive);
     collision_object.primitive_poses.push_back(target_pose);
     collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;

     moveit_msgs::msg::AttachedCollisionObject attached_object;
     attached_object.link_name = move_group->getEndEffectorLink();
     attached_object.object = collision_object;

     planning_scene_interface->applyAttachedCollisionObject(attached_object);
     RCLCPP_INFO(this->get_logger(), "Attached collision object");
 }

 void detach_collision_object_from_gripper()
 {
     moveit_msgs::msg::AttachedCollisionObject detach_object;
     detach_object.link_name = move_group->getEndEffectorLink();
     detach_object.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
     detach_object.object.id = "Attached";

     planning_scene_interface->applyAttachedCollisionObject(detach_object);

     moveit_msgs::msg::CollisionObject remove_object;
     remove_object.id = "Attached";
     remove_object.operation = moveit_msgs::msg::CollisionObject::REMOVE;

     planning_scene_interface->applyCollisionObject(remove_object);
     RCLCPP_INFO(this->get_logger(), "Detached collision object");
 }

 void set_joint_limits()
 {
     moveit::core::JointModel* panda_joint2 = kinematic_model->getJointModel("panda_joint2");

     moveit::core::VariableBounds bounds;
     bounds.position_bounded_ = true;
     bounds.min_position_ = -1.25;
     bounds.max_position_ = 1.25;

     panda_joint2->setVariableBounds(panda_joint2->getName(), bounds);
     kinematic_state->enforceBounds();
     move_group->setJointValueTarget(*kinematic_state);
 }

 void send_homing_goal()
 {
     if (!gripper_homing_client->wait_for_action_server(std::chrono::seconds(1))) {
         RCLCPP_ERROR(this->get_logger(), "Homing action server not available");
         return;
     }

     auto homing_goal = franka_msgs::action::Homing::Goal();
     auto homing_future = gripper_homing_client->async_send_goal(homing_goal);
     auto homing_result = homing_future.get();
 }

 void send_grasp_goal(double width, double speed, double force)
 {
   if (!gripper_grasping_client->wait_for_action_server(std::chrono::seconds(1))) {
       RCLCPP_ERROR(this->get_logger(), "Grasp action server not available");
       return;
   }

   auto grasp_goal = franka_msgs::action::Grasp::Goal();
   grasp_goal.width = width;
   grasp_goal.speed = speed;
   grasp_goal.force = force;
   grasp_goal.epsilon.inner = 0.2;
   grasp_goal.epsilon.outer = 0.2;

   auto grasp_future = gripper_grasping_client->async_send_goal(grasp_goal);

   auto result_future = gripper_grasping_client->async_get_result(grasp_future.get());
   RCLCPP_INFO(this->get_logger(), "Moving gripper");

   if (rclcpp::spin_until_future_complete(this->shared_from_this(), result_future, std::chrono::seconds(1)) !=
       rclcpp::FutureReturnCode::SUCCESS)
   {
       return;
   }

   auto result = result_future.get();
   if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
       RCLCPP_INFO(this->get_logger(), "Grasp succeeded");
   } else {
       RCLCPP_INFO(this->get_logger(), "Grasp failed");
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