#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <future>
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
#include <std_msgs/msg/bool.hpp>
#include <rclcpp/callback_group.hpp>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "franka_hri_interfaces/action/pose_action.hpp"
#include "franka_hri_interfaces/action/empty_action.hpp"
#include "franka_hri_interfaces/msg/block_placement_info.hpp"
#include "franka_hri_interfaces/srv/create_box.hpp"
#include "franka_hri_interfaces/srv/update_markers.hpp"
#include "franka_hri_interfaces/srv/sort_net.hpp"
#include "franka_hri_interfaces/srv/gest_net.hpp"
#include "franka_hri_interfaces/srv/move_block.hpp"

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
    declare_parameter("scan_position_x", 0.5);
    loop_rate = get_parameter("rate").as_double();
    scan_position_x = get_parameter("scan_position_x").as_double();
        
    markers_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    prediction_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);

    action_server_sort_blocks = rclcpp_action::create_server<franka_hri_interfaces::action::EmptyAction>(
      this,
      "sort_blocks",
      std::bind(&ManipulateBlocks::handle_goal, this, std::placeholders::_1),
      std::bind(&ManipulateBlocks::handle_cancel, this, std::placeholders::_1),
      std::bind(&ManipulateBlocks::handle_accepted, this, std::placeholders::_1)
    );

    pretrain_franka_srv = this->create_service<std_srvs::srv::Empty>(
            "pretrain_franka",
            std::bind(&ManipulateBlocks::pretrain_franka_callback, this, std::placeholders::_1, std::placeholders::_2));

    update_piles_service = this->create_service<franka_hri_interfaces::srv::MoveBlock>(
        "update_piles",
        std::bind(&ManipulateBlocks::update_piles_callback, this, 
                  std::placeholders::_1, std::placeholders::_2));

    scan_overhead_cli = this->create_client<franka_hri_interfaces::srv::UpdateMarkers>("scan_overhead");
    update_markers_cli = this->create_client<franka_hri_interfaces::srv::UpdateMarkers>(
        "update_markers", 
        rmw_qos_profile_services_default,
        markers_callback_group_
    );

    block_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("blocks", 10);

    block_placement_pub = this->create_publisher<franka_hri_interfaces::msg::BlockPlacementInfo>(
        "block_placement_info", 10);  

    block_markers_sub = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        "blocks", 10, std::bind(&ManipulateBlocks::marker_array_callback, this, std::placeholders::_1));

    human_sorting_sub = this->create_subscription<std_msgs::msg::Int8>(
        "human_sorting",
        10,
        std::bind(&ManipulateBlocks::human_sorting_callback, this, std::placeholders::_1)
    );

    train_network_client = create_client<franka_hri_interfaces::srv::SortNet>("train_network");
    get_network_prediction_client = this->create_client<franka_hri_interfaces::srv::SortNet>(
        "get_network_prediction",
        rmw_qos_profile_services_default,
        prediction_callback_group_
    );

    // Complex gesture clients
    get_complex_gesture_prediction_client = this->create_client<franka_hri_interfaces::srv::GestNet>(
        "get_complex_gesture_prediction",
        rmw_qos_profile_services_default,
        prediction_callback_group_
    );

    train_complex_gesture_client = this->create_client<franka_hri_interfaces::srv::GestNet>(
        "train_complex_gesture",
        rmw_qos_profile_services_default,
        prediction_callback_group_
    );

    pretrain_network_client = create_client<std_srvs::srv::Empty>("pretrain_network");

    home_joint_positions = {0.0, -0.132, 0.002, -2.244, 0.004, 2.111, 0.785};

    // Add hands_detected subscriber
    hands_detected_sub = this->create_subscription<std_msgs::msg::Bool>(
      "hands_detected",
      10,
      std::bind(&ManipulateBlocks::hands_detected_callback, this, std::placeholders::_1)
    );

    gripper_homing_client = rclcpp_action::create_client<franka_msgs::action::Homing>(this, "panda_gripper/homing");
    gripper_grasping_client = rclcpp_action::create_client<franka_msgs::action::Grasp>(this, "panda_gripper/grasp");
    
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    block_markers = visualization_msgs::msg::MarkerArray();

    // Initialize pile positions for 4 categories
    pile_0_start.position.x = scan_position_x - 0.2;
    pile_0_start.position.y = -0.4;
    pile_0_start.position.z = 0.0;
    pile_0_start.orientation.x = 1.0;
    pile_0_start.orientation.y = 0.0;
    pile_0_start.orientation.z = 0.0;
    pile_0_start.orientation.w = 0.0;

    pile_1_start = pile_0_start;
    pile_1_start.position.y = 0.4;

    pile_2_start = pile_0_start;
    pile_2_start.position.x = scan_position_x + 0.2;
    pile_2_start.position.y = -0.35;

    pile_3_start = pile_2_start;
    pile_3_start.position.y = 0.35;

    human_sort_input = -1;
  }

  void initialize()
  {
    planning_scene_interface = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
    move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "panda_manipulator");

    double box_width = 3.;
    double box_height = 0.3;

    std_msgs::msg::String box_id;
    box_id.data = "table_1";

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = (box_width / 2 + 0.1) / 2;
    target_pose.position.y = 0.0;
    target_pose.position.z = -(box_height / 2);
    target_pose.orientation.x = 0.0;
    target_pose.orientation.y = 0.0;
    target_pose.orientation.z = 0.0;
    target_pose.orientation.w = 1.0;

    geometry_msgs::msg::Vector3 box_size;
    box_size.x = box_width / 2 - 0.1;
    box_size.y = box_width;
    box_size.z = box_height;

    create_collision_box(target_pose, box_size, box_id);

    target_pose.position.x = -target_pose.position.x;
    box_id.data = "table_2";
    create_collision_box(target_pose, box_size, box_id);

    target_pose.position.x = 0.0;
    target_pose.position.y = (box_width / 2 + 0.1) / 2;
    box_size.y = box_width / 2 - 0.1;
    box_size.x = box_width;
    box_id.data = "table_3";
    create_collision_box(target_pose, box_size, box_id);

    target_pose.position.y = -target_pose.position.y;
    box_id.data = "table_4";
    create_collision_box(target_pose, box_size, box_id);
  }

private:
  double loop_rate;
  double scan_position_x;
  visualization_msgs::msg::MarkerArray block_markers;
  geometry_msgs::msg::Pose pile_0_start, pile_1_start, pile_2_start, pile_3_start;
  std::vector<int> pile_0_index, pile_1_index, pile_2_index, pile_3_index, sorted_index;
  std::vector<double> home_joint_positions;
  int human_sort_input;
  bool hands_detected;
  bool waiting_for_human;

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
  rclcpp::CallbackGroup::SharedPtr markers_callback_group_;
  rclcpp::CallbackGroup::SharedPtr prediction_callback_group_;
  rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedPtr train_gesture_client;
  rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedPtr train_complex_gesture_client;
  rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedPtr get_gesture_prediction_client;
  rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedPtr get_complex_gesture_prediction_client;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr hands_detected_sub;
  rclcpp::Publisher<franka_hri_interfaces::msg::BlockPlacementInfo>::SharedPtr block_placement_pub;
  rclcpp::Service<franka_hri_interfaces::srv::MoveBlock>::SharedPtr update_piles_service;
  std::shared_ptr<tf2_ros::Buffer> tfBuffer;
  std::shared_ptr<tf2_ros::TransformListener> tfListener;
  std::shared_ptr<moveit::core::RobotModel> kinematic_model;
  std::shared_ptr<moveit::core::RobotState> kinematic_state;

  void pretrain_franka_callback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                        std::shared_ptr<std_srvs::srv::Empty::Response> response)
  {
    move_to_joints(home_joint_positions);

    auto overhead_scan_pose = geometry_msgs::msg::PoseStamped();
    overhead_scan_pose.header.stamp = this->get_clock()->now();
    overhead_scan_pose.header.frame_id = "world";
    overhead_scan_pose.pose.position.x = scan_position_x;
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
      move_to_joints(home_joint_positions);

      auto overhead_scan_pose = geometry_msgs::msg::PoseStamped();
      overhead_scan_pose.header.stamp = this->get_clock()->now();
      overhead_scan_pose.header.frame_id = "world";
      overhead_scan_pose.pose.position.x = scan_position_x;
      overhead_scan_pose.pose.position.y = 0.0;
      overhead_scan_pose.pose.position.z = 0.25;
      overhead_scan_pose.pose.orientation.x = 1.0;
      overhead_scan_pose.pose.orientation.y = 0.0;
      overhead_scan_pose.pose.orientation.z = 0.0;
      overhead_scan_pose.pose.orientation.w = 0.0;

      double planning_time = 20.;
      double vel_factor = 0.3;
      double accel_factor = 0.1;

      move_to_pose(overhead_scan_pose, planning_time, vel_factor, accel_factor);

      // Create promise/future pair for initial scan
      auto scan_response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::UpdateMarkers::Response::SharedPtr>>();
      auto scan_future_result = scan_response_received->get_future();

      auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
      request->input_markers = block_markers;
      request->update_scale = true;

      // Create callback for initial scan
      auto scan_callback = [this, scan_response_received](
          rclcpp::Client<franka_hri_interfaces::srv::UpdateMarkers>::SharedFuture future) {
          try {
              auto result = future.get();
              this->block_markers = result->output_markers;
              scan_response_received->set_value(result);
              RCLCPP_INFO(this->get_logger(), "Initial scan complete");
          } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Error in scan callback: %s", e.what());
          }
      };

      int count = 0;
      while ((!scan_overhead_cli->wait_for_service(1s)) && (count < 10)) {
          RCLCPP_INFO(this->get_logger(), "Waiting for scan_overhead service");
          count++;
      }

      scan_overhead_cli->async_send_request(request, scan_callback);

      // Wait for initial scan with timeout
      if (scan_future_result.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
          RCLCPP_ERROR(this->get_logger(), "Initial scan failed - timeout");
          return;
      }

    for (std::vector<visualization_msgs::msg::Marker>::size_type i = 0; i < (block_markers.markers.size()); i++) {
        if (std::find(sorted_index.begin(), sorted_index.end(), i) == sorted_index.end()) {
            scan_block(i, true);
            grab_block(i);

            double sort_pred = get_network_prediction(i);
            RCLCPP_INFO(this->get_logger(), "Sort prediction: %f", sort_pred);
            
            const double CONFIDENCE_THRESHOLD = 0.8;
            int predicted_category = static_cast<int>(sort_pred);
            bool high_confidence = (sort_pred - predicted_category) > CONFIDENCE_THRESHOLD;
            int final_category;
            double simple_gesture_pred = -1;  // Track if simple gesture was used
            double complex_gesture_pred = -1;  // Track if complex gesture was used
            
            if (high_confidence) {
                // Move to predicted pile position
                auto wait_pose = get_pile_pose(predicted_category);
                wait_pose.pose.position.z += 0.2;
                move_to_pose(wait_pose, planning_time, vel_factor, accel_factor);
                
                // Get binary gesture confirmation
                std::this_thread::sleep_for(std::chrono::milliseconds(800));
                simple_gesture_pred = get_gesture_prediction();
                
                if (simple_gesture_pred >= 0.5) {
                    final_category = predicted_category;
                } else {
                    // Move to center for complex gesture
                    auto center_pose = overhead_scan_pose;
                    center_pose.pose.position.z = 0.3;
                    move_to_pose(center_pose, planning_time, vel_factor, accel_factor);
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(800));
                    complex_gesture_pred = get_complex_gesture_prediction();
                    final_category = static_cast<int>(complex_gesture_pred);
                }
            } else {
                // Low confidence - get complex gesture input immediately
                auto center_pose = overhead_scan_pose;
                center_pose.pose.position.z = 0.3;
                move_to_pose(center_pose, planning_time, vel_factor, accel_factor);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(800));
                complex_gesture_pred = get_complex_gesture_prediction();
                final_category = static_cast<int>(complex_gesture_pred);
            }

            if (final_category < 0 || final_category > 3) {
              final_category = predicted_category;
            }
            
            place_in_stack(i, final_category);
            publish_placement_info(i, final_category);

            // Train networks based on actual results
            if (simple_gesture_pred != -1) {
                // Simple gesture was used - train it with whether its prediction matched the final outcome
                bool was_prediction_correct = (simple_gesture_pred >= 0.5) == (final_category == predicted_category);
                train_gesture_network(was_prediction_correct);
            }
            
            if (complex_gesture_pred != -1) {
                // Complex gesture was used - train it with the actual final category
                train_complex_gesture_network(final_category);
            }

            // Train sorting network with final category
            train_network(i, final_category);

            sorted_index.push_back(i);
            move_to_joints(home_joint_positions);
        }
    }
    
    move_to_pose(overhead_scan_pose, planning_time, vel_factor, accel_factor);
}

void update_piles_callback(
    const std::shared_ptr<franka_hri_interfaces::srv::MoveBlock::Request> request,
    std::shared_ptr<franka_hri_interfaces::srv::MoveBlock::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Updating piles");
    // Find which pile currently contains the block
    std::vector<std::vector<int>*> piles = {&pile_0_index, &pile_1_index, &pile_2_index, &pile_3_index};
    std::vector<geometry_msgs::msg::Pose*> pile_starts = {&pile_0_start, &pile_1_start, &pile_2_start, &pile_3_start};
    std::vector<int>* source_pile = nullptr;
    std::vector<int>* target_pile = nullptr;
    geometry_msgs::msg::Pose* target_pile_start = nullptr;
    
    // First remove collision objects from both source and potential target piles
    // Remove from source pile
    for (auto pile : piles) {
        auto it = std::find(pile->begin(), pile->end(), request->block_index);
        if (it != pile->end()) {
            source_pile = pile;
            // Remove collision objects for the entire source pile
            for (int id : *pile) {
                std::string id_string = std::to_string(id);
                auto block_id = std_msgs::msg::String();
                block_id.data = id_string;
                remove_collision_box(block_id);
            }
            pile->erase(it);
            break;
        }
    }

    // Get target pile based on new category
    if (request->new_category >= 0 && request->new_category < 4) {
        target_pile = piles[request->new_category];
        target_pile_start = pile_starts[request->new_category];
        
        // Remove collision objects from target pile
        for (int id : *target_pile) {
            std::string id_string = std::to_string(id);
            auto block_id = std_msgs::msg::String();
            block_id.data = id_string;
            remove_collision_box(block_id);
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "Invalid target category");
        response->success = false;
        return;
    }

    // Check if target pile is at capacity
    if (target_pile->size() >= 3) {
        target_pile->clear();  // Start a new pile if current one is full
        target_pile_start->position.x += -0.15;  // Move pile start back for new stack
    }

    // Calculate the new position for the block
    geometry_msgs::msg::Pose new_pose = *target_pile_start;
    if (!target_pile->empty()) {
        // Get the top block's position and add height
        auto top_block = block_markers.markers[target_pile->back()];
        auto new_block = block_markers.markers[request->block_index];
        new_pose.position.z = top_block.pose.position.z + (top_block.scale.z / 2) + (new_block.scale.z / 2);
    }

    // Update the marker's position
    auto marker = block_markers.markers[request->block_index];
    marker.pose = new_pose;
    
    // Prepare the update markers request
    auto update_request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
    marker.action = 0;  // Update action
    update_request->input_markers.markers.push_back(marker);
    std::vector<int> update = {request->block_index};
    update_request->markers_to_update = update;

    // Create promise/future pair for marker update
    auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::UpdateMarkers::Response::SharedPtr>>();
    auto future_result = response_received->get_future();

    // Callback for marker update
    auto callback = [this, response_received](
        rclcpp::Client<franka_hri_interfaces::srv::UpdateMarkers>::SharedFuture future) {
        try {
            auto result = future.get();
            this->block_markers = result->output_markers;
            response_received->set_value(result);
            RCLCPP_INFO(this->get_logger(), "Marker update complete");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in update callback: %s", e.what());
        }
    };

    // Send marker update request
    int count = 0;
    while ((!update_markers_cli->wait_for_service(1s)) && (count < 5)) {
        RCLCPP_INFO(this->get_logger(), "Waiting for update_markers service");
        count++;
    }
    update_markers_cli->async_send_request(update_request, callback);

    // Wait for marker update to complete
    if (future_result.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        RCLCPP_ERROR(this->get_logger(), "Failed to update markers - timeout");
        response->success = false;
        return;
    }

    // Add block to new pile's tracking
    target_pile->push_back(request->block_index);

    // Recreate collision objects for both piles
    // Recreate source pile collision objects if it's not empty
    if (source_pile && !source_pile->empty()) {
        for (int id : *source_pile) {
            std::string id_string = std::to_string(id);
            auto block_id = std_msgs::msg::String();
            block_id.data = id_string;
            create_collision_box(block_markers.markers[id].pose, 
                               block_markers.markers[id].scale, 
                               block_id);
        }
    }

    // Recreate target pile collision objects
    for (int id : *target_pile) {
        std::string id_string = std::to_string(id);
        auto block_id = std_msgs::msg::String();
        block_id.data = id_string;
        create_collision_box(block_markers.markers[id].pose, 
                           block_markers.markers[id].scale, 
                           block_id);
    }
    
    response->success = true;
}

// helper function to publish placement info
void publish_placement_info(int last_block_index, int last_placed_category) {
    auto msg = franka_hri_interfaces::msg::BlockPlacementInfo();
    msg.last_block_index = last_block_index;
    
    // Set last block info
    if (!block_markers.markers.empty()) {
        std::vector<int>* last_pile_index;
        switch(last_placed_category) {
            case 0: last_pile_index = &pile_0_index; break;
            case 1: last_pile_index = &pile_1_index; break;
            case 2: last_pile_index = &pile_2_index; break;
            case 3: last_pile_index = &pile_3_index; break;
            default: return;
        }
        if (!last_pile_index->empty()) {
            msg.last_block_pose = block_markers.markers[last_pile_index->back()].pose;
            msg.last_block_category = last_placed_category;
        }
    }

    // Calculate next positions for each category
    msg.next_positions.resize(4);
    
    // For each category
    for (int cat = 0; cat < 4; cat++) {
        geometry_msgs::msg::Pose next_pose;
        std::vector<int>* pile_index;
        geometry_msgs::msg::Pose* pile_start;
        
        switch(cat) {
            case 0: 
                pile_index = &pile_0_index;
                pile_start = &pile_0_start;
                break;
            case 1: 
                pile_index = &pile_1_index;
                pile_start = &pile_1_start;
                break;
            case 2: 
                pile_index = &pile_2_index;
                pile_start = &pile_2_start;
                break;
            case 3: 
                pile_index = &pile_3_index;
                pile_start = &pile_3_start;
                break;
        }

        next_pose = *pile_start;
        
        // If pile is full (3 blocks), adjust position for new pile
        if (pile_index->size() >= 3) {
            next_pose.position.x += -0.15;  // New pile position
            next_pose.position.z = pile_start->position.z;  // Reset height for new pile
        } else if (!pile_index->empty()) {
            // Calculate height based on current top block
            auto last_block = block_markers.markers[pile_index->back()];
            next_pose.position.z = last_block.pose.position.z + (last_block.scale.z / 2) + 0.02; // Add a fixed offset bc lazy
        }
        
        msg.next_positions[cat] = next_pose;
    }

    block_placement_pub->publish(msg);
}

geometry_msgs::msg::PoseStamped get_pile_pose(int category) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = this->get_clock()->now();
    pose.header.frame_id = "world";
    
    // Get current pile position based on category
    switch(category) {
        case 0:
            // Use current pile_0_start position
            pose.pose = pile_0_start;
            if (!pile_0_index.empty()) {
                // Get height of current pile for z position
                auto last_block = block_markers.markers[pile_0_index.back()];
                pose.pose.position.z = last_block.pose.position.z + (last_block.scale.z / 2);
            }
            break;
        case 1:
            pose.pose = pile_1_start;
            if (!pile_1_index.empty()) {
                auto last_block = block_markers.markers[pile_1_index.back()];
                pose.pose.position.z = last_block.pose.position.z + (last_block.scale.z / 2);
            }
            break;
        case 2:
            pose.pose = pile_2_start;
            if (!pile_2_index.empty()) {
                auto last_block = block_markers.markers[pile_2_index.back()];
                pose.pose.position.z = last_block.pose.position.z + (last_block.scale.z / 2);
            }
            break;
        case 3:
            pose.pose = pile_3_start;
            if (!pile_3_index.empty()) {
                auto last_block = block_markers.markers[pile_3_index.back()];
                pose.pose.position.z = last_block.pose.position.z + (last_block.scale.z / 2);
            }
            break;
        default:
            RCLCPP_ERROR(this->get_logger(), "Invalid category: %d", category);
            pose.pose = pile_0_start;
    }

    // Add hovering height for waiting position
    pose.pose.position.z += 0.15;  // Hover 15cm above where block will be placed
    
    return pose;
}

  // Training function for complex gesture network
  void train_complex_gesture_network(int category) {
      auto request = std::make_shared<franka_hri_interfaces::srv::GestNet::Request>();
      request->label = category;

      while (!train_complex_gesture_client->wait_for_service(std::chrono::seconds(1))) {
          if (!rclcpp::ok()) {
              RCLCPP_ERROR(this->get_logger(), "Complex gesture training service interrupted");
              return;
          }
          RCLCPP_INFO(this->get_logger(), "Waiting for complex gesture training service");
      }

      // Send training request and handle response
      auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::GestNet::Response::SharedPtr>>();
      auto future_result = response_received->get_future();

      auto callback = [this, response_received](
          rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedFuture future) {
          try {
              auto result = future.get();
              response_received->set_value(result);
              RCLCPP_INFO(this->get_logger(), "Complex gesture network training complete");
          } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Error in complex gesture training callback: %s", e.what());
          }
      };

      train_complex_gesture_client->async_send_request(request, callback);

      if (future_result.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
          RCLCPP_ERROR(this->get_logger(), "Failed to train complex gesture network - timeout");
          return;
      }
  }

  void human_sorting_callback(const std_msgs::msg::Int8::SharedPtr msg)
  {
      RCLCPP_INFO(this->get_logger(), "Human sorting input: %d", msg->data);
      human_sort_input = msg->data;
  }

  double get_complex_gesture_prediction()
  {
      auto request = std::make_shared<franka_hri_interfaces::srv::GestNet::Request>();

      while (!get_complex_gesture_prediction_client->wait_for_service(std::chrono::seconds(1)))
      {
          if (!rclcpp::ok())
          {
              RCLCPP_ERROR(this->get_logger(), "Gesture prediction service interrupted");
              return -1;
          }
          RCLCPP_INFO(this->get_logger(), "Waiting for gesture prediction service");
      }

      // Use a promise/future pair to handle the async response
      auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::GestNet::Response::SharedPtr>>();
      auto future_result = response_received->get_future();

      auto callback = [response_received](rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedFuture future) {
          response_received->set_value(future.get());
      };

      // Send the request with a callback
      get_complex_gesture_prediction_client->async_send_request(request, callback);

      // Wait for the response with a timeout
      if (future_result.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
          RCLCPP_ERROR(this->get_logger(), "Failed to get gesture prediction - timeout");
          return -1;
      }

      auto result = future_result.get();
      RCLCPP_INFO(this->get_logger(), "Complex Gesture prediction: %f", result->prediction);
      return result->prediction;
  }

  double get_gesture_prediction()
  {
      auto request = std::make_shared<franka_hri_interfaces::srv::GestNet::Request>();

      while (!get_gesture_prediction_client->wait_for_service(std::chrono::seconds(1)))
      {
          if (!rclcpp::ok())
          {
              RCLCPP_ERROR(this->get_logger(), "Gesture prediction service interrupted");
              return -1;
          }
          RCLCPP_INFO(this->get_logger(), "Waiting for gesture prediction service");
      }

      // Use a promise/future pair to handle the async response
      auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::GestNet::Response::SharedPtr>>();
      auto future_result = response_received->get_future();

      auto callback = [response_received](rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedFuture future) {
          response_received->set_value(future.get());
      };

      // Send the request with a callback
      get_gesture_prediction_client->async_send_request(request, callback);

      // Wait for the response with a timeout
      if (future_result.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
          RCLCPP_ERROR(this->get_logger(), "Failed to get gesture prediction - timeout");
          return -1;
      }

      auto result = future_result.get();
      RCLCPP_INFO(this->get_logger(), "Gesture prediction: %f", result->prediction);
      return result->prediction;
  }

  void train_gesture_network(bool was_correct)
  {
      auto request = std::make_shared<franka_hri_interfaces::srv::GestNet::Request>();
      request->label = was_correct ? 1 : 0;

      while (!train_gesture_client->wait_for_service(std::chrono::seconds(1)))
      {
          if (!rclcpp::ok())
          {
              RCLCPP_ERROR(this->get_logger(), "Gesture training service interrupted");
              return;
          }
          RCLCPP_INFO(this->get_logger(), "Waiting for gesture training service");
      }

      // Create promise/future pair for handling response
      auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::GestNet::Response::SharedPtr>>();
      auto future_result = response_received->get_future();

      auto callback = [this, response_received](
          rclcpp::Client<franka_hri_interfaces::srv::GestNet>::SharedFuture future) {
          try {
              auto result = future.get();
              response_received->set_value(result);
              RCLCPP_INFO(this->get_logger(), "Gesture network training complete");
          } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Error in gesture training callback: %s", e.what());
          }
      };

      // Send request asynchronously with callback
      train_gesture_client->async_send_request(request, callback);

      // Wait with timeout for response
      if (future_result.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
          RCLCPP_ERROR(this->get_logger(), "Failed to train gesture network - timeout");
          return;
      }
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

      // Use a promise/future pair to handle the async response
      auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::SortNet::Response::SharedPtr>>();
      auto future_result = response_received->get_future();

      auto callback = [response_received](rclcpp::Client<franka_hri_interfaces::srv::SortNet>::SharedFuture future) {
          response_received->set_value(future.get());
      };

      // Send the request with a callback
      get_network_prediction_client->async_send_request(request, callback);

      // Wait for the response with a timeout
      if (future_result.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
          RCLCPP_ERROR(this->get_logger(), "Failed to get prediction - timeout");
          return -1;
      }

      auto result = future_result.get();
      RCLCPP_INFO(this->get_logger(), "Prediction: %f", result->prediction);
      return result->prediction;
  }

  void place_in_stack(int i, int stack_id)
  {
      if (stack_id >= 0 && stack_id <= 3) {
          double top_z = (stack_id == 0) ? pile_0_start.position.z :
                        (stack_id == 1) ? pile_1_start.position.z :
                        (stack_id == 2) ? pile_2_start.position.z : 
                                        pile_3_start.position.z;
          auto place_pose = geometry_msgs::msg::Pose();
          
          int max_blocks = 3;
          std::vector<int>* current_pile_index;
          geometry_msgs::msg::Pose* current_pile_start;

          // Determine which pile we're working with
          switch(stack_id) {
              case 0:
                  current_pile_index = &pile_0_index;
                  current_pile_start = &pile_0_start;
                  break;
              case 1:
                  current_pile_index = &pile_1_index;
                  current_pile_start = &pile_1_start;
                  break;
              case 2:
                  current_pile_index = &pile_2_index;
                  current_pile_start = &pile_2_start;
                  break;
              case 3:
                  current_pile_index = &pile_3_index;
                  current_pile_start = &pile_3_start;
                  break;
              default:
                  RCLCPP_ERROR(this->get_logger(), "Invalid stack ID");
                  return;
          }

          // Check if current pile is full
          if (current_pile_index->size() >= static_cast<std::vector<int>::size_type>(max_blocks)) {
              current_pile_index->clear();
              current_pile_start->position.x += -0.15;  // Start new pile further back
          }

          // Calculate top of current pile
          if (current_pile_index->size() > 0) {
              auto last_marker_index = current_pile_index->back();
              auto last_marker = block_markers.markers[last_marker_index];
              top_z = last_marker.pose.position.z + (last_marker.scale.z / 2);
          }
          place_pose = *current_pile_start;

          // Set proper z height
          double z_add = (block_markers.markers[i].scale.z / 2);
          if (z_add < 0.02) {
              z_add = 0.02;
          }
          place_pose.position.z = top_z + z_add;

          // Move to hover position
          auto hover_pose = geometry_msgs::msg::PoseStamped();
          hover_pose.header.stamp = this->get_clock()->now();
          hover_pose.header.frame_id = "world";
          hover_pose.pose.position.x = place_pose.position.x;
          hover_pose.pose.position.y = place_pose.position.y;
          hover_pose.pose.position.z = place_pose.position.z + 0.1;
          hover_pose.pose.orientation.x = 1.0;
          hover_pose.pose.orientation.y = 0.0;
          hover_pose.pose.orientation.z = 0.0;
          hover_pose.pose.orientation.w = 0.0;
          
          double planning_time = 10.;
          double vel_factor = 0.4;
          double accel_factor = 0.1;
          move_to_pose(hover_pose, planning_time, vel_factor, accel_factor);

          // Remove collision objects for current pile
          for (int j = 0; j < current_pile_index->size(); ++j) {
              int id = (*current_pile_index)[j];
              std::string id_string = std::to_string(id);
              auto block_id = std_msgs::msg::String();
              block_id.data = id_string;
              remove_collision_box(block_id);
          }

          // Add block to pile
          current_pile_index->push_back(i);

          // Place block
          place_block(place_pose, i);

          // Recreate collision boxes for entire pile
          for (int j = 0; j < current_pile_index->size(); ++j) {
              int id = (*current_pile_index)[j];
              std::string id_string = std::to_string(id);
              auto block_id = std_msgs::msg::String();
              block_id.data = id_string;
              create_collision_box(block_markers.markers[id].pose, block_markers.markers[id].scale, block_id);
          }
      } else {
          RCLCPP_ERROR(this->get_logger(), "Invalid stack ID: %d", stack_id);
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
    drop_pose.pose.position.z = place_pose.position.z + 0.04;
    drop_pose.pose.orientation.x = 1.0;
    drop_pose.pose.orientation.y = 0.0;
    drop_pose.pose.orientation.z = 0.0;
    drop_pose.pose.orientation.w = 0.0;

    move_to_pose(drop_pose, planning_time, vel_factor, accel_factor);

    double width = 0.05;
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
    retreat_pose.pose.position.x += -0.05;
    retreat_pose.pose.position.z += 0.15;
    move_to_pose(retreat_pose, planning_time, vel_factor, accel_factor);

    // Create promise/future pair to handle async response
    auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::UpdateMarkers::Response::SharedPtr>>();
    auto future_result = response_received->get_future();

    // Create callback to handle response
    auto callback = [this, response_received](
        rclcpp::Client<franka_hri_interfaces::srv::UpdateMarkers>::SharedFuture future) {
        try {
            auto result = future.get();
            this->block_markers = result->output_markers;
            response_received->set_value(result);
            RCLCPP_INFO(this->get_logger(), "Marker update complete");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in update callback: %s", e.what());
        }
    };

    int count = 0;
    while ((!update_markers_cli->wait_for_service(1s)) && (count < 5)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for update_markers service");
      count++;
    }

    // Send request asynchronously with callback
    update_markers_cli->async_send_request(request, callback);

    // Wait with timeout for response
    if (future_result.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        RCLCPP_ERROR(this->get_logger(), "Failed to update markers - timeout");
        return;
    }
  }

 void scan_block(int i, bool update_scale)
 {

   RCLCPP_INFO(this->get_logger(), "scanning block called");
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

   RCLCPP_INFO(this->get_logger(), "moving to scan pose");

   move_to_pose(scan_pose, planning_time, vel_factor, accel_factor);

   auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
   request->input_markers = block_markers;
   std::vector<int> update = {i};
   request->markers_to_update = update;
   request->update_scale = update_scale;

   int count = 0;
   while ((!scan_overhead_cli->wait_for_service(1s)) && (count < 10)) {
     RCLCPP_INFO(this->get_logger(), "Waiting for scan_overhead service");
     count++;
   }
    RCLCPP_INFO(this->get_logger(), "requesting scan");
   auto future = scan_overhead_cli->async_send_request(request);
   RCLCPP_INFO(this->get_logger(), "scanning");
   auto result = future.get();
   RCLCPP_INFO(this->get_logger(), "scan complete");
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
    
    RCLCPP_INFO(this->get_logger(), "Grab pose 1");
    move_to_pose(grab_pose_1, planning_time, vel_factor, accel_factor);

    auto grab_pose_2 = grab_pose_1;
    grab_pose_2.header.stamp = this->get_clock()->now();
    grab_pose_2.pose.position.z = marker.pose.position.z + 0.04;
    if (grab_pose_2.pose.position.z < 0.05) {
      grab_pose_2.pose.position.z = 0.06;
    }
    RCLCPP_INFO(this->get_logger(), "Grab pose 2");
    move_to_pose(grab_pose_2, planning_time, vel_factor, accel_factor);

    int count = 0;
    while ((!update_markers_cli->wait_for_service(1s)) && (count < 5)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for update_markers service");
      count++;
    }
    RCLCPP_INFO(this->get_logger(), "updating markers!");

    auto request = std::make_shared<franka_hri_interfaces::srv::UpdateMarkers::Request>();
    marker.action = 2;
    request->input_markers.markers.push_back(marker);
    std::vector<int> update = {i};
    request->markers_to_update = update;
    RCLCPP_INFO(this->get_logger(), "sending update request");

    // Create promise/future pair to handle async response
    auto response_received = std::make_shared<std::promise<franka_hri_interfaces::srv::UpdateMarkers::Response::SharedPtr>>();
    auto future_result = response_received->get_future();

    // Create callback to handle response
    auto callback = [this, response_received](
        rclcpp::Client<franka_hri_interfaces::srv::UpdateMarkers>::SharedFuture future) {
        try {
            auto result = future.get();
            this->block_markers = result->output_markers;
            response_received->set_value(result);
            RCLCPP_INFO(this->get_logger(), "update complete");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in update callback: %s", e.what());
        }
    };

    // Send request asynchronously with callback
    update_markers_cli->async_send_request(request, callback);

    // Wait with timeout for response to ensure we have updated markers before continuing
    if (future_result.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        RCLCPP_ERROR(this->get_logger(), "Failed to update markers - timeout");
        return;
    }

    double width = 0.02;
    double speed = 0.2;
    double force = 0.001;
    RCLCPP_INFO(this->get_logger(), "Grasping!");
    send_grasp_goal(width, speed, force);

    auto retreat_pose = grab_pose_1;
    RCLCPP_INFO(this->get_logger(), "Retreating");
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

    double vel_factor = 0.5;
    double accel_factor = 0.15;

    // Set the joint value target
    move_group->setJointValueTarget(joint_positions);
    move_group->setMaxVelocityScalingFactor(vel_factor);
    move_group->setMaxAccelerationScalingFactor(accel_factor);

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
   collision_object.header.frame_id = "world";
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

      // Create promise/future pair for handling result
      auto result_received = std::make_shared<std::promise<bool>>();
      auto future_result = result_received->get_future();

      // Send goal with callback
      auto send_goal_options = rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions();
      
      send_goal_options.result_callback =
          [this, result_received](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult& result) {
              if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                  RCLCPP_INFO(this->get_logger(), "Grasp succeeded");
                  result_received->set_value(true);
              } else {
                  RCLCPP_INFO(this->get_logger(), "Grasp failed");
                  result_received->set_value(false);
              }
          };

      RCLCPP_INFO(this->get_logger(), "Moving gripper");
      auto goal_handle_future = gripper_grasping_client->async_send_goal(grasp_goal, send_goal_options);

      // Wait for result with timeout
      if (future_result.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
          RCLCPP_ERROR(this->get_logger(), "Gripper action timed out");
          return;
      }
  }

  void hands_detected_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    hands_detected = msg->data;
    
    if (waiting_for_human && !hands_detected) {
      // Hands were present but now gone - trigger block update
      waiting_for_human = false;
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
    
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 
        4
    );
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}