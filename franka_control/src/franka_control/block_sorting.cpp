#include "franka_control/block_sorting.hpp"
#include "franka_control/utils.hpp"
#include <franka_hri_interfaces/srv/sort_net.hpp>

namespace franka_control {

BlockSorting::BlockSorting(const rclcpp::Node::SharedPtr& node)
    : node_(node)
{
    robot_control_ = std::make_unique<RobotControl>(node);
    collision_management_ = std::make_unique<CollisionManagement>(node);
    gripper_control_ = std::make_unique<GripperControl>(node);
    marker_handling_ = std::make_unique<MarkerHandling>(node);
}

void BlockSorting::sortBlocks()
{
    auto markers = marker_handling_->getMarkers();
    for (size_t i = 0; i < markers.markers.size(); ++i) {
        scanBlock(i, true);
        double prediction = getNetworkPrediction(i);
        grabBlock(i);
        int stack_id = (prediction >= 0.5) ? 1 : 0;
        placeInStack(i, stack_id);
        trainNetwork(i, stack_id);
    }
}

void BlockSorting::scanBlock(int index, bool update_scale)
{
    auto markers = marker_handling_->getMarkers();
    auto marker = markers.markers[index];
    
    geometry_msgs::msg::PoseStamped scan_pose;
    scan_pose.header.frame_id = "world";
    scan_pose.header.stamp = node_->now();
    scan_pose.pose = utils::createPose(
        marker.pose.position.x,
        marker.pose.position.y,
        marker.pose.position.z + 0.15,
        1.0, 0.0, 0.0, 0.0
    );

    robot_control_->moveToPose(scan_pose, 20.0, 0.4, 0.1);

    // Here you would typically call a service to update the marker's position
    // For this example, we'll just use the existing marker data
    marker_handling_->updateMarkers(markers);
}

void BlockSorting::grabBlock(int index)
{
    auto markers = marker_handling_->getMarkers();
    auto marker = markers.markers[index];

    geometry_msgs::msg::PoseStamped grab_pose;
    grab_pose.header.frame_id = "world";
    grab_pose.header.stamp = node_->now();
    grab_pose.pose = utils::createPose(
        marker.pose.position.x,
        marker.pose.position.y,
        marker.pose.position.z + 0.05,
        1.0, 0.0, 0.0, 0.0
    );

    robot_control_->moveToPose(grab_pose, 20.0, 0.4, 0.1);
    gripper_control_->sendGraspGoal(0.02, 0.1, 10.0);

    collision_management_->attachCollisionObjectToGripper(marker.scale);
}

void BlockSorting::placeInStack(int index, int stack_id)
{
    // Define stack positions (you may want to make these configurable)
    std::vector<geometry_msgs::msg::Pose> stack_positions = {
        utils::createPose(0.5, 0.2, 0.05, 1.0, 0.0, 0.0, 0.0),  // Stack 0
        utils::createPose(0.5, -0.2, 0.05, 1.0, 0.0, 0.0, 0.0)  // Stack 1
    };

    geometry_msgs::msg::PoseStamped place_pose;
    place_pose.header.frame_id = "world";
    place_pose.header.stamp = node_->now();
    place_pose.pose = stack_positions[stack_id];

    robot_control_->moveToPose(place_pose, 20.0, 0.4, 0.1);
    
    gripper_control_->sendGraspGoal(0.08, 0.1, 10.0);  // Open gripper
    collision_management_->detachCollisionObjectFromGripper();

    // Update marker position
    auto markers = marker_handling_->getMarkers();
    markers.markers[index].pose = place_pose.pose;
    marker_handling_->updateMarkers(markers);
}

double BlockSorting::getNetworkPrediction(int index)
{
    auto client = node_->create_client<franka_hri_interfaces::srv::SortNet>("get_network_prediction");
    auto request = std::make_shared<franka_hri_interfaces::srv::SortNet::Request>();
    request->index = index;

    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node_, future) != rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to call get_network_prediction service");
        return -1.0;
    }

    return future.get()->prediction;
}

void BlockSorting::trainNetwork(int index, int label)
{
    auto client = node_->create_client<franka_hri_interfaces::srv::SortNet>("train_network");
    auto request = std::make_shared<franka_hri_interfaces::srv::SortNet::Request>();
    request->index = index;
    request->label = label;

    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node_, future) != rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to call train_network service");
    }
}

}