#pragma once

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/string.hpp>

namespace franka_control {

class CollisionManagement {
public:
    CollisionManagement(const rclcpp::Node::SharedPtr& node);
    void createCollisionBox(const geometry_msgs::msg::Pose& target_pose, const geometry_msgs::msg::Vector3& size, const std::string& box_id);
    void removeCollisionBox(const std::string& box_id);
    void attachCollisionObjectToGripper(const geometry_msgs::msg::Vector3& size);
    void detachCollisionObjectFromGripper();

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
};

}