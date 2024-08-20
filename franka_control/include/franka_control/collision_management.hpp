#pragma once

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/string.hpp>

namespace franka_control {

/**
 * @brief Class for managing collisions in the robot's environment.
 */
class CollisionManagement {
public:
    /**
     * @brief Construct a new Collision Management object.
     * 
     * @param node Shared pointer to the ROS2 node.
     */
    CollisionManagement(const rclcpp::Node::SharedPtr& node);

    /**
     * @brief Create a collision box in the planning scene.
     * 
     * @param target_pose Pose of the collision box.
     * @param size Size of the collision box.
     * @param box_id Unique identifier for the collision box.
     */
    void createCollisionBox(const geometry_msgs::msg::Pose& target_pose, const geometry_msgs::msg::Vector3& size, const std::string& box_id);

    /**
     * @brief Remove a collision box from the planning scene.
     * 
     * @param box_id Unique identifier of the collision box to remove.
     */
    void removeCollisionBox(const std::string& box_id);

    /**
     * @brief Attach a collision object to the robot's gripper.
     * 
     * @param size Size of the collision object to attach.
     */
    void attachCollisionObjectToGripper(const geometry_msgs::msg::Vector3& size);

    /**
     * @brief Detach the collision object from the robot's gripper.
     */
    void detachCollisionObjectFromGripper();

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
};

}