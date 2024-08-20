#include "franka_control/collision_management.hpp"

namespace franka_control {

CollisionManagement::CollisionManagement(const rclcpp::Node::SharedPtr& node)
    : node_(node)
{
    planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
}

void CollisionManagement::createCollisionBox(const geometry_msgs::msg::Pose& target_pose, const geometry_msgs::msg::Vector3& size, const std::string& box_id)
{
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = "world";
    collision_object.id = box_id;

    shape_msgs::msg::SolidPrimitive box_primitive;
    box_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    box_primitive.dimensions = {size.x, size.y, size.z};

    collision_object.primitives.push_back(box_primitive);
    collision_object.primitive_poses.push_back(target_pose);
    collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface_->applyCollisionObject(collision_object);
    RCLCPP_INFO(node_->get_logger(), "Created collision box.");
}

void CollisionManagement::removeCollisionBox(const std::string& box_id)
{
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.id = box_id;
    collision_object.operation = moveit_msgs::msg::CollisionObject::REMOVE;

    planning_scene_interface_->applyCollisionObject(collision_object);
    RCLCPP_INFO(node_->get_logger(), "Removed collision box.");
}

void CollisionManagement::attachCollisionObjectToGripper(const geometry_msgs::msg::Vector3& size)
{
    moveit_msgs::msg::AttachedCollisionObject attached_object;
    attached_object.link_name = "panda_hand";
    attached_object.object.header.frame_id = "panda_hand";
    attached_object.object.id = "attached_object";

    shape_msgs::msg::SolidPrimitive box_primitive;
    box_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    box_primitive.dimensions = {size.x, size.y, size.z};

    geometry_msgs::msg::Pose object_pose;
    object_pose.position.z = size.z / 2;

    attached_object.object.primitives.push_back(box_primitive);
    attached_object.object.primitive_poses.push_back(object_pose);
    attached_object.object.operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface_->applyAttachedCollisionObject(attached_object);
    RCLCPP_INFO(node_->get_logger(), "Attached collision object to gripper.");
}

void CollisionManagement::detachCollisionObjectFromGripper()
{
    moveit_msgs::msg::AttachedCollisionObject detach_object;
    detach_object.object.id = "attached_object";
    detach_object.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;

    planning_scene_interface_->applyAttachedCollisionObject(detach_object);
    RCLCPP_INFO(node_->get_logger(), "Detached collision object from gripper.");
}

}