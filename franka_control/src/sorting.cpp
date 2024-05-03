#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <std_srvs/srv/empty.hpp>

using namespace std::chrono_literals;

class Sorting : public rclcpp::Node
{
public:
  Sorting()
  : Node("sorting")
  {
    declare_parameter("rate", 200.);
    loop_rate = get_parameter("rate").as_double();

    scan_srv = create_service<std_srvs::srv::Empty>(
      "move_to_scan_pose",
      std::bind(&Sorting::move_to_scan, this, std::placeholders::_1, std::placeholders::_2));

    left_wait_srv = create_service<std_srvs::srv::Empty>(
      "move_to_left_wait",
      std::bind(&Sorting::move_to_left_wait_pose, this, std::placeholders::_1, std::placeholders::_2));

    left_drop_srv = create_service<std_srvs::srv::Empty>(
      "drop_left",
      std::bind(&Sorting::drop_left, this, std::placeholders::_1, std::placeholders::_2));

    right_wait_srv = create_service<std_srvs::srv::Empty>(
      "move_to_right_wait",
      std::bind(&Sorting::move_to_right_wait_pose, this, std::placeholders::_1, std::placeholders::_2));

    right_drop_srv = create_service<std_srvs::srv::Empty>(
      "drop_right",
      std::bind(&Sorting::drop_right, this, std::placeholders::_1, std::placeholders::_2));

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
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr scan_srv;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr left_wait_srv;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr left_drop_srv;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr right_wait_srv;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr right_drop_srv;
  rclcpp::TimerBase::SharedPtr main_timer;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_hand;

  void timer_callback()
  {
    int something = 0;
  }

  void move_to_scan(
    std_srvs::srv::Empty::Request::SharedPtr,
    std_srvs::srv::Empty::Response::SharedPtr)
  {
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
    move_group->setMaxVelocityScalingFactor(0.8);   // Max speed (1.0 is full speed)
    move_group->setMaxAccelerationScalingFactor(0.2); // Max acceleration

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      control_gripper(0.00); 
    } else {
      RCLCPP_INFO(this->get_logger(), "Planning failed RIP");
    }
  }

  void move_to_left_wait_pose(
    std_srvs::srv::Empty::Request::SharedPtr,
    std_srvs::srv::Empty::Response::SharedPtr)
  {
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
    move_group->setMaxVelocityScalingFactor(0.8);   // Max speed (1.0 is full speed)
    move_group->setMaxAccelerationScalingFactor(0.2); // Max acceleration

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
    } else {
      RCLCPP_INFO(this->get_logger(), "Planning failed RIP");
    }
  }

  void drop_left(
    std_srvs::srv::Empty::Request::SharedPtr,
    std_srvs::srv::Empty::Response::SharedPtr)
  {
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
    move_group->setMaxVelocityScalingFactor(0.8);   // Max speed (1.0 is full speed)
    move_group->setMaxAccelerationScalingFactor(0.2); // Max acceleration

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      control_gripper(0.04); 
    } else {
      RCLCPP_INFO(this->get_logger(), "Planning failed RIP");
    }
  }

  void move_to_right_wait_pose(
    std_srvs::srv::Empty::Request::SharedPtr,
    std_srvs::srv::Empty::Response::SharedPtr)
  {
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
    move_group->setMaxVelocityScalingFactor(0.8);   // Max speed (1.0 is full speed)
    move_group->setMaxAccelerationScalingFactor(0.2); // Max acceleration

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
    } else {
      RCLCPP_INFO(this->get_logger(), "Planning failed RIP");
    }
  }

  void drop_right(
    std_srvs::srv::Empty::Request::SharedPtr,
    std_srvs::srv::Empty::Response::SharedPtr)
  {
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
    move_group->setMaxVelocityScalingFactor(0.8);   // Max speed (1.0 is full speed)
    move_group->setMaxAccelerationScalingFactor(0.2); // Max acceleration

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group->execute(plan);
      control_gripper(0.04); 
    } else {
      RCLCPP_INFO(this->get_logger(), "Planning failed RIP");
    }
  }

  void control_gripper(double position)
  {
    std::map<std::string, double> joint_values;
    joint_values["panda_finger_joint1"] = position;

    move_group_hand->setJointValueTarget(joint_values);
    move_group_hand->move();
    RCLCPP_INFO(this->get_logger(), "Gripper moved to position: %f", position);
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
