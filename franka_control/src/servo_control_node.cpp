#include <Eigen/Geometry>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/Transform.h>
#include <atomic>
#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <moveit_servo/servo.hpp>
#include <moveit_servo/utils/common.hpp>
#include <mutex>
#include <std_srvs/srv/empty.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <franka_hri_interfaces/msg/ee_command.hpp>
#include <thread>

using std::placeholders::_1, std::placeholders::_2;
using namespace moveit_servo;

// Current commanded velocities
static Eigen::Vector3d linear_vel_cmd{0.00, 0.00, 0.00};
static Eigen::Vector3d angular_vel_cmd{0.00, 0.00, 0.00};

// Current smoothed velocities
static Eigen::Vector3d current_linear_vel{0.00, 0.00, 0.00};
static Eigen::Vector3d current_angular_vel{0.00, 0.00, 0.00};

bool move_robot = false;

void ee_command_callback(const franka_hri_interfaces::msg::EECommand::SharedPtr msg)
{
  linear_vel_cmd = Eigen::Vector3d{msg->linear.x, msg->linear.y, msg->linear.z};
  angular_vel_cmd = Eigen::Vector3d{msg->angular.x, msg->angular.y, msg->angular.z};
}

int main(int argc, char* argv[])
{
  std::this_thread::sleep_for(std::chrono::milliseconds(750));
  rclcpp::init(argc, argv);

  const rclcpp::Node::SharedPtr servo_control_node = std::make_shared<rclcpp::Node>("servo_control_node");

  // Declare and get smoothing parameters
  servo_control_node->declare_parameter("max_linear_acc", 2.0);  // m/s^2
  servo_control_node->declare_parameter("max_angular_acc", 4.0); // rad/s^2
  servo_control_node->declare_parameter("smoothing_factor", 0.8); // 0-1, higher = less smoothing

  const double max_linear_acc = servo_control_node->get_parameter("max_linear_acc").as_double();
  const double max_angular_acc = servo_control_node->get_parameter("max_angular_acc").as_double();
  const double smoothing_factor = servo_control_node->get_parameter("smoothing_factor").as_double();

  auto ee_command_sub = servo_control_node->create_subscription<franka_hri_interfaces::msg::EECommand>(
      "ee_command", 10, &ee_command_callback);

  const std::string param_namespace = "moveit_servo";
  const std::shared_ptr<const servo::ParamListener> servo_param_listener =
      std::make_shared<const servo::ParamListener>(servo_control_node, param_namespace);
  servo::Params servo_params = servo_param_listener->get_params();

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_outgoing_cmd_pub =
      servo_control_node->create_publisher<trajectory_msgs::msg::JointTrajectory>(servo_params.command_out_topic,
                                                                         rclcpp::SystemDefaultsQoS());

  const planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor =
      createPlanningSceneMonitor(servo_control_node, servo_params);

  Servo servo = Servo(servo_control_node, servo_param_listener, planning_scene_monitor);

  std::mutex twist_guard;

  servo.setCommandType(CommandType::TWIST);

  TwistCommand target_twist;
  target_twist.frame_id = servo_params.planning_frame;
  target_twist.velocities.resize(6);

  auto robot_state = planning_scene_monitor->getStateMonitor()->getCurrentState();
  const moveit::core::JointModelGroup* joint_model_group =
      robot_state->getJointModelGroup(servo_params.move_group_name);

  const double update_rate = 30.0; // Hz
  const double dt = 1.0 / update_rate;
  rclcpp::WallRate command_rate(update_rate);
  RCLCPP_INFO_STREAM(servo_control_node->get_logger(), servo.getStatusMessage());

  while (rclcpp::ok())
  {
    {
      std::lock_guard<std::mutex> tguard(twist_guard);
      
      // Calculate velocity differences
      Eigen::Vector3d linear_vel_diff = linear_vel_cmd - current_linear_vel;
      Eigen::Vector3d angular_vel_diff = angular_vel_cmd - current_angular_vel;

      // Calculate accelerations
      Eigen::Vector3d linear_acc = linear_vel_diff / dt;
      Eigen::Vector3d angular_acc = angular_vel_diff / dt;

      // Limit accelerations
      if (linear_acc.norm() > max_linear_acc) {
        linear_acc = linear_acc.normalized() * max_linear_acc;
      }
      if (angular_acc.norm() > max_angular_acc) {
        angular_acc = angular_acc.normalized() * max_angular_acc;
      }

      // Calculate next velocities based on limited accelerations
      Eigen::Vector3d next_linear_vel = current_linear_vel + linear_acc * dt;
      Eigen::Vector3d next_angular_vel = current_angular_vel + angular_acc * dt;

      // Apply smoothing filter
      current_linear_vel = smoothing_factor * next_linear_vel + 
                          (1.0 - smoothing_factor) * current_linear_vel;
      current_angular_vel = smoothing_factor * next_angular_vel + 
                           (1.0 - smoothing_factor) * current_angular_vel;

      // Use smoothed velocities for the twist command
      target_twist.velocities << current_linear_vel.x(), current_linear_vel.y(), current_linear_vel.z(),
                                current_angular_vel.x(), current_angular_vel.y(), current_angular_vel.z();

      KinematicState joint_state = servo.getNextJointState(target_twist);
      StatusCode status = servo.getStatus();

      if (status != StatusCode::INVALID)
      {
        trajectory_msgs::msg::JointTrajectory trajectory_msg = composeTrajectoryMessage(servo_params, joint_state);
        trajectory_outgoing_cmd_pub->publish(trajectory_msg);
        
        robot_state->setJointGroupPositions(joint_model_group, joint_state.positions);
        robot_state->setJointGroupVelocities(joint_model_group, joint_state.velocities);
      }
    }
    rclcpp::spin_some(servo_control_node);
    command_rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}