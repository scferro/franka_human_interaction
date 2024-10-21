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
#include <franka_hri_interfaces/msg/increment.hpp>
#include <thread>

using std::placeholders::_1, std::placeholders::_2;
using namespace moveit_servo;

static Eigen::Vector3d linear_vel_cmd{0.00, 0.00, 0.00};
static Eigen::Vector3d angular_vel_cmd{0.00, 0.00, 0.00};
bool move_robot = false;

void increment_callback(const franka_hri_interfaces::msg::Increment::SharedPtr msg)
{
  linear_vel_cmd = Eigen::Vector3d{msg->linear.x, msg->linear.y, msg->linear.z};
  angular_vel_cmd = Eigen::Vector3d{msg->angular.x, msg->angular.y, msg->angular.z};
}

int main(int argc, char* argv[])
{
  std::this_thread::sleep_for(std::chrono::milliseconds(750));
  rclcpp::init(argc, argv);

  const rclcpp::Node::SharedPtr servo_control_node = std::make_shared<rclcpp::Node>("servo_control_node");

  auto increment_sub = servo_control_node->create_subscription<franka_hri_interfaces::msg::Increment>(
      "increment", 10, &increment_callback);

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

  rclcpp::WallRate command_rate(30);
  RCLCPP_INFO_STREAM(servo_control_node->get_logger(), servo.getStatusMessage());

  while (rclcpp::ok())
  {
    {
      std::lock_guard<std::mutex> tguard(twist_guard);
      target_twist.velocities << linear_vel_cmd.x(), linear_vel_cmd.y(), linear_vel_cmd.z(),
                                 angular_vel_cmd.x(), angular_vel_cmd.y(), angular_vel_cmd.z();

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