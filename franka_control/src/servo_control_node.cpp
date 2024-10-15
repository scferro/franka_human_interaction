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
#include <franka_hri_interfaces/srv/set_increment.hpp>

using std::placeholders::_1, std::placeholders::_2;
using namespace moveit_servo;

double smoothing_factor = 0.025; // Adjust this value to change smoothing strength

static Eigen::Vector3d linear_step_size{0.00, 0.00, 0.00};
static Eigen::Vector3d angular_step_size{0.00, 0.00, 0.00};
bool move_robot = false;

void set_increment_callback(
    const std::shared_ptr<franka_hri_interfaces::srv::SetIncrement::Request> request,
    std::shared_ptr<franka_hri_interfaces::srv::SetIncrement::Response> response)
{
  linear_step_size = Eigen::Vector3d{
    request->linear.x,
    request->linear.y,
    request->linear.z};

  angular_step_size = Eigen::Vector3d{
    request->angular.x,
    request->angular.y,
    request->angular.z};

  response->success = true;
  response->message = "Increment set successfully";
}

int main(int argc, char* argv[])
{
  std::this_thread::sleep_for(std::chrono::milliseconds(750));
  rclcpp::init(argc, argv);

  const rclcpp::Node::SharedPtr servo_control_node = std::make_shared<rclcpp::Node>("servo_control_node");

  rclcpp::Service<franka_hri_interfaces::srv::SetIncrement>::SharedPtr service = 
      servo_control_node->create_service<franka_hri_interfaces::srv::SetIncrement>(
          "set_increment", &set_increment_callback);

  const std::string param_namespace = "moveit_servo";
  const std::shared_ptr<const servo::ParamListener> servo_param_listener =
      std::make_shared<const servo::ParamListener>(servo_control_node, param_namespace);
  const servo::Params servo_params = servo_param_listener->get_params();

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_outgoing_cmd_pub =
      servo_control_node->create_publisher<trajectory_msgs::msg::JointTrajectory>(servo_params.command_out_topic,
                                                                         rclcpp::SystemDefaultsQoS());
  const planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor =
      createPlanningSceneMonitor(servo_control_node, servo_params);
  Servo servo = Servo(servo_control_node, servo_param_listener, planning_scene_monitor);

  std::mutex pose_guard;

  servo.setCommandType(CommandType::POSE);

  PoseCommand target_pose_command;
  target_pose_command.frame_id = servo_params.planning_frame;
  target_pose_command.pose = servo.getEndEffectorPose();

  rclcpp::WallRate command_rate(50);
  Eigen::Isometry3d current_pose = servo.getEndEffectorPose();
  Eigen::Isometry3d target_pose = current_pose;

  while (rclcpp::ok())
  {
    {
      std::lock_guard<std::mutex> pguard(pose_guard);

      // Apply smoothing to the input step sizes
      smoothed_linear_step_size = smoothed_linear_step_size + smoothing_factor * (linear_step_size - smoothed_linear_step_size);
      smoothed_angular_step_size = smoothed_angular_step_size + smoothing_factor * (angular_step_size - smoothed_angular_step_size);

      // Update target pose using smoothed input increments
      Eigen::Isometry3d new_target = target_pose;
      new_target.translate(smoothed_linear_step_size);
      new_target.rotate(Eigen::AngleAxisd(smoothed_angular_step_size.x(), Eigen::Vector3d::UnitX()));
      new_target.rotate(Eigen::AngleAxisd(smoothed_angular_step_size.y(), Eigen::Vector3d::UnitY()));
      new_target.rotate(Eigen::AngleAxisd(smoothed_angular_step_size.z(), Eigen::Vector3d::UnitZ()));

      // No need to smooth deltas now, directly update the target pose
      target_pose = new_target;

      // Update the target_pose_command with the new, smoothed target pose
      target_pose_command.pose = target_pose;

      // Get next joint state based on the smoothed target pose
      KinematicState joint_state = servo.getNextJointState(target_pose_command);

      // Publish the joint state
      trajectory_outgoing_cmd_pub->publish(composeTrajectoryMessage(servo_params, joint_state));

      // Update current pose
      current_pose = servo.getEndEffectorPose();

      rclcpp::spin_some(servo_control_node);
    }
    command_rate.sleep();
  }


  rclcpp::shutdown();
}