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

  const rclcpp::Node::SharedPtr demo_node = std::make_shared<rclcpp::Node>("franka_servo");

  rclcpp::Service<franka_hri_interfaces::srv::SetIncrement>::SharedPtr service = 
      demo_node->create_service<franka_hri_interfaces::srv::SetIncrement>(
          "set_increment", &set_increment_callback);

  const std::string param_namespace = "moveit_servo";
  const std::shared_ptr<const servo::ParamListener> servo_param_listener =
      std::make_shared<const servo::ParamListener>(demo_node, param_namespace);
  const servo::Params servo_params = servo_param_listener->get_params();

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_outgoing_cmd_pub =
      demo_node->create_publisher<trajectory_msgs::msg::JointTrajectory>(servo_params.command_out_topic,
                                                                         rclcpp::SystemDefaultsQoS());
  const planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor =
      createPlanningSceneMonitor(demo_node, servo_params);
  Servo servo = Servo(demo_node, servo_param_listener, planning_scene_monitor);

  std::mutex pose_guard;

  servo.setCommandType(CommandType::POSE);

  PoseCommand target_pose;
  target_pose.frame_id = servo_params.planning_frame;
  target_pose.pose = servo.getEndEffectorPose();

  auto pose_tracker = [&]() {
    KinematicState joint_state;
    rclcpp::WallRate tracking_rate(1 / servo_params.publish_period);
    while (rclcpp::ok())
    {
      {
        std::lock_guard<std::mutex> pguard(pose_guard);
        joint_state = servo.getNextJointState(target_pose);
      }
      StatusCode status = servo.getStatus();
      if (status != StatusCode::INVALID)
        trajectory_outgoing_cmd_pub->publish(composeTrajectoryMessage(servo_params, joint_state));

      tracking_rate.sleep();
    }
  };

  std::thread tracker_thread(pose_tracker);
  tracker_thread.detach();

  rclcpp::WallRate command_rate(50);
  RCLCPP_INFO_STREAM(demo_node->get_logger(), servo.getStatusMessage());

  while (rclcpp::ok())
  {
    {
      std::lock_guard<std::mutex> pguard(pose_guard);
      target_pose.pose = servo.getEndEffectorPose();
      target_pose.pose.translate(linear_step_size);
      target_pose.pose.rotate(Eigen::AngleAxisd(angular_step_size.x(), Eigen::Vector3d::UnitX()));
      target_pose.pose.rotate(Eigen::AngleAxisd(angular_step_size.y(), Eigen::Vector3d::UnitY()));
      target_pose.pose.rotate(Eigen::AngleAxisd(angular_step_size.z(), Eigen::Vector3d::UnitZ()));

      rclcpp::spin_some(demo_node);
    }
    command_rate.sleep();
  }

  if (tracker_thread.joinable())
    tracker_thread.join();

  rclcpp::shutdown();
}