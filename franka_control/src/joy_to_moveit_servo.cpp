#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

class JoyToMoveItServo : public rclcpp::Node
{
public:
  JoyToMoveItServo()
  : Node("joy_to_moveit_servo")
  {
    // Subscribe to joy messages
    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
      "joy", 10, std::bind(&JoyToMoveItServo::joyCallback, this, std::placeholders::_1));

    // Create publisher for MoveIt Servo commands
    servo_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("servo_server/delta_twist_cmds", 10);
  }

private:
  void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    geometry_msgs::msg::TwistStamped twist_cmd;
    twist_cmd.header.stamp = this->now();
    twist_cmd.header.frame_id = "base_link";  // Adjust as needed

    // Map joystick axes to Cartesian velocities
    // Assuming axes[0] is left/right, axes[1] is forward/backward, axes[2] is up/down
    twist_cmd.twist.linear.x = msg->axes[1] * 0.1;  // Scale factor of 0.1 m/s
    twist_cmd.twist.linear.y = msg->axes[0] * 0.1;
    twist_cmd.twist.linear.z = msg->axes[2] * 0.1;

    // Map joystick axes to angular velocities
    // Assuming axes[3] is rotation around z, axes[4] is rotation around y, axes[5] is rotation around x
    twist_cmd.twist.angular.x = msg->axes[5] * 0.1;  // Scale factor of 0.1 rad/s
    twist_cmd.twist.angular.y = msg->axes[4] * 0.1;
    twist_cmd.twist.angular.z = msg->axes[3] * 0.1;

    // Publish the twist command
    servo_pub_->publish(twist_cmd);
  }

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr servo_pub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<JoyToMoveItServo>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}