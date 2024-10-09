#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <franka_hri_interfaces/srv/set_increment.hpp>

class TeleopControlNode : public rclcpp::Node
{
public:
    TeleopControlNode() : Node("teleop_control_node")
    {
        // Parameters
        this->declare_parameter("frequency", 10.0);
        this->declare_parameter("linear_scale", 0.1);
        this->declare_parameter("angular_scale", 0.1);

        frequency_ = this->get_parameter("frequency").as_double();
        linear_scale_ = this->get_parameter("linear_scale").as_double();
        angular_scale_ = this->get_parameter("angular_scale").as_double();

        // Subscribers
        joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "joy", 10, std::bind(&TeleopControlNode::joyCallback, this, std::placeholders::_1));

        // Services
        set_increment_client_ = this->create_client<franka_hri_interfaces::srv::SetIncrement>("set_increment");
        enable_teleop_service_ = this->create_service<std_srvs::srv::SetBool>(
            "enable_teleop", std::bind(&TeleopControlNode::enableTeleopCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Timer
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / frequency_),
            std::bind(&TeleopControlNode::timerCallback, this));

        RCLCPP_INFO(this->get_logger(), "Teleop Control Node initialized");
    }

private:
    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
    {
        // Store joystick values
        left_stick_x_ = msg->axes[0];
        left_stick_y_ = msg->axes[1];
        right_stick_x_ = msg->axes[3];
        right_stick_y_ = msg->axes[4];
        left_trigger_ = msg->axes[2];
        right_trigger_ = msg->axes[5];
        left_bumper_ = msg->buttons[4];
        right_bumper_ = msg->buttons[5];
    }

    void timerCallback()
    {
        if (!teleop_enabled_) return;

        auto request = std::make_shared<franka_hri_interfaces::srv::SetIncrement::Request>();

        // Linear motion (left stick)
        request->linear.x = left_stick_y_ * linear_scale_;
        request->linear.y = left_stick_x_ * linear_scale_;
        request->linear.z = (right_trigger_ - left_trigger_) * linear_scale_;

        // Angular motion (right stick)
        request->angular.x = right_stick_y_ * angular_scale_;
        request->angular.y = right_stick_x_ * angular_scale_;
        request->angular.z = (right_bumper_ - left_bumper_) * angular_scale_;

        auto result = set_increment_client_->async_send_request(request);
    }

    void enableTeleopCallback(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        teleop_enabled_ = request->data;
        response->success = true;
        response->message = teleop_enabled_ ? "Teleop enabled" : "Teleop disabled";
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
    }

    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Client<franka_hri_interfaces::srv::SetIncrement>::SharedPtr set_increment_client_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_teleop_service_;
    rclcpp::TimerBase::SharedPtr timer_;

    double frequency_;
    double linear_scale_;
    double angular_scale_;
    bool teleop_enabled_ = false;

    // Joystick values
    double left_stick_x_ = 0.0;
    double left_stick_y_ = 0.0;
    double right_stick_x_ = 0.0;
    double right_stick_y_ = 0.0;
    double left_trigger_ = 0.0;
    double right_trigger_ = 0.0;
    int left_bumper_ = 0;
    int right_bumper_ = 0;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TeleopControlNode>());
    rclcpp::shutdown();
    return 0;
}