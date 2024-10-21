#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <franka_hri_interfaces/msg/increment.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

class TeleopControlNode : public rclcpp::Node
{
public:
    TeleopControlNode() : Node("teleop_control_node")
    {
        // Parameters
        this->declare_parameter("frequency", 10.0);
        this->declare_parameter("linear_scale", 0.0001);
        this->declare_parameter("angular_scale", 0.001);
        this->declare_parameter("teleop_enabled", true);

        frequency_ = this->get_parameter("frequency").as_double();
        linear_scale_ = this->get_parameter("linear_scale").as_double();
        angular_scale_ = this->get_parameter("angular_scale").as_double();
        teleop_enabled_ = this->get_parameter("teleop_enabled").as_bool();

        // Subscribers
        joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "joy", 10, std::bind(&TeleopControlNode::joyCallback, this, std::placeholders::_1));

        // Publishers
        increment_pub_ = this->create_publisher<franka_hri_interfaces::msg::Increment>("increment", 10);

        // Services
        enable_teleop_service_ = this->create_service<std_srvs::srv::SetBool>(
            "enable_teleop", std::bind(&TeleopControlNode::enableTeleopCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Action client for gripper control
        gripper_action_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
            this, "panda_gripper/grasp");

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

        // Handle A button (open gripper)
        if (msg->buttons[0] && !a_button_pressed_) {
            a_button_pressed_ = true;
            sendGripperCommand(false);  // Open gripper
        } else if (!msg->buttons[0]) {
            a_button_pressed_ = false;
        }

        // Handle B button (close gripper)
        if (msg->buttons[1] && !b_button_pressed_) {
            b_button_pressed_ = true;
            sendGripperCommand(true);  // Close gripper
        } else if (!msg->buttons[1]) {
            b_button_pressed_ = false;
        }
    }

    void timerCallback()
    {
        if (!teleop_enabled_) return;

        auto increment_msg = franka_hri_interfaces::msg::Increment();

        // Linear motion (left stick)
        increment_msg.linear.x = left_stick_y_ * linear_scale_;
        increment_msg.linear.y = left_stick_x_ * linear_scale_;
        increment_msg.linear.z = (right_trigger_ - left_trigger_) * linear_scale_ / 2.;

        // Angular motion (right stick)
        increment_msg.angular.x = right_stick_y_ * angular_scale_;
        increment_msg.angular.y = right_stick_x_ * angular_scale_;
        increment_msg.angular.z = (right_bumper_ - left_bumper_) * angular_scale_;
        
        RCLCPP_INFO(this->get_logger(), "Linear X: %f, Linear Y: %f, Linear Z: %f", 
                    increment_msg.linear.x, increment_msg.linear.y, increment_msg.linear.z);
        RCLCPP_INFO(this->get_logger(), "Angular X: %f, Angular Y: %f, Angular Z: %f", 
                    increment_msg.angular.x, increment_msg.angular.y, increment_msg.angular.z);

        increment_pub_->publish(increment_msg);
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

    void sendGripperCommand(bool close_gripper)
    {
        auto goal_msg = franka_msgs::action::Grasp::Goal();
        goal_msg.width = close_gripper ? 0.02 : 0.08;  // 0.0 for closed, 0.08 for open
        goal_msg.speed = 0.1;  // Adjust as needed
        goal_msg.force = 0.001;  // Adjust as needed

        RCLCPP_INFO(this->get_logger(), "Sending gripper command: %s", close_gripper ? "Close" : "Open");

        auto send_goal_options = rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions();
        send_goal_options.goal_response_callback =
            std::bind(&TeleopControlNode::gripperGoalResponseCallback, this, std::placeholders::_1);
        send_goal_options.result_callback =
            std::bind(&TeleopControlNode::gripperResultCallback, this, std::placeholders::_1);

        gripper_action_client_->async_send_goal(goal_msg, send_goal_options);
    }

    void gripperGoalResponseCallback(const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::SharedPtr & goal_handle)
    {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Gripper goal was rejected by server");
        } else {
            RCLCPP_INFO(this->get_logger(), "Gripper goal accepted by server, waiting for result");
        }
    }

    void gripperResultCallback(const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult & result)
    {
        switch (result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                RCLCPP_INFO(this->get_logger(), "Gripper action succeeded");
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_ERROR(this->get_logger(), "Gripper action was aborted");
                break;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_ERROR(this->get_logger(), "Gripper action was canceled");
                break;
            default:
                RCLCPP_ERROR(this->get_logger(), "Unknown gripper action result code");
                break;
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Publisher<franka_hri_interfaces::msg::Increment>::SharedPtr increment_pub_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_teleop_service_;
    rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_action_client_;
    rclcpp::TimerBase::SharedPtr timer_;

    double frequency_;
    double linear_scale_;
    double angular_scale_;
    bool teleop_enabled_;

    // Joystick values
    double left_stick_x_ = 0.0;
    double left_stick_y_ = 0.0;
    double right_stick_x_ = 0.0;
    double right_stick_y_ = 0.0;
    double left_trigger_ = 0.0;
    double right_trigger_ = 0.0;
    int left_bumper_ = 0;
    int right_bumper_ = 0;

    // Gripper control
    bool a_button_pressed_ = false;
    bool b_button_pressed_ = false;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TeleopControlNode>());
    rclcpp::shutdown();
    return 0;
}