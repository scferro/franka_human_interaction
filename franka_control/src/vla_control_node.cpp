#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <franka_hri_interfaces/srv/set_increment.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_hri_interfaces/srv/vla_service.hpp>
#include <franka_hri_interfaces/action/do_action_model.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <chrono>

class VLAControlNode : public rclcpp::Node
{
public:
    using DoActionModel = franka_hri_interfaces::action::DoActionModel;
    using GoalHandleDoActionModel = rclcpp_action::ServerGoalHandle<DoActionModel>;

    VLAControlNode() : Node("vla_control_node")
    {
        // Parameters
        this->declare_parameter("frequency", 5.0);
        this->declare_parameter("linear_scale", 0.001);
        this->declare_parameter("angular_scale", 0.01);
        this->declare_parameter("vla_enabled", true);
        this->declare_parameter("observation_buffer_size", 10);


        frequency_ = this->get_parameter("frequency").as_double();
        linear_scale_ = this->get_parameter("linear_scale").as_double();
        angular_scale_ = this->get_parameter("angular_scale").as_double();
        vla_enabled_ = this->get_parameter("vla_enabled").as_bool();
        observation_buffer_size_ = this->get_parameter("observation_buffer_size").as_int();
        gripper_state_ = false;

        // Subscribers
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10, std::bind(&VLAControlNode::cameraCallback, this, std::placeholders::_1));

        // Services
        set_increment_client_ = this->create_client<franka_hri_interfaces::srv::SetIncrement>("set_increment");
        enable_vla_service_ = this->create_service<std_srvs::srv::SetBool>(
            "enable_vla", std::bind(&VLAControlNode::enableVLACallback, this, std::placeholders::_1, std::placeholders::_2));

        // VLA Service Client
        vla_client_ = this->create_client<franka_hri_interfaces::srv::VLAService>("vla_service");

        // Action client for gripper control
        gripper_action_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
            this, "panda_gripper/grasp");

        // Action server
        this->action_server_ = rclcpp_action::create_server<DoActionModel>(
            this,
            "do_action_model",
            std::bind(&VLAControlNode::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&VLAControlNode::handleCancel, this, std::placeholders::_1),
            std::bind(&VLAControlNode::handleAccepted, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "VLA Control Node initialized");
    }

private:
    void cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Store the latest image
        observation_buffer_.push_back(*msg);
        if (observation_buffer_.size() >= observation_buffer_size_) {
            observation_buffer_.pop_front();
        }
    }

    rclcpp_action::GoalResponse handleGoal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const DoActionModel::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received goal request with command: %s", goal->text_command.c_str());
        (void)uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleCancel(
        const std::shared_ptr<GoalHandleDoActionModel> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handleAccepted(const std::shared_ptr<GoalHandleDoActionModel> goal_handle)
    {
        using namespace std::placeholders;
        // this needs to return quickly to avoid blocking the executor, so spin up a new thread
        std::thread{std::bind(&VLAControlNode::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandleDoActionModel> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Executing goal");
        const auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<DoActionModel::Feedback>();
        auto result = std::make_shared<DoActionModel::Result>();

        auto start_time = this->now();
        double elapsed_time = 0.0;

        while (rclcpp::ok() && elapsed_time < 5.0) {
            if (goal_handle->is_canceling()) {
                result->success = false;
                result->message = "Goal canceled";
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal canceled");
                return;
            }

            if (observation_buffer_.size() < observation_buffer_size_) {
                RCLCPP_WARN(this->get_logger(), "Not enough observations, waiting...");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            auto vla_result = callVLAService(goal->text_command);
            if (vla_result) {
                sendIncrementCommand(vla_result->linear, vla_result->angular);
                if (vla_result->gripper != gripper_state_) {
                    sendGripperCommand(vla_result->gripper);
                    gripper_state_ = vla_result->gripper;
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to get VLA result, aborting action");
                result->success = false;
                result->message = "VLA service call failed";
                goal_handle->abort(result);
                return;
            }

            elapsed_time = (this->now() - start_time).seconds();
            feedback->progress = elapsed_time / 5.0 * 100.0;
            goal_handle->publish_feedback(feedback);

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / frequency_)));
        }

        if (rclcpp::ok()) {
            result->success = true;
            result->message = "Goal succeeded";
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        }
    }

    std::shared_ptr<franka_hri_interfaces::srv::VLAService::Response> callVLAService(const std::string& text_command)
    {
        auto request = std::make_shared<franka_hri_interfaces::srv::VLAService::Request>();
        request->text_command = text_command;
        request->observations = std::vector<sensor_msgs::msg::Image>(observation_buffer_.begin(), observation_buffer_.end());

        auto result_future = vla_client_->async_send_request(request);
        
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result_future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            return result_future.get();
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to call VLA service");
            return nullptr;
        }
    }

    void sendIncrementCommand(const geometry_msgs::msg::Vector3& linear, const geometry_msgs::msg::Vector3& angular)
    {
        auto increment_request = std::make_shared<franka_hri_interfaces::srv::SetIncrement::Request>();

        // Scale the linear and angular values
        increment_request->linear.x = linear.x * linear_scale_;
        increment_request->linear.y = linear.y * linear_scale_;
        increment_request->linear.z = linear.z * linear_scale_;

        increment_request->angular.x = angular.x * angular_scale_;
        increment_request->angular.y = angular.y * angular_scale_;
        increment_request->angular.z = angular.z * angular_scale_;

        RCLCPP_INFO(this->get_logger(), "Sending increment command: Linear (%.3f, %.3f, %.3f), Angular (%.3f, %.3f, %.3f)",
                    increment_request->linear.x, increment_request->linear.y, increment_request->linear.z,
                    increment_request->angular.x, increment_request->angular.y, increment_request->angular.z);

        auto result = set_increment_client_->async_send_request(increment_request);
    }

    void enableVLACallback(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        vla_enabled_ = request->data;
        response->success = true;
        response->message = vla_enabled_ ? "VLA enabled" : "VLA disabled";
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
    }

    void sendGripperCommand(bool close_gripper)
    {
        auto goal_msg = franka_msgs::action::Grasp::Goal();
        goal_msg.width = close_gripper ? 0.02 : 0.08;  // 0.02 for closed, 0.08 for open
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

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    rclcpp::Client<franka_hri_interfaces::srv::SetIncrement>::SharedPtr set_increment_client_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_vla_service_;
    rclcpp::Client<franka_hri_interfaces::srv::VLAService>::SharedPtr vla_client_;
    rclcpp_action::Server<DoActionModel>::SharedPtr action_server_;

    double frequency_;
    double linear_scale_;
    double angular_scale_;
    bool vla_enabled_;
    int observation_buffer_size_;
    bool gripper_state_;

    std::deque<sensor_msgs::msg::Image> observation_buffer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VLAControlNode>());
    rclcpp::shutdown();
    return 0;
}