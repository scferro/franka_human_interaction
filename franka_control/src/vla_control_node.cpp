#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <franka_hri_interfaces/srv/set_increment.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_hri_interfaces/srv/vla_service.hpp>
#include <franka_hri_interfaces/action/do_action_model.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
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
        this->declare_parameter("observation_buffer_size", 2);
        this->declare_parameter("action_timeout", 60.0);
        this->declare_parameter("enable_depth_masking", false);
        this->declare_parameter("depth_mask_threshold", 1.2); // in meters

        frequency_ = this->get_parameter("frequency").as_double();
        linear_scale_ = this->get_parameter("linear_scale").as_double();
        angular_scale_ = this->get_parameter("angular_scale").as_double();
        vla_enabled_ = this->get_parameter("vla_enabled").as_bool();
        observation_buffer_size_ = this->get_parameter("observation_buffer_size").as_int();
        action_timeout_ = this->get_parameter("action_timeout").as_double();
        enable_depth_masking_ = this->get_parameter("enable_depth_masking").as_bool();
        depth_mask_threshold_ = this->get_parameter("depth_mask_threshold").as_double();
        gripper_state_ = true;

        // Subscribers for D435 main camera
        color_sub_main_.subscribe(this, "/camera/d435i/color/image_raw");
        depth_sub_main_.subscribe(this, "/camera/d435i/aligned_depth_to_color/image_raw");

        // Subscriber for D405 wrist camera
        wrist_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/d405/color/image_rect_raw", 10,
            std::bind(&VLAControlNode::wristImageCallback, this, std::placeholders::_1));

        // Publishers
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("observations/image_main", 10);
        wrist_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("observations/image_wrist", 10);

        // Synchronizer for color and depth images
        sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), color_sub_main_, depth_sub_main_);
        sync_->registerCallback(&VLAControlNode::imageCallback, this);

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

        if (enable_depth_masking_) {
            RCLCPP_INFO(this->get_logger(), "VLA Control Node initialized with depth masking enabled, threshold: %.2f meters", depth_mask_threshold_);
        } else {
            RCLCPP_INFO(this->get_logger(), "VLA Control Node initialized with depth masking disabled");
        }
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_msg,
                                   const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {

        cv_bridge::CvImagePtr cv_color_ptr, cv_depth_ptr;
        try {
            cv_color_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
            cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Resize depth image to match color image
        cv::Mat resized_depth;
        cv::resize(cv_depth_ptr->image, resized_depth, cv_color_ptr->image.size(), 0, 0, cv::INTER_NEAREST);

        cv::Mat processed_image = cv_color_ptr->image.clone();

        if (enable_depth_masking_) {            
            cv::Mat depth_mask;
            double min_depth, max_depth;
            cv::minMaxLoc(resized_depth, &min_depth, &max_depth);

            double threshold = depth_mask_threshold_ * 1000; // Convert to millimeters

            // Create a binary mask
            cv::Mat binary_mask;
            cv::threshold(resized_depth, binary_mask, threshold, 255, cv::THRESH_BINARY_INV);
            binary_mask.convertTo(binary_mask, CV_8U);

            // Apply morphological operations to reduce noise
            int morph_size = 5; // Adjust this value to control the strength of noise reduction
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1));
            
            cv::Mat opened_mask, closed_mask;
            cv::morphologyEx(binary_mask, opened_mask, cv::MORPH_OPEN, element);
            cv::morphologyEx(opened_mask, closed_mask, cv::MORPH_CLOSE, element);

            // Ensure the mask has the same number of channels as the color image
            cv::Mat mask_3ch;
            cv::cvtColor(opened_mask, mask_3ch, cv::COLOR_GRAY2BGR);

            // Apply the mask
            cv::bitwise_and(cv_color_ptr->image, mask_3ch, processed_image);
        } 

        // Convert the processed image back to a ROS message
        sensor_msgs::msg::Image::SharedPtr output_msg = cv_bridge::CvImage(cv_color_ptr->header, "bgr8", processed_image).toImageMsg();

        // Publish the processed image
        image_pub_->publish(*output_msg);

        latest_image_ = *output_msg;

        // Add the image to the observation buffer only if it's not full
        if (observation_buffer_.size() < static_cast<size_t>(observation_buffer_size_)) {
            observation_buffer_.push_back(latest_image_);
            RCLCPP_INFO(this->get_logger(), "Added image to buffer. Current buffer size: %zu", observation_buffer_.size());
        }
    }

    void wristImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& wrist_msg)
    {
        cv_bridge::CvImagePtr cv_wrist_ptr;
        try {
            cv_wrist_ptr = cv_bridge::toCvCopy(wrist_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Store the latest wrist image
        latest_wrist_image_ = *wrist_msg;

        // Add the wrist image to the wrist observation buffer
        wrist_observation_buffer_.push_front(latest_wrist_image_);
        if (wrist_observation_buffer_.size() > static_cast<size_t>(wrist_observation_buffer_size_)) {
            wrist_observation_buffer_.pop_back();
        }

        // Add the image to the observation buffer only if it's not full
        if (wrist_observation_buffer_.size() < static_cast<size_t>(observation_buffer_size_)) {
            wrist_observation_buffer_.push_back(latest_wrist_image_);
            RCLCPP_INFO(this->get_logger(), "Added wrist image to buffer. Current buffer size: %zu", wrist_observation_buffer_.size());
        }

        // Publish the wrist image
        wrist_image_pub_->publish(latest_wrist_image_);
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

        while (rclcpp::ok() && elapsed_time < action_timeout_) {
            if (goal_handle->is_canceling()) {
                result->success = false;
                result->message = "Goal canceled";
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal canceled");
                return;
            }

            // Add the latest image to the observation buffer
            observation_buffer_.push_front(latest_image_);
            if (static_cast<int>(observation_buffer_.size()) > observation_buffer_size_) {
                observation_buffer_.pop_back();
            }

            if (static_cast<int>(observation_buffer_.size()) < observation_buffer_size_) {
                RCLCPP_WARN(this->get_logger(), "Not enough observations, %ld stored observations. Waiting...", observation_buffer_.size());
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
            feedback->progress = elapsed_time / action_timeout_ * 100.0;
            goal_handle->publish_feedback(feedback);

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / frequency_)));
        }

        if (rclcpp::ok()) {
            sendZeroCommand();
            result->success = true;
            result->message = "Goal succeeded";
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        }
    }

    void sendZeroCommand()
    {
        geometry_msgs::msg::Vector3 zero_vector;
        zero_vector.x = 0.0;
        zero_vector.y = 0.0;
        zero_vector.z = 0.0;
        
        sendIncrementCommand(zero_vector, zero_vector);
        RCLCPP_INFO(this->get_logger(), "Sent zero command to stop movement");
    }


    std::shared_ptr<franka_hri_interfaces::srv::VLAService::Response> callVLAService(const std::string& text_command)
    {
        auto request = std::make_shared<franka_hri_interfaces::srv::VLAService::Request>();
        request->text_command = text_command;
        request->observations_main = std::vector<sensor_msgs::msg::Image>(observation_buffer_.begin(), observation_buffer_.end());
        request->observations_wrist = std::vector<sensor_msgs::msg::Image>(wrist_observation_buffer_.begin(), wrist_observation_buffer_.end());

        auto result_future = vla_client_->async_send_request(request);
        
        // Wait for the result with a timeout
        const auto timeout = std::chrono::seconds(20);
        if (result_future.wait_for(timeout) == std::future_status::ready) {
            return result_future.get();
        } else {
            RCLCPP_ERROR(this->get_logger(), "VLA service call timed out");
            return nullptr;
        }
    }

    void sendIncrementCommand(const geometry_msgs::msg::Vector3& linear, const geometry_msgs::msg::Vector3& angular)
    {
        auto increment_request = std::make_shared<franka_hri_interfaces::srv::SetIncrement::Request>();

        // Scale the linear and angular values
        increment_request->linear.x = linear.x * linear_scale_;
        increment_request->linear.y = -linear.y * linear_scale_; // Invert y-axis to match Franka's coordinate system to Octo
        increment_request->linear.z = -linear.z * linear_scale_; // Invert z-axis to match Franka's coordinate system to Octo

        increment_request->angular.x = angular.x * angular_scale_;
        increment_request->angular.y = -angular.y * angular_scale_; // Invert y-axis to match Franka's coordinate system to Octo
        increment_request->angular.z = -angular.z * angular_scale_; // Invert z-axis to match Franka's coordinate system to Octo

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

    void sendGripperCommand(bool open_gripper)
    {
        auto goal_msg = franka_msgs::action::Grasp::Goal();
        goal_msg.width = open_gripper ? 0.08 : 0.02;  // 0.02 for closed, 0.08 for open
        goal_msg.speed = 0.1;  // Adjust as needed
        goal_msg.force = 0.001;  // Adjust as needed

        RCLCPP_INFO(this->get_logger(), "Sending gripper command: %s", open_gripper ? "Open" : "Close");

        auto send_goal_options = rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions();
        send_goal_options.goal_response_callback =
            std::bind(&VLAControlNode::gripperGoalResponseCallback, this, std::placeholders::_1);
        send_goal_options.result_callback =
            std::bind(&VLAControlNode::gripperResultCallback, this, std::placeholders::_1);

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

    rclcpp::Client<franka_hri_interfaces::srv::SetIncrement>::SharedPtr set_increment_client_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_vla_service_;
    rclcpp::Client<franka_hri_interfaces::srv::VLAService>::SharedPtr vla_client_;
    rclcpp_action::Server<DoActionModel>::SharedPtr action_server_;
    rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_action_client_;
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_main_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_main_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr wrist_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr wrist_image_pub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
    std::shared_ptr<Synchronizer> sync_;

    double frequency_;
    double action_timeout_;
    double linear_scale_;
    double angular_scale_;
    bool vla_enabled_;
    int observation_buffer_size_;
    int wrist_observation_buffer_size_;
    bool gripper_state_;
    bool enable_depth_masking_;
    double depth_mask_threshold_;

    sensor_msgs::msg::Image latest_image_;
    sensor_msgs::msg::Image latest_wrist_image_;
    std::deque<sensor_msgs::msg::Image> wrist_observation_buffer_;
    std::deque<sensor_msgs::msg::Image> observation_buffer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VLAControlNode>());
    rclcpp::shutdown();
    return 0;
}