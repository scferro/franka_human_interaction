#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace franka_control {

class MarkerHandling {
public:
    MarkerHandling(const rclcpp::Node::SharedPtr& node);
    void updateMarkers(const visualization_msgs::msg::MarkerArray& markers);
    visualization_msgs::msg::MarkerArray getMarkers() const;

private:
    rclcpp::Node::SharedPtr node_;
    visualization_msgs::msg::MarkerArray block_markers_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr block_markers_sub_;
    void markerArrayCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);
};

}