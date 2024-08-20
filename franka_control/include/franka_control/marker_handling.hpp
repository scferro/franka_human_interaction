#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace franka_control {

/**
 * @brief Class for handling visualization markers.
 */
class MarkerHandling {
public:
    /**
     * @brief Construct a new Marker Handling object.
     * 
     * @param node Shared pointer to the ROS2 node.
     */
    MarkerHandling(const rclcpp::Node::SharedPtr& node);

    /**
     * @brief Update the stored markers with new data.
     * 
     * @param markers New marker array to update with.
     */
    void updateMarkers(const visualization_msgs::msg::MarkerArray& markers);

    /**
     * @brief Get the current stored markers.
     * 
     * @return visualization_msgs::msg::MarkerArray The current marker array.
     */
    visualization_msgs::msg::MarkerArray getMarkers() const;

private:
    rclcpp::Node::SharedPtr node_;
    visualization_msgs::msg::MarkerArray block_markers_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr block_markers_sub_;

    /**
     * @brief Callback function for receiving marker array messages.
     * 
     * @param msg Shared pointer to the received marker array message.
     */
    void markerArrayCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);
};

}