#include "franka_control/marker_handling.hpp"

namespace franka_control {

MarkerHandling::MarkerHandling(const rclcpp::Node::SharedPtr& node)
    : node_(node)
{
    block_markers_sub_ = node_->create_subscription<visualization_msgs::msg::MarkerArray>(
        "blocks", 10, std::bind(&MarkerHandling::markerArrayCallback, this, std::placeholders::_1));
}

void MarkerHandling::updateMarkers(const visualization_msgs::msg::MarkerArray& markers)
{
    block_markers_ = markers;
}

visualization_msgs::msg::MarkerArray MarkerHandling::getMarkers() const
{
    return block_markers_;
}

void MarkerHandling::markerArrayCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
{
    block_markers_ = *msg;
}

}