/// \file control_servos.cpp
/// \brief Controls the steering servo and ESC via a PCA 9685 over I2C
///
/// PARAMETERS:
///     rate (double): the publishing rate for wheel speed messages
/// SERVERS:
///     move_robot (std_srvs::srv::Empty): enables/disables drive motor

#include <chrono>
#include <memory>
#include <string>
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstring>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

using namespace std::chrono_literals;

class Plates : public rclcpp::Node
{
public:
  Plates()
  : Node("plates")
  {
    // Parameters and default values
    declare_parameter("rate", 200.);

    // Define parameter variables
    loop_rate = get_parameter("rate").as_double();

    // Servers
    grab_plate_srv = create_service<std_srvs::srv::Empty>(
      "grab_plate",
      std::bind(&Plates::grab_plate_callback, this, std::placeholders::_1, std::placeholders::_2));

    // Main timer
    int cycle_time = 1000.0 / loop_rate;
    main_timer = this->create_wall_timer(
      std::chrono::milliseconds(cycle_time),
      std::bind(&Plates::timer_callback, this));
  }

private:
  // Initialize parameter variables
  int rate;
  double loop_rate;

  // Initialize subscriptions and timer
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr grab_plate_srv;
  rclcpp::TimerBase::SharedPtr main_timer;

  /// \brief The main timer callback, 
  void timer_callback()
  {
    int something = 0;
  }

  /// \brief Callback for the grab_plate server, moves robot
  void grab_plate_callback(
    std_srvs::srv::Empty::Request::SharedPtr,
    std_srvs::srv::Empty::Response::SharedPtr)
  {
    
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Plates>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
