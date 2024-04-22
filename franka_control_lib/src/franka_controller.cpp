#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "geometry_msgs/msg/pose_stamped.hpp"


namespace franka_control_lib {

    FrankaController::FrankaController()
    : {}

    int FrankaController::move_to_pose(geometry_msgs::msg::PoseStamped pose) {
        return 0
    }

    int FrankaController::close_gripper_to(double grip_width) {
        return 0
    }

    int FrankaController::open_gripper() {
        return 0
    }

}