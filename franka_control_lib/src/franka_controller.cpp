#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <array>
#include <iostream>

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/gripper.h>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include "motion_generator.hpp"


namespace franka_control_lib {

    FrankaController::FrankaController(const std::string& ip_address){
        try {
            robot = new franka::Robot(ip_address);
            robot->setCollisionBehavior(
                {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
                {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
                {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
                {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}
            );
            std::cout << "Robot initialized successfully." << std::endl;
        } catch (const franka::Exception& e) {
            std::cerr << "Failed to connect to robot: " << e.what() << std::endl;
            throw;
        }
    }

    int FrankaController::move_to_pose(geometry_msgs::msg::PoseStamped pose) {
        try {
            auto motion_generator = [&desired_pose](const franka::RobotState&, franka::Duration) -> franka::CartesianPose {
                return franka::CartesianPose(desired_pose);
            };

            robot->control(motion_generator, franka::ControllerMode::kCartesianImpedance);
            std::cout << "Motion completed successfully!" << std::endl;
        } catch (const franka::Exception& e) {
            std::cerr << "Exception caught in motion generator: " << e.what() << std::endl;
        }
            return 0;
        }

    int FrankaController::move_at_vel(geometry_msgs::msg::Twist twist) {
        return 0
    }

    int FrankaController::close_gripper_to(double grip_width) 
        return 0;
    }

    int FrankaController::open_gripper() {
        return 0
    }

}