/// \file
/// \brief Class for controlling the Franka via libfranka


#include<iosfwd> // contains forward definitions for iostream objects
#include<vector>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "motion_generator.hpp"

#include 

namespace franka_control_lib
{
    /// \brief a representation of a differential drive robot
    class FrankaController
    {
    private:
        /// \brief the franka robot object
        franka::Robot* robot;
    public:
        /// \brief Create a new FrankaController object
        explicit FrankaController(const std::string& ip_address);

        /// \brief Move the gripper to a specified pose
        /// \param goal_pose the desired pose of the robot, in the franka base frame
        int move_to_pose(geometry_msgs::msg::PoseStamped pose);

        /// \brief Move the gripper to a specified pose
        /// \param goal_pose the desired pose of the robot, in the franka base frame
        int move_at_vel(geometry_msgs::msg::Twist twist);

        /// \brief Close the gripper to hold an object of a specified width
        /// \param grip_width the desired width to grip
        int close_gripper_to(double grip_width);

        /// \brief Open the gripper
        int open_gripper();

        /// \brief Return the current cartesian pose of the robot
        geometry_msgs::msg::PoseStamped pose get_pose();
    };
}

#endif