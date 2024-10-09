from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_param_builder import ParameterBuilder
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("numsr_franka")
        .robot_description(file_path="config/panda_arm_real.urdf.xacro")
        .robot_description_semantic(file_path="config/panda_arm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/panda_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    # Declare the use_rviz argument
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    # RViz node
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("franka_control"), "config", "franka_hri_servo.rviz"])
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        condition=IfCondition(LaunchConfiguration("use_rviz"))
    )

    # VLA node
    teleop_control_node = Node(
        package="franka_vla",
        executable="franka_octo",
        name="franka_octo",
        parameters=[{
            "inference_frequency": 10.0,
            "servo_frequency": 10.0,
            "buffer_size": 5,
        }]
    )

    return LaunchDescription([
        use_rviz_arg,
        rviz_node,
        teleop_control_node,
    ])