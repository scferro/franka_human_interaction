from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
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
        condition=IfCondition(LaunchConfiguration("use_rviz"))
    )

    # Joy node
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        parameters=[{
            "dev": "/dev/input/js0",
            "deadzone": 0.05,
            "autorepeat_rate": 20.0,
        }]
    )

    # Teleop control node
    teleop_control_node = Node(
        package="franka_control",
        executable="teleop_control_node",
        name="teleop_control_node",
        parameters=[{
            "frequency": 10.0,
            "linear_scale": 1.0,
            "angular_scale": 1.0,
            "teleop_enabled": True,
        }]
    )

    return LaunchDescription([
        use_rviz_arg,
        rviz_node,
        joy_node,
        teleop_control_node,
    ])