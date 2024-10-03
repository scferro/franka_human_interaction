from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    servo_yaml = PathJoinSubstitution([FindPackageShare("franka_control"), "config", "franka_servo_config.yaml"])
    
    return LaunchDescription([
        Node(
            package="moveit_servo",
            executable="servo_server",
            name="servo_server",
            output="screen",
            parameters=[servo_yaml],
            arguments=['--ros-args', '--log-level', 'info'],
        ),
    ])