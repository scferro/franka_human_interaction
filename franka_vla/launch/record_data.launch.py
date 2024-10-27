from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_param_builder import ParameterBuilder
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # Data recorder node
    data_recorder_node = Node(
        package="franka_vla",
        executable="data_recorder_node",
        name="data_recorder_node",
        parameters=[{
            "frequency": 4.0,
        }]
    )

    # Include the RealSense launch file
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_vla'),
                'launch',
                'realsense.launch.py'
            ])
        ]),
    )

    return LaunchDescription([
        realsense_launch,
        data_recorder_node,
    ])