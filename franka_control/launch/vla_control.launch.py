from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_param_builder import ParameterBuilder
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Declare the use_rviz argument
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    # VLA control node
    vla_control_node = Node(
        package="franka_control",
        executable="vla_control_node",
        name="vla_control_node",
        parameters=[{
            "frequency": 4.0,
            "linear_scale": 10.0,
            "angular_scale": 4.0,
            "vla_enabled": True,
            "observation_buffer_size": 2,
            "action_timeout": 100.0,
        }]
    )

    # VLA service node
    franka_vla_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('franka_vla'), 'launch', 'franka_vla.launch.py')
        ]),
        launch_arguments={
            'example_arg': LaunchConfiguration('use_rviz')
        }.items()
    )

    return LaunchDescription([
        use_rviz_arg,
        vla_control_node,
        franka_vla_launch,
    ])