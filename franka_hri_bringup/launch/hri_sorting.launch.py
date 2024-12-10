from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    franka_hri_sorting_dir = get_package_share_directory('franka_hri_sorting')
    franka_control_dir = get_package_share_directory('franka_control')

    # Load sorting launch file
    sorting_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_hri_sorting_dir, 'launch', 'sorting.launch.py')
        ),
    )

    # Load human interaction launch file
    interaction_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_hri_sorting_dir, 'launch', 'human_interaction.launch.py')
        ),
    )

    # Load blocks_control launch file
    franka_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_control_dir, 'launch', 'blocks_control.launch.py')
        ),
    )

    # Load the sensor_publisher node
    sensor_node = Node(
        package='sensor_ros_bridge',
        executable='sensor_publisher',
        name='sensor_publisher',
    )


    # Create and return launch description
    return LaunchDescription([
        sorting_launch,
        interaction_launch,
        franka_control_launch,
        sensor_node,
    ])