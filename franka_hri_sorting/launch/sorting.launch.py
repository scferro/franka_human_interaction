from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    franka_hri_sorting_dir = get_package_share_directory('franka_hri_sorting')
    franka_moveit_config_dir = get_package_share_directory('franka_moveit_config')

    # Define launch actions
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_moveit_config_dir, 'launch', 'rviz.launch.py')
        ),
        launch_arguments={'robot_ip': 'panda0.robot'}.items()
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_hri_sorting_dir, 'launch', 'realsense.launch.py')
        )
    )

    blocks_node = Node(
        package='franka_hri_sorting',
        executable='blocks',
        name='blocks_node'
    )

    human_input_node = Node(
        package='franka_hri_sorting',
        executable='human_input',
        name='human_input_node'
    )

    manipulate_blocks_node = Node(
        package='franka_control',
        executable='manipulate_blocks',
        name='manipulate_blocks_node'
    )

    # Create and return launch description
    return LaunchDescription([
        rviz_launch,
        realsense_launch,
        blocks_node,
        human_input_node,
        manipulate_blocks_node
    ])