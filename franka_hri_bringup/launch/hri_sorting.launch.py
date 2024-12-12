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

    # Define model path arguments
    sorting_model_path_arg = DeclareLaunchArgument(
        'sorting_model_path',
        default_value='',
        description='Path to pretrained sorting model'
    )

    binary_gesture_model_path_arg = DeclareLaunchArgument(
        'binary_gesture_model_path',
        default_value='',
        description='Path to binary (simple) gesture model'
    )

    complex_gesture_model_path_arg = DeclareLaunchArgument(
        'complex_gesture_model_path',
        default_value='',
        description='Path to complex gesture model'
    )

    # Load sorting launch file with model path arguments
    sorting_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_hri_sorting_dir, 'launch', 'sorting.launch.py')
        ),
        launch_arguments={
            'sorting_model_path': LaunchConfiguration('sorting_model_path'),
            'binary_gesture_model_path': LaunchConfiguration('binary_gesture_model_path'),
            'complex_gesture_model_path': LaunchConfiguration('complex_gesture_model_path'),
        }.items()
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
        # Launch arguments for model paths and network type
        sorting_model_path_arg,
        binary_gesture_model_path_arg,
        complex_gesture_model_path_arg,
        
        # Launch files and nodes
        sorting_launch,
        interaction_launch,
        franka_control_launch,
        sensor_node,
    ])