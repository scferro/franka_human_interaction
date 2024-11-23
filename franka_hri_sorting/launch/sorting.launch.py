from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import OpaqueFunction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    franka_hri_sorting_dir = get_package_share_directory('franka_hri_sorting')
    franka_moveit_config_dir = get_package_share_directory('franka_moveit_config')

    # Define paths for model saving/loading
    default_models_dir = os.path.join(franka_hri_sorting_dir, 'models')
    default_save_dir = default_models_dir
    
    # Ensure directories exist
    os.makedirs(default_models_dir, exist_ok=True)
    os.makedirs(default_save_dir, exist_ok=True)

    # Declare launch arguments with defaults
    sorting_model_path_arg = DeclareLaunchArgument(
        'sorting_model_path',
        default_value='',
        description='Path to pretrained sorting model'
    )

    gesture_model_path_arg = DeclareLaunchArgument(
        'gesture_model_path',
        default_value='',
        description='Path to pretrained gesture model'
    )

    save_directory_arg = DeclareLaunchArgument(
        'save_directory',
        default_value=str(default_save_dir),
        description='Directory to save trained models'
    )

    buffer_size_arg = DeclareLaunchArgument(
        'buffer_size',
        default_value='25',
        description='Size of the training buffer'
    )

    sequence_length_arg = DeclareLaunchArgument(
        'sequence_length',
        default_value='20',
        description='Length of gesture sequences'
    )

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
        name='blocks'
    )

    network_node = Node(
        package='franka_hri_sorting',
        executable='network_node',
        name='network_node',
        parameters=[{
            'sorting_model_path': LaunchConfiguration('sorting_model_path'),
            'gesture_model_path': LaunchConfiguration('gesture_model_path'),
            'save_directory': LaunchConfiguration('save_directory'),
            'buffer_size': LaunchConfiguration('buffer_size'),
            'sequence_length': LaunchConfiguration('sequence_length'),
        }],
        output='screen'
    )

    # Create and return launch description
    return LaunchDescription([
        # Launch arguments
        sorting_model_path_arg,
        gesture_model_path_arg,
        save_directory_arg,
        buffer_size_arg,
        sequence_length_arg,
        # Nodes and includes
        rviz_launch,
        realsense_launch,
        blocks_node,
        network_node,
    ])