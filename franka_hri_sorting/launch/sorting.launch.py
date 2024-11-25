from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    franka_hri_sorting_dir = get_package_share_directory('franka_hri_sorting')
    franka_moveit_config_dir = get_package_share_directory('franka_moveit_config')

    # Define paths for model saving/loading
    default_models_dir = os.path.join(franka_hri_sorting_dir, 'models')
    default_save_dir = default_models_dir
    rviz_config_file = os.path.join(franka_hri_sorting_dir, 'config', 'sorting.rviz')
    
    # Ensure directories exist
    os.makedirs(default_models_dir, exist_ok=True)
    os.makedirs(default_save_dir, exist_ok=True)

    # Mode selection argument
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='robot',
        description='Launch mode: "robot" or "images"'
    )

    # Training mode argument
    training_mode_arg = DeclareLaunchArgument(
        'training_mode',
        default_value='sorting_only',
        description='Training mode: "sorting_only", "gestures_only", or "both"'
    )

    # Standard arguments
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

    # Training images directory argument
    training_images_arg = DeclareLaunchArgument(
        'training_images_path',
        default_value="/home/scferro/Documents/final_project/training_images",
        description='Path to training images directory'
    )

    # Define robot-mode nodes
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_moveit_config_dir, 'launch', 'rviz.launch.py')
        ),
        launch_arguments={
            'robot_ip': 'panda0.robot',
            'rviz_config': rviz_config_file
        }.items(),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'robot'"]))
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_hri_sorting_dir, 'launch', 'realsense.launch.py')
        ),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'robot'"]))
    )

    blocks_node = Node(
        package='franka_hri_sorting',
        executable='blocks',
        name='blocks',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'robot'"]))
    )

    # Define shared nodes
    human_input_node = Node(
        package='franka_hri_sorting',
        executable='human_input',
        name='human_input'
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

    # Define image-mode node
    network_training_node = Node(
        package='franka_hri_sorting',
        executable='network_training',
        name='network_training',
        parameters=[{
            'training_images_path': LaunchConfiguration('training_images_path'),
            'training_mode': LaunchConfiguration('training_mode'),
            'display_time': 2.0,
            'gesture_warning_time': 3.0,
            'prediction_timeout': 5.0
        }],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'images'"]))
    )

    # Create and return launch description
    return LaunchDescription([
        # Launch arguments
        mode_arg,
        training_mode_arg,
        sorting_model_path_arg,
        gesture_model_path_arg,
        save_directory_arg,
        buffer_size_arg,
        sequence_length_arg,
        training_images_arg,
        # Nodes and includes
        rviz_launch,
        realsense_launch,
        blocks_node,
        network_node,
        human_input_node,
        network_training_node,
    ])