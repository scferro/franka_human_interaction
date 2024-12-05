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
    franka_moveit_config_dir = get_package_share_directory('franka_moveit_config')

    # Define all default directories
    default_base_dir = '/home/scferro/Documents/final_project/hri_data'
    default_models_dir = os.path.join(default_base_dir, 'models')
    default_logs_dir = os.path.join(default_base_dir, 'logs')
    default_training_images = os.path.join(default_base_dir, 'training_images')
    rviz_config_file = os.path.join(franka_hri_sorting_dir, 'config', 'sorting.rviz')
    
    # Ensure all required directories exist
    for directory in [default_models_dir, default_logs_dir, default_training_images]:
        os.makedirs(directory, exist_ok=True)

    # System mode arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='robot',
        description='Launch mode: "robot" or "images"'
    )

    training_mode_arg = DeclareLaunchArgument(
        'training_mode',
        default_value='sorting_only',
        description='Training mode: "sorting_only", "gestures_only", or "both"'
    )

    gesture_network_type_arg = DeclareLaunchArgument(
        'gesture_network_type',
        default_value='binary',
        description='Gesture network type: "binary" or "complex"'
    )

    # Model path arguments
    complex_sorting_model_path_arg = DeclareLaunchArgument(
        'complex_sorting_model_path',
        default_value='',
        description='Path to complex sorting network model'
    )

    binary_gesture_model_path_arg = DeclareLaunchArgument(
        'binary_gesture_model_path',
        default_value='',
        description='Path to binary gesture network model'
    )

    complex_gesture_model_path_arg = DeclareLaunchArgument(
        'complex_gesture_model_path',
        default_value='',
        description='Path to complex gesture network model'
    )

    # Directory arguments
    save_directory_arg = DeclareLaunchArgument(
        'save_directory',
        default_value=str(default_models_dir),
        description='Directory to save trained models'
    )

    log_directory_arg = DeclareLaunchArgument(
        'log_directory',
        default_value=str(default_logs_dir),
        description='Directory to save network logs'
    )

    training_images_arg = DeclareLaunchArgument(
        'training_images_path',
        default_value=str(default_training_images),
        description='Path to training images directory'
    )

    # Network configuration arguments
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

    # Static transforms 
    world_to_panda = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_panda_publisher',
        arguments=['0', '0', '-0.02', '0', '0', '0', '1', 'world', 'panda_link0']
    )

    panda_hand_to_d405 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='hand_to_d405_publisher',
        arguments=['0.07', '0', '0.055', '0.7071068', '0', '0.7071068', '0', 'panda_hand', 'd405_link']
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

    # Network node with correctly named model paths
    network_node = Node(
        package='franka_hri_sorting',
        executable='network_node',
        name='network_node',
        parameters=[{
            # Model paths with updated names
            'complex_sorting_model_path': LaunchConfiguration('complex_sorting_model_path'),
            'binary_gesture_model_path': LaunchConfiguration('binary_gesture_model_path'),
            'complex_gesture_model_path': LaunchConfiguration('complex_gesture_model_path'),
            
            # Directories
            'save_directory': LaunchConfiguration('save_directory'),
            'log_directory': LaunchConfiguration('log_directory'),
            
            # Network configuration
            'buffer_size': LaunchConfiguration('buffer_size'),
            'sequence_length': LaunchConfiguration('sequence_length')
        }],
        output='screen'
    )

    # Training node
    network_training_node = Node(
        package='franka_hri_sorting',
        executable='network_training',
        name='network_training',
        parameters=[{
            'training_images_path': LaunchConfiguration('training_images_path'),
            'training_mode': LaunchConfiguration('training_mode'),
            'gesture_network_type': LaunchConfiguration('gesture_network_type'),
            'display_time': 2.0,
            'gesture_warning_time': 3.0,
            'prediction_timeout': 5.0
        }],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'images'"]))
    )

    # Human input node
    human_input_node = Node(
        package='franka_hri_sorting',
        executable='human_input',
        name='human_input',
        parameters=[{
            # Configure input mode based on training settings
            'training_mode': LaunchConfiguration('training_mode'),
            'gesture_network_type': LaunchConfiguration('gesture_network_type'),
            'display_time': 2.0
        }]
    )

    # Create and return launch description
    return LaunchDescription([
        # Launch arguments
        mode_arg,
        training_mode_arg,
        gesture_network_type_arg,
        
        # Model paths with updated names
        complex_sorting_model_path_arg,
        binary_gesture_model_path_arg,
        complex_gesture_model_path_arg,
        
        # Directories
        save_directory_arg,
        log_directory_arg,
        training_images_arg,
        
        # Configuration
        buffer_size_arg,
        sequence_length_arg,
        
        # Nodes and includes
        rviz_launch,
        realsense_launch,
        blocks_node,
        network_node,
        human_input_node,
        network_training_node,
        world_to_panda,
        panda_hand_to_d405,
    ])