from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare('franka_hri_sorting')
    
    # Declare arguments
    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'd435_calibration.csv']),
        description='Path to camera calibration file'
    )
    
    calibration_timeout_arg = DeclareLaunchArgument(
        'calibration_timeout',
        default_value='5.0',
        description='Timeout for camera calibration'
    )

    # Get config file paths
    d405_config = PathJoinSubstitution([pkg_share, 'config', 'apriltag_d405_config.yaml'])
    d435_config = PathJoinSubstitution([pkg_share, 'config', 'apriltag_d435_config.yaml'])

    # Launch AprilTag nodes
    apriltag_d405 = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_d405',
        remappings=[
            ('image_rect', '/camera/d405/color/image_rect_raw'),
            ('camera_info', '/camera/d405/color/camera_info'),
        ],
        parameters=[d405_config]
    )
    
    apriltag_d435 = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_d435',
        remappings=[
            ('image_rect', '/camera/d435i/color/image_raw'),
            ('camera_info', '/camera/d435i/color/camera_info'),
        ],
        parameters=[d435_config]
    )
    
    # Launch Human Interaction node
    human_interaction_node = Node(
        package='franka_hri_sorting',
        executable='human_interaction',
        name='human_interaction',
        parameters=[{
            'calibration_file': LaunchConfiguration('calibration_file'),
            'calibration_timeout': LaunchConfiguration('calibration_timeout')
        }]
    )
    
    return LaunchDescription([
        calibration_file_arg,
        calibration_timeout_arg,
        apriltag_d405,
        apriltag_d435,
        human_interaction_node
    ])