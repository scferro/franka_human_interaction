from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    return LaunchDescription([
        # Launch the Realsense D405 camera node without conditional checks
        IncludeLaunchDescription(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            ),
            launch_arguments={
                "camera_name": "d405",
                "device_type": "d405",
                "pointcloud.enable": "true",
                "enable_depth": "true",
                # "enable_sync": "true",
                "align_depth.enable": "true",
                "spatial_filter.enable": "true",
                "temporal_filter.enable": "true",
                "decimation_filter.enable": "true",
                "depth_module.enable_auto_exposure": "true",
            }.items(),
        ),
        # Launch the Realsense D435i camera node without conditional checks
        IncludeLaunchDescription(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            ),
            launch_arguments={
                "camera_name": "d435i",
                "device_type": "d435i",
                "rgb_camera.profile": "640x480x6",
                "enable_depth": "true",
                # "enable_sync": "true",
                "align_depth.enable": "true",
                "depth_module.enable_auto_exposure": "true",
                "rgb_camera.enable_auto_exposure": "true",
                "pointcloud.enable": "true",
                "spatial_filter.enable": "true",
                "temporal_filter.enable": "true",
                "decimation_filter.enable": "true",
            }.items(),
        ),
    ])
