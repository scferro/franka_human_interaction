from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    rate_arg = DeclareLaunchArgument(
        'rate',
        default_value='10.0',
        description='Control loop rate in Hz'
    )

    # Define manipulate blocks node
    manipulate_blocks_node = Node(
        package='franka_control',
        executable='manipulate_blocks',
        name='manipulate_blocks',
        parameters=[{
            'rate': LaunchConfiguration('rate'),
        }],
        output='screen'
    )

    return LaunchDescription([
        # Launch arguments
        rate_arg,
        # Nodes
        manipulate_blocks_node,
    ])