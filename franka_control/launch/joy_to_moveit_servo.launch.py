from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch argument for joy device
        DeclareLaunchArgument(
            'joy_dev',
            default_value='/dev/input/js0',
            description='Joystick device'
        ),

        # Joy node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'dev': LaunchConfiguration('joy_dev'),
                'deadzone': 0.1,
                'autorepeat_rate': 20.0,
            }]
        ),

        # Custom Joy to MoveIt Servo node
        Node(
            package='franka_control',
            executable='joy_to_moveit_servo',
            name='joy_to_moveit_servo',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': False,
                # Add any other parameters your node might need
            }]
        ),
    ])