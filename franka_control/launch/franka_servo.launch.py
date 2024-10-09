import os 
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, Shutdown, DeclareLaunchArgument, IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch_param_builder import ParameterBuilder
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable, Command, AndSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("numsr_franka")
        .robot_description(file_path="config/panda_arm_real.urdf.xacro")
        .robot_description_semantic(file_path="config/panda_arm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/panda_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    # Get parameters for the Servo node
    servo_params = {
        "moveit_servo": ParameterBuilder("franka_control")
        .yaml("config/panda_config.yaml")
        .to_dict()
    }

    # This filter parameter should be >1. Increase it for greater smoothing but slower motion.
    low_pass_filter_coeff = {"butterworth_filter_coeff": 3.0}

    # Load controllers
    load_controllers = []
    for controller in [
        "joint_state_broadcaster",
        "panda_arm_controller",
    ]:
        load_controllers += [
            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner {}".format(
                    controller)],
                shell=True,
                output="screen",
            )
        ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(name="use_fake_hardware", default_value="false",
                                  description="whether or not to use fake hardware."),
            DeclareLaunchArgument(name="use_rviz", default_value="true",
                                  description="whether or not to use rviz."),
            DeclareLaunchArgument(name="robot_ip", default_value="dont-care",
                                  description="IP address of the robot"),
            DeclareLaunchArgument(
                'acceleration_filter_update_period',
                default_value='0.01',
                description='Update period for acceleration limiting filter'
            ),
            DeclareLaunchArgument(
                'planning_group_name',
                default_value='panda_arm',
                description='Planning group name'
            ),
            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner joint_state_broadcaster"],
                shell=True,
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                condition=UnlessCondition(
                    LaunchConfiguration("use_fake_hardware")),
                remappings=[('joint_states', 'franka/joint_states')],
                parameters=[moveit_config.robot_description, PathJoinSubstitution([
                    FindPackageShare(
                        "numsr_franka_moveit_config"), "config", "panda_ros_controllers.yaml"
                ])],
                output="both",
            ),
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher',
                parameters=[
                    {'source_list': ['franka/joint_states', 'panda_gripper/joint_states'], 'rate': 30}],
                ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                on_exit=Shutdown(),
                output="log",
                arguments=["--frame-id", "world",
                           "--child-frame-id", "panda_link0"],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                condition=UnlessCondition(LaunchConfiguration("use_fake_hardware")),
                output="both",
                parameters=[moveit_config.robot_description],
            ),

            # ServoControlNode
            Node(
                package='franka_control',
                executable='servo_control_node',
                name='servo_control_node',
                output='screen',
                emulate_tty=True,
                parameters=[
                    servo_params,
                    {'update_period': LaunchConfiguration('acceleration_filter_update_period')},
                    {'planning_group_name': LaunchConfiguration('planning_group_name')},
                    low_pass_filter_coeff,
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                ],
            ),

            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner panda_arm_controller"],
                shell=True,
                output="screen",
            ),
        ]
        # + load_controllers
    )
