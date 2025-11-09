"""
Gazebo Launch File Optimized for Fast RL Training

Key optimizations:
- Headless mode (no GUI rendering)
- Unlimited real-time factor (runs as fast as CPU allows)
- Reduced physics solver iterations (50 vs 300)
- Larger physics step size (5ms vs 1ms)
- No shadows or visual rendering
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.actions import IncludeLaunchDescription
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    resources_package = 'pidog_description'

    # Make path to resources dir without last package_name fragment.
    path_to_share_dir_clipped = ''.join(get_package_share_directory(resources_package).rsplit('/' + resources_package, 1))

    # Gazebo hint for resources.
    os.environ['GZ_SIM_RESOURCE_PATH'] = path_to_share_dir_clipped

    # Ensure `SDF_PATH` is populated since `sdformat_urdf` uses this rather
    # than `GZ_SIM_RESOURCE_PATH` to locate resources.
    if "GZ_SIM_RESOURCE_PATH" in os.environ:
        gz_sim_resource_path = os.environ["GZ_SIM_RESOURCE_PATH"]

        if "SDF_PATH" in os.environ:
            sdf_path = os.environ["SDF_PATH"]
            os.environ["SDF_PATH"] = sdf_path + ":" + gz_sim_resource_path
        else:
            os.environ["SDF_PATH"] = gz_sim_resource_path

    # Gazebo Sim.
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Path to FAST RL training world file
    world_file = os.path.join(
        get_package_share_directory(resources_package),
        'worlds',
        'pidog_rl_fast.sdf'
    )

    # Launch Gazebo HEADLESS with fast world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        # -s: headless (no GUI), -r: run on start, -v 1: minimal verbosity
        launch_arguments=dict(gz_args=f'-s -r -v 1 {world_file}').items(),
    )

    # Spawn robot with stand pose
    spawn = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'Robot.urdf',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.12',
                '-topic', '/robot_description',
                # Stand pose joint positions
                '-J', 'body_to_back_right_leg_b', '-1.208',
                '-J', 'back_right_leg_b_to_a', '0.180',
                '-J', 'body_to_front_right_leg_b', '-1.208',
                '-J', 'front_right_leg_b_to_a', '0.180',
                '-J', 'body_to_back_left_leg_b', '1.208',
                '-J', 'back_left_leg_b_to_a', '-0.180',
                '-J', 'body_to_front_left_leg_b', '1.208',
                '-J', 'front_left_leg_b_to_a', '-0.180',
                '-J', 'motor_8_to_tail', '0.0',
                '-J', 'neck1_to_motor_9', '0.0',
                '-J', 'neck2_to_motor_10', '0.0',
                '-J', 'neck3_to_motor_11', '0.0',
            ],
            output='screen',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_launch_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    # Robot state publisher (no RViz for RL training)
    robot_state_publisher = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare(resources_package),
                    'launch',
                    'description.launch.py',
                ]),
            ]),
            launch_arguments=dict(use_sim_time=use_sim_time).items(),
    )

    # Load joint_state_broadcaster
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    # Load position_controller
    load_position_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['position_controller'],
        output='screen',
    )

    # Bridge clock from Gazebo to ROS 2
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    # Bridge IMU sensor data
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/imu@sensor_msgs/msg/Imu@gz.msgs.IMU'],
        output='screen',
    )

    # Bridge Camera sensor for vision-based RL
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/camera@sensor_msgs/msg/Image@gz.msgs.Image'],
        output='screen',
    )

    # Bridge Ultrasonic sensor (GPU Lidar simulating HC-SR04)
    ultrasonic_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/ultrasonic@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'],
        output='screen',
    )

    # Delay controller spawning until after robot is spawned
    # This prevents race condition in headless mode
    delayed_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn,
            on_exit=[
                TimerAction(
                    period=2.0,  # Wait 2 seconds after spawn completes
                    actions=[load_joint_state_broadcaster]
                )
            ],
        )
    )

    delayed_position_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn,
            on_exit=[
                TimerAction(
                    period=3.0,  # Wait 3 seconds (after joint_state_broadcaster)
                    actions=[load_position_controller]
                )
            ],
        )
    )

    return LaunchDescription([
        use_sim_time_launch_arg,
        robot_state_publisher,
        gazebo,
        clock_bridge,
        imu_bridge,
        camera_bridge,
        ultrasonic_bridge,
        spawn,
        delayed_joint_state_broadcaster,  # Wait for spawn to finish
        delayed_position_controller,      # Wait for spawn to finish
    ])