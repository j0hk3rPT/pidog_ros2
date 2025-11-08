import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
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

    # Path to custom world file with optimized physics settings
    world_file = os.path.join(
        get_package_share_directory(resources_package),
        'worlds',
        'pidog.sdf'
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments=dict(gz_args=f'-r {world_file} --verbose').items(),
    )

    # Spawn with initial joint positions EXACTLY matching IK-generated stand pose
    # Stand pose from IK: shoulders=±1.208 rad, knees=±0.180 rad
    # This prevents violent transitions when gait_generator starts
    # IMPORTANT: Left legs have flipped joint axes due to 180° rotation in URDF
    # Right legs: negative shoulder/positive knee
    # Left legs: positive shoulder/negative knee
    spawn = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'Robot.urdf',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.12',  # Spawn higher to prevent ground penetration (12cm)
                '-topic', '/robot_description',
                # Set initial joint positions to IK-generated stand pose
                '-J', 'body_to_back_right_leg_b', '-1.208',   # BR shoulder
                '-J', 'back_right_leg_b_to_a', '0.180',       # BR knee
                '-J', 'body_to_front_right_leg_b', '-1.208',  # FR shoulder
                '-J', 'front_right_leg_b_to_a', '0.180',      # FR knee
                '-J', 'body_to_back_left_leg_b', '1.208',     # BL shoulder (flipped)
                '-J', 'back_left_leg_b_to_a', '-0.180',       # BL knee (flipped)
                '-J', 'body_to_front_left_leg_b', '1.208',    # FL shoulder (flipped)
                '-J', 'front_left_leg_b_to_a', '-0.180',      # FL knee (flipped)
                '-J', 'motor_8_to_tail', '0.0',               # Tail neutral
                '-J', 'neck1_to_motor_9', '0.0',              # Head neutral
                '-J', 'neck2_to_motor_10', '0.0',
                '-J', 'neck3_to_motor_11', '0.0',
            ],
            output='screen',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    use_sim_time_launch_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    use_rviz = LaunchConfiguration('use_rviz')

    use_rviz_arg = DeclareLaunchArgument("use_rviz", default_value='true')

    robot_state_publisher = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare(resources_package),
                    'launch',
                    'description.launch.py',
                ]),
            ]),
            condition=UnlessCondition(use_rviz),  # rviz launch includes rsp.
            launch_arguments=dict(use_sim_time=use_sim_time).items(),
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare(resources_package),
                'launch',
                'display.launch.py',
            ]),
        ]),
        condition=IfCondition(use_rviz),
        launch_arguments=dict(
            gui='false',  # Disable joint_state_publisher_gui - use Gazebo's joint states
            use_sim_time=use_sim_time,
        ).items(),
    )

    # Load joint_state_broadcaster
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    # Load position_controller - now with balanced gains
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

    # Bridge IMU sensor data from Gazebo to ROS 2
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/imu@sensor_msgs/msg/Imu@gz.msgs.IMU'],
        output='screen',
    )

    # Bridge Camera sensor data from Gazebo to ROS 2
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

    # Bridge Touch sensor (contact sensor) - single sensor behind eyes
    touch_sensor_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/touch_sensor/contacts@gazebo_msgs/msg/Contacts@gz.msgs.Contacts'],
        output='screen',
    )

    # Gazebo controller to hold robot in standing pose
    gazebo_controller = Node(
        package='pidog_control',
        executable='pidog_gazebo_controller',
        name='pidog_gazebo_controller',
        output='screen',
    )

    return LaunchDescription([
        use_sim_time_launch_arg,
        use_rviz_arg,
        robot_state_publisher,
        rviz,
        gazebo,
        clock_bridge,           # Bridge clock from Gazebo
        imu_bridge,             # Bridge IMU sensor from Gazebo
        camera_bridge,          # Bridge camera sensor from Gazebo
        ultrasonic_bridge,      # Bridge ultrasonic/lidar sensor from Gazebo (eyes)
        touch_sensor_bridge,    # Bridge touch sensor from Gazebo (behind eyes)
        spawn,
        load_joint_state_broadcaster,
        load_position_controller,
        gazebo_controller,
    ])
