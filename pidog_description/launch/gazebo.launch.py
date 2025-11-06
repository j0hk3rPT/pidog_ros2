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

    # Spawn with initial joint positions matching standing pose
    # Height calculated from leg kinematics: 0.055m = body center height with knees bent at 0.8 rad
    # IMPORTANT: Left legs have flipped joint axes due to 180° rotation in URDF
    # Right legs: negative angle bends knee DOWN
    # Left legs: positive angle bends knee DOWN
    spawn = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'Robot.urdf',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.055',  # Precise height for standing pose (5.5cm)
                '-topic', '/robot_description',
                # Set initial joint positions to standing pose (symmetric!)
                '-J', 'body_to_back_right_leg_b', '0.0',
                '-J', 'back_right_leg_b_to_a', '-0.8',   # Right: negative = down
                '-J', 'body_to_from_right_leg_b', '0.0',
                '-J', 'front_right_leg_b_to_a', '-0.8',  # Right: negative = down
                '-J', 'body_to_back_left_leg_b', '0.0',
                '-J', 'back_left_leg_b_to_a', '0.8',     # Left: positive = down (axis flipped!)
                '-J', 'body_to_front_left_leg_b', '0.0',
                '-J', 'front_left_leg_b_to_a', '0.8',    # Left: positive = down (axis flipped!)
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
        clock_bridge,  # Add clock bridge before spawning robot
        spawn,
        load_joint_state_broadcaster,
        load_position_controller,
        gazebo_controller,
    ])
