import os
import launch
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch.actions import RegisterEventHandler, TimerAction, LogInfo
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node

def generate_launch_description():
    package_dir = get_package_share_directory('pidog_sim')
    robot_description_path = os.path.join(package_dir, 'resource', 'pidog_minimal.urdf')

    # Read URDF content as string (not path!)
    with open(robot_description_path, 'r') as urdf_file:
        robot_description_content = urdf_file.read()

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'pidog_world.wbt'),
    )

    my_robot_driver = WebotsController(
        robot_name='PiDog',
        parameters=[
            {'robot_description': robot_description_content},
        ],
    )

    # Bridge between gait generator and webots motors
    bridge_node = Node(
        package='pidog_control',
        executable='pidog_webots_bridge',
        name='pidog_webots_bridge',
        output='screen'
    )

    return LaunchDescription([
        webots,
        my_robot_driver,
        bridge_node,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])