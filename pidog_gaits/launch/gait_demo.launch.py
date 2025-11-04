"""Launch file for demonstrating gaits with Webots simulator."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Launch Webots simulator with gait generator.

    This demonstrates the traditional (non-neural network) gaits.
    """

    pidog_sim_dir = get_package_share_directory('pidog_sim')

    return LaunchDescription([
        # Launch Webots simulator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pidog_sim_dir, 'launch', 'pidog_launch.py')
            )
        ),

        # Launch gait generator (replaces the simple sit controller)
        Node(
            package='pidog_gaits',
            executable='gait_generator',
            name='gait_generator',
            output='screen',
            parameters=[{
                'frequency': 30,
                'default_gait': 'stand'
            }]
        ),
    ])
