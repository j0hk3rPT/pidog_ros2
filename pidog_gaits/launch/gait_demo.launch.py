"""Launch file for demonstrating gaits with Gazebo simulator."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Launch Gazebo simulator with gait generator.

    This demonstrates the traditional (non-neural network) gaits.
    """

    pidog_description_dir = get_package_share_directory('pidog_description')

    return LaunchDescription([
        # Launch Gazebo simulator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pidog_description_dir, 'launch', 'gazebo.launch.py')
            )
        ),

        # Launch gait generator (replaces the simple controller)
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
