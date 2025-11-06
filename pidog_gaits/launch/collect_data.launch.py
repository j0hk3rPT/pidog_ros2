"""Launch file for collecting training data."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Launch Gazebo + Gait Generator + Data Collector.

    This collects training data by running various gaits and recording the joint angles.
    """

    pidog_description_dir = get_package_share_directory('pidog_description')

    return LaunchDescription([
        # Launch Gazebo simulator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pidog_description_dir, 'launch', 'gazebo.launch.py')
            )
        ),

        # Launch gait generator
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

        # Launch data collector
        Node(
            package='pidog_gaits',
            executable='data_collector',
            name='data_collector',
            output='screen',
            parameters=[{
                'output_dir': './training_data',
                'collect_duration': 60.0  # 60 seconds per gait
            }]
        ),
    ])
