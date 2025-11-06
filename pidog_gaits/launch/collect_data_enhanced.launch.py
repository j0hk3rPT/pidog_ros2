"""Launch file for collecting enhanced training data with observation noise."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Launch Gazebo + Gait Generator + Enhanced Data Collector.

    This collects training data with observation noise for robust sim-to-real transfer.
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

        # Launch enhanced data collector with observation noise
        Node(
            package='pidog_gaits',
            executable='data_collector_enhanced',
            name='data_collector_enhanced',
            output='screen',
            parameters=[{
                'output_dir': './training_data',
                'collect_duration': 60.0,  # 60 seconds per gait
                'add_noise': True,  # Enable observation noise
                'position_noise_std': 0.01,  # ~0.57° position noise
                'velocity_noise_std': 0.1,  # Velocity noise
                'collect_velocities': True  # For LSTM training
            }]
        ),
    ])
