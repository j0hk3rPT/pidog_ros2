"""Launch file for demonstrating neural network controller."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Launch Gazebo simulator with neural network controller.

    This uses the trained model to generate gaits.
    """

    pidog_description_dir = get_package_share_directory('pidog_description')

    return LaunchDescription([
        # Launch Gazebo simulator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pidog_description_dir, 'launch', 'gazebo.launch.py')
            )
        ),

        # Launch neural network controller
        Node(
            package='pidog_gaits',
            executable='nn_controller',
            name='nn_controller',
            output='screen',
            parameters=[{
                'model_path': './models/best_model.pth',
                'model_type': 'simple_lstm',
                'frequency': 30,
                'device': 'cpu'
            }]
        ),
    ])
