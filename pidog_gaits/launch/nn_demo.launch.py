"""Launch file for demonstrating neural network controller."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Launch Webots simulator with neural network controller.

    This uses the trained model to generate gaits.
    """

    pidog_sim_dir = get_package_share_directory('pidog_sim')

    return LaunchDescription([
        # Launch Webots simulator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pidog_sim_dir, 'launch', 'pidog_launch.py')
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
                'model_type': 'simple',
                'frequency': 30,
                'device': 'cpu'
            }]
        ),
    ])
