from setuptools import setup
import os
from glob import glob

package_name = 'pidog_gaits'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Install scripts to lib/pidog_gaits for ROS 2 launch compatibility
        (os.path.join('lib', package_name), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='PiDog gait generation and neural network training',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gait_generator = pidog_gaits.gait_generator_node:main',
            'data_collector = pidog_gaits.data_collector:main',
            'data_collector_enhanced = pidog_gaits.data_collector_enhanced:main',
            'nn_controller = pidog_gaits.nn_controller:main',
        ],
    },
)
