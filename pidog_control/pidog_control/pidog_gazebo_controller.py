#!/usr/bin/env python3
"""
Gazebo controller for PiDog - publishes standing pose to ros2_control.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class PiDogGazeboController(Node):
    """Gazebo controller that commands standing pose."""

    def __init__(self):
        super().__init__('pidog_gazebo_controller')

        # Publisher to position controller
        self.position_pub = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        # Joint names (must match URDF and controller config)
        self.joint_names = [
            'body_to_back_right_leg_b',
            'back_right_leg_b_to_a',
            'body_to_from_right_leg_b',
            'front_right_leg_b_to_a',
            'body_to_back_left_leg_b',
            'back_left_leg_b_to_a',
            'body_to_front_left_leg_b',
            'front_left_leg_b_to_a',
        ]

        # Standing pose: shoulders straight (0), knees bent (-0.8 rad = ~45°)
        self.standing_pose = [
            0.0, -0.8,  # Back Right
            0.0, -0.8,  # Front Right
            0.0, -0.8,  # Back Left
            0.0, -0.8,  # Front Left
        ]

        # Current commanded position
        self.current_position = self.standing_pose.copy()

        # Subscribe to gait commands (optional - for future gait control)
        self.create_subscription(
            JointState,
            '/motor_pos',
            self.motor_callback,
            10
        )

        # Publish standing pose at 50Hz
        self.timer = self.create_timer(0.02, self.publish_position)

        self.get_logger().info('PiDog Gazebo Controller started')
        self.get_logger().info(f'Commanding standing pose: {self.standing_pose}')

    def motor_callback(self, msg):
        """Receive joint commands from gait generator."""
        if len(msg.position) == len(self.joint_names):
            self.current_position = list(msg.position)
            self.get_logger().debug('Received new joint positions from gait controller')

    def publish_position(self):
        """Publish current position to ros2_control."""
        msg = Float64MultiArray()
        msg.data = self.current_position
        self.position_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    controller = PiDogGazeboController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
