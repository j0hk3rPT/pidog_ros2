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
            'body_to_front_right_leg_b',
            'front_right_leg_b_to_a',
            'body_to_back_left_leg_b',
            'back_left_leg_b_to_a',
            'body_to_front_left_leg_b',
            'front_left_leg_b_to_a',
            # Temporarily disable head/tail to test
            # 'motor_8_to_tail',
            # 'neck1_to_motor_9',
            # 'neck2_to_motor_10',
            # 'neck3_to_motor_11',
        ]

        # Standing pose: shoulders straight (0), knees bent (~45°)
        # NOTE: Left legs have flipped joint axes (rpy="0 1.57 3.1415" in URDF)
        # Right legs: negative angle bends knee DOWN
        # Left legs: positive angle bends knee DOWN
        self.standing_pose = [
            0.0, -0.8,  # Back Right: shoulder 0°, knee -45° (down)
            0.0, -0.8,  # Front Right: shoulder 0°, knee -45° (down)
            0.0, +0.8,  # Back Left: shoulder 0°, knee +45° (down, axis flipped!)
            0.0, +0.8,  # Front Left: shoulder 0°, knee +45° (down, axis flipped!)
            # Temporarily disable head/tail
            # 0.0,        # Tail: neutral straight
            # 0.0, 0.0, 0.0,  # Head/neck: neutral straight forward
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

        # Startup delay to allow physics settling before control
        # Shorter delay now that gains are properly tuned
        self.startup_delay = 3.0  # seconds - enough for settling
        self.start_time = self.get_clock().now()
        self.controller_active = False

        # Publish standing pose at 50Hz
        self.timer = self.create_timer(0.02, self.publish_position)

        self.get_logger().info('PiDog Gazebo Controller started')
        self.get_logger().info(f'Waiting {self.startup_delay}s for physics to settle...')
        self.get_logger().info('Robot will remain passive until fully stable on ground')
        self.get_logger().info(f'Target standing pose: {self.standing_pose}')

    def motor_callback(self, msg):
        """Receive joint commands from gait generator."""
        if len(msg.position) == len(self.joint_names):
            self.current_position = list(msg.position)
            self.get_logger().debug('Received new joint positions from gait controller')

    def publish_position(self):
        """Publish current position to ros2_control."""
        # Check if startup delay has passed
        if not self.controller_active:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed < self.startup_delay:
                return  # Don't publish yet
            else:
                self.controller_active = True
                self.get_logger().info('Controller now active - publishing commands')

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
