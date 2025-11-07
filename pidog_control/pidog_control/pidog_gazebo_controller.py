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

        # All 12 joint names: 8 legs + 1 tail + 3 head/neck
        self.joint_names = [
            'body_to_back_right_leg_b',
            'back_right_leg_b_to_a',
            'body_to_front_right_leg_b',
            'front_right_leg_b_to_a',
            'body_to_back_left_leg_b',
            'back_left_leg_b_to_a',
            'body_to_front_left_leg_b',
            'front_left_leg_b_to_a',
            'motor_8_to_tail',
            'neck1_to_motor_9',
            'neck2_to_motor_10',
            'neck3_to_motor_11',
        ]

        # Standing pose: 8 leg joints + 4 head/tail joints (all 12 motors)
        # IK stand pose: shoulders=±1.208 rad, knees=±0.180 rad
        # This matches the spawn position in gazebo.launch.py
        # NOTE: Left legs have flipped joint axes (rpy="0 1.57 3.1415" in URDF)
        # Right legs: negative shoulder, positive knee
        # Left legs: positive shoulder, negative knee
        self.standing_pose = [
            # Leg joints (8)
            -1.208, +0.180,  # Back Right: shoulder -1.208, knee +0.180
            -1.208, +0.180,  # Front Right: shoulder -1.208, knee +0.180
            +1.208, -0.180,  # Back Left: shoulder +1.208, knee -0.180 (axis flipped!)
            +1.208, -0.180,  # Front Left: shoulder +1.208, knee -0.180 (axis flipped!)
            # Tail joint (1) - neutral position
            0.0,  # Tail centered
            # Head/neck joints (3) - neutral position, head forward and level
            0.0,  # Neck yaw (motor_9) - centered
            0.0,  # Neck roll (motor_10) - level
            0.0,  # Neck pitch (motor_11) - forward
        ]

        self.get_logger().info(f"Target standing pose: {self.standing_pose}")

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

    def motor_callback(self, msg):
        """Receive joint commands from gait generator."""
        # Accept all 12 joint positions: 8 legs + 1 tail + 3 head/neck
        if len(msg.position) >= 12:
            # Use all 12 joint positions
            self.current_position = list(msg.position[:12])
            # Only log first few messages to avoid spam
            if not hasattr(self, '_msg_count'):
                self._msg_count = 0
            self._msg_count += 1
            if self._msg_count <= 3:
                self.get_logger().info(f'✓ Received {len(msg.position)} joint positions from gait generator (using all 12)')
        elif len(msg.position) >= 8:
            # Backward compatibility: If only 8 positions sent, fill head/tail with neutral
            self.current_position = list(msg.position[:8]) + [0.0, 0.0, 0.0, 0.0]
            if not hasattr(self, '_msg_count'):
                self._msg_count = 0
            self._msg_count += 1
            if self._msg_count <= 3:
                self.get_logger().info(f'✓ Received {len(msg.position)} joint positions (padded to 12 with neutral head/tail)')
        else:
            self.get_logger().warn(
                f'✗ Received {len(msg.position)} positions but need at least 8 for leg joints'
            )

    def publish_position(self):
        """Publish current position to ros2_control."""
        # Check if startup delay has passed
        if not self.controller_active:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed >= self.startup_delay:
                self.controller_active = True
                self.get_logger().info('Controller now active - accepting gait commands')
            # During startup, keep publishing standing pose to prevent collapse

        # Log position array length (should be 12 joints: 8 legs + 1 tail + 3 head)
        if not hasattr(self, '_pos_log_count'):
            self._pos_log_count = 0
        self._pos_log_count += 1
        if self._pos_log_count <= 5 or self._pos_log_count % 100 == 0:
            self.get_logger().info(f'Publishing {len(self.current_position)} joint values (8 legs + 4 head/tail): legs={self.current_position[:4]}... tail/head={self.current_position[8:]}...')

        msg = Float64MultiArray()
        # Explicitly ensure we're publishing exactly 12 values as a proper list
        msg.data = list(self.current_position) if len(self.current_position) == 12 else self.standing_pose
        if len(msg.data) != 12:
            self.get_logger().error(f'ERROR: Trying to publish {len(msg.data)} values instead of 12!')
            msg.data = self.standing_pose  # Fallback to known good 12-value pose
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
