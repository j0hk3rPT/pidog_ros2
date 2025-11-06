#!/usr/bin/env python3
"""
Simple walking gait for Gazebo testing.
Generates a basic forward walking motion directly for position_controller.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import math


class SimpleWalkNode(Node):
    """Simple walking gait generator for testing in Gazebo."""

    def __init__(self):
        super().__init__('simple_walk')

        # Publisher to position controller
        self.position_pub = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        # Subscriber for gait commands
        self.command_sub = self.create_subscription(
            String,
            '/gait_command',
            self.gait_command_callback,
            10
        )

        # Gait parameters
        self.gait_active = False
        self.phase = 0.0
        self.frequency = 0.5  # Hz - how fast to walk (slower for stability)
        self.step_height = 0.6  # radians - how high to lift knee (increased from 0.3)
        self.shoulder_swing = 0.5  # radians - how much to swing shoulder (increased from 0.2)

        # Standing pose
        self.stand_pose = [0.0, -0.8, 0.0, -0.8, 0.0, -0.8, 0.0, -0.8]

        # Timer for publishing at 50Hz
        self.timer = self.create_timer(0.02, self.update)

        self.get_logger().info('Simple Walk Node started')
        self.get_logger().info('Send "walk" to /gait_command to start walking')
        self.get_logger().info('Send "stand" to /gait_command to stop')

    def gait_command_callback(self, msg):
        """Handle gait commands."""
        command = msg.data.lower()

        if command == 'walk' or command == 'walk_forward':
            self.gait_active = True
            self.phase = 0.0
            self.get_logger().info('🐕 Starting walk!')
        elif command == 'stand' or command == 'stop':
            self.gait_active = False
            self.get_logger().info('⏹️  Stopping walk')
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def update(self):
        """Generate and publish gait positions."""
        if self.gait_active:
            # Calculate walking gait
            positions = self.calculate_walk_gait(self.phase)
            self.phase += (2.0 * math.pi * self.frequency * 0.02)  # Increment phase

            if self.phase >= 2.0 * math.pi:
                self.phase -= 2.0 * math.pi
        else:
            # Standing pose
            positions = self.stand_pose

        # Publish
        msg = Float64MultiArray()
        msg.data = positions
        self.position_pub.publish(msg)

    def calculate_walk_gait(self, phase):
        """
        Calculate joint angles for walking gait.

        Uses a simple trotting pattern:
        - Diagonal legs move together
        - Phase 0-π: BR+FL lift, BL+FR on ground
        - Phase π-2π: BL+FR lift, BR+FL on ground

        Joint order: [BR_shoulder, BR_knee, FR_shoulder, FR_knee,
                      BL_shoulder, BL_knee, FL_shoulder, FL_knee]
        """
        # Back Right (0, 1)
        if 0 <= phase < math.pi:
            # Lifting phase
            br_shoulder = self.shoulder_swing * math.sin(phase)
            br_knee = -0.8 + self.step_height * math.sin(phase)
        else:
            # Ground phase
            br_shoulder = self.shoulder_swing * math.sin(phase)
            br_knee = -0.8

        # Front Right (2, 3) - opposite to back right
        if 0 <= phase < math.pi:
            # Ground phase
            fr_shoulder = -self.shoulder_swing * math.sin(phase + math.pi)
            fr_knee = -0.8
        else:
            # Lifting phase
            fr_shoulder = -self.shoulder_swing * math.sin(phase + math.pi)
            fr_knee = -0.8 + self.step_height * math.sin(phase - math.pi)

        # Back Left (4, 5) - opposite to back right
        if 0 <= phase < math.pi:
            # Ground phase
            bl_shoulder = -self.shoulder_swing * math.sin(phase + math.pi)
            bl_knee = -0.8
        else:
            # Lifting phase
            bl_shoulder = -self.shoulder_swing * math.sin(phase + math.pi)
            bl_knee = -0.8 + self.step_height * math.sin(phase - math.pi)

        # Front Left (6, 7) - same as back right
        if 0 <= phase < math.pi:
            # Lifting phase
            fl_shoulder = self.shoulder_swing * math.sin(phase)
            fl_knee = -0.8 + self.step_height * math.sin(phase)
        else:
            # Ground phase
            fl_shoulder = self.shoulder_swing * math.sin(phase)
            fl_knee = -0.8

        return [
            br_shoulder, br_knee,  # Back Right
            fr_shoulder, fr_knee,  # Front Right
            bl_shoulder, bl_knee,  # Back Left
            fl_shoulder, fl_knee,  # Front Left
        ]


def main(args=None):
    rclpy.init(args=args)
    node = SimpleWalkNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
