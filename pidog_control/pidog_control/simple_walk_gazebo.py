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

        # Gait parameters - conservative values for stability
        self.gait_active = False
        self.phase = 0.0
        self.frequency = 0.3  # Hz - slow walk for stability
        self.step_height = 0.3  # radians - moderate knee lift
        self.shoulder_swing = 0.3  # radians - moderate shoulder swing

        # Standing pose - left legs have flipped axes (positive = down)
        self.stand_pose = [0.0, -0.8, 0.0, -0.8, 0.0, +0.8, 0.0, +0.8]

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
        - Diagonal legs move together (BR+FL, BL+FR)
        - Phase 0-π: BR+FL lift and swing forward, BL+FR on ground and push backward
        - Phase π-2π: BL+FR lift and swing forward, BR+FL on ground and push backward

        Shoulder motion creates forward propulsion:
        - When lifting: shoulder swings forward (recovery stroke)
        - When on ground: shoulder swings backward (power stroke)

        Joint order: [BR_shoulder, BR_knee, FR_shoulder, FR_knee,
                      BL_shoulder, BL_knee, FL_shoulder, FL_knee]
        """
        # For diagonal pair BR+FL (phase 0-π lift, π-2π ground)
        # Shoulder: swings backward→forward during lift, forward→backward during ground
        # Using -cos(phase) gives: phase=0→forward, phase=π→backward, phase=2π→forward
        br_shoulder = -self.shoulder_swing * math.cos(phase)
        fl_shoulder = -self.shoulder_swing * math.cos(phase)

        # For diagonal pair BL+FR (phase 0-π ground, π-2π lift)
        # Shoulder: phase shifted by π from BR+FL
        bl_shoulder = -self.shoulder_swing * math.cos(phase + math.pi)
        fr_shoulder = -self.shoulder_swing * math.cos(phase + math.pi)

        # Knee movements: lift during first half of leg's cycle
        # BR and FL lift during phase 0-π
        if 0 <= phase < math.pi:
            br_knee = -0.8 + self.step_height * math.sin(phase)
            fl_knee = -0.8 + self.step_height * math.sin(phase)
            bl_knee = -0.8  # On ground
            fr_knee = -0.8  # On ground
        else:
            # BL and FR lift during phase π-2π
            br_knee = -0.8  # On ground
            fl_knee = -0.8  # On ground
            bl_knee = -0.8 + self.step_height * math.sin(phase - math.pi)
            fr_knee = -0.8 + self.step_height * math.sin(phase - math.pi)

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
