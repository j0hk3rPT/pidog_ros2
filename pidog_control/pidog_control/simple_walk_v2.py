#!/usr/bin/env python3
"""
Ultra-simple walking test - very basic alternating leg movements.
Designed to be obviously visible for debugging.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import math


class SimpleWalkV2(Node):
    def __init__(self):
        super().__init__('simple_walk_v2')

        self.position_pub = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/gait_command',
            self.gait_command_callback,
            10
        )

        self.gait_active = False
        self.step = 0

        # Standing pose - all legs at 0 shoulder, -0.8 knee
        self.stand_pose = [0.0, -0.8] * 4

        # Timer at 2Hz (very slow for visibility)
        self.timer = self.create_timer(0.5, self.update)

        self.get_logger().info('Simple Walk V2 started')
        self.get_logger().info('Send "walk" to /gait_command')

    def gait_command_callback(self, msg):
        if msg.data.lower() in ['walk', 'walk_forward']:
            self.gait_active = True
            self.step = 0
            self.get_logger().info('🐕 WALK STARTED')
        else:
            self.gait_active = False
            self.get_logger().info('⏹️ WALK STOPPED')

    def update(self):
        msg = Float64MultiArray()

        if self.gait_active:
            # Very simple: alternate lifting left and right legs
            # Step 0: Lift left legs
            # Step 1: Lower left, lift right
            # Repeat

            if self.step % 2 == 0:
                # LEFT legs up (indices 4,5,6,7 are BL and FL)
                msg.data = [
                    0.0, -0.8,  # BR - on ground
                    0.0, -0.8,  # FR - on ground
                    0.0, -0.3,  # BL - LIFTED
                    0.0, -0.3,  # FL - LIFTED
                ]
                self.get_logger().info(f'Step {self.step}: LEFT LEGS UP')
            else:
                # RIGHT legs up (indices 0,1,2,3 are BR and FR)
                msg.data = [
                    0.0, -0.3,  # BR - LIFTED
                    0.0, -0.3,  # FR - LIFTED
                    0.0, -0.8,  # BL - on ground
                    0.0, -0.8,  # FL - on ground
                ]
                self.get_logger().info(f'Step {self.step}: RIGHT LEGS UP')

            self.step += 1
        else:
            # Standing
            msg.data = self.stand_pose

        self.position_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleWalkV2()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
