#!/usr/bin/env python3
"""
Simple test to move a single joint and verify the pipeline works.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time


class SingleJointTest(Node):
    def __init__(self):
        super().__init__('single_joint_test')

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        # Wait for publisher to be ready
        time.sleep(1.0)

        self.get_logger().info('Testing single joint movement...')
        self.get_logger().info('Will alternate between two positions every 2 seconds')

        self.timer = self.create_timer(2.0, self.publish_position)
        self.toggle = False

    def publish_position(self):
        msg = Float64MultiArray()

        if self.toggle:
            # Position 1: All shoulders forward, knees bent
            msg.data = [
                0.5, -0.8,  # BR
                0.5, -0.8,  # FR
                0.5, -0.8,  # BL
                0.5, -0.8,  # FL
            ]
            self.get_logger().info('Position 1: Shoulders forward')
        else:
            # Position 2: All shoulders backward, knees bent
            msg.data = [
                -0.5, -0.8,  # BR
                -0.5, -0.8,  # FR
                -0.5, -0.8,  # BL
                -0.5, -0.8,  # FL
            ]
            self.get_logger().info('Position 2: Shoulders backward')

        self.pub.publish(msg)
        self.toggle = not self.toggle


def main():
    rclpy.init()
    node = SingleJointTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
