#!/usr/bin/env python3
"""
Debug script to monitor PiDog topics and joint states.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class TopicMonitor(Node):
    def __init__(self):
        super().__init__('topic_monitor')

        self.cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/position_controller/commands',
            self.cmd_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.cmd_count = 0
        self.joint_count = 0

    def cmd_callback(self, msg):
        self.cmd_count += 1
        if self.cmd_count % 50 == 0:  # Print every 50 messages (1 second at 50Hz)
            self.get_logger().info(f'Commands (#{self.cmd_count}): {[f"{x:.3f}" for x in msg.data]}')

    def joint_callback(self, msg):
        self.joint_count += 1
        if self.joint_count % 100 == 0:  # Print every 100 messages
            # Print first 8 joints (the legs)
            positions = msg.position[:8]
            self.get_logger().info(f'Joint States (#{self.joint_count}): {[f"{x:.3f}" for x in positions]}')


def main():
    rclpy.init()
    node = TopicMonitor()

    print("Monitoring topics...")
    print("  - /position_controller/commands")
    print("  - /joint_states")
    print()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
