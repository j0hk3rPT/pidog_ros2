#!/usr/bin/env python3
"""
Helper script to send gait commands to PiDog.

Usage:
    python3 send_gait_command.py walk
    python3 send_gait_command.py stand
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def send_command(command):
    """Send a single command to the gait controller."""
    rclpy.init()
    node = Node('gait_command_sender')

    publisher = node.create_publisher(String, '/gait_command', 10)

    # Wait for publisher to be ready
    import time
    time.sleep(0.5)

    msg = String()
    msg.data = command

    publisher.publish(msg)
    node.get_logger().info(f'Sent command: {command}')

    # Give time for message to be sent
    time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 send_gait_command.py <command>")
        print("Available commands: walk, stand")
        sys.exit(1)

    command = sys.argv[1]
    send_command(command)
