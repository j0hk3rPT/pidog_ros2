#!/usr/bin/env python3
"""
Bridge between gait generator and Webots motors.
Subscribes to /motor_pos from gait_generator and publishes to individual motor topics for Webots.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class PiDogWebotsBridge(Node):
    """Bridge node that converts JointState commands to individual motor commands."""

    def __init__(self):
        super().__init__('pidog_webots_bridge')

        # Motor names matching Webots device names
        self.motor_names = [
            'body_to_front_left_leg_b', 'front_left_leg_b_to_a',
            'body_to_front_right_leg_b', 'front_right_leg_b_to_a',
            'body_to_back_left_leg_b', 'back_left_leg_b_to_a',
            'body_to_back_right_leg_b', 'back_right_leg_b_to_a',
            'motor_8_to_tail',
            'neck1_to_motor_9', 'neck2_to_motor_10', 'neck3_to_motor_11'
        ]

        # Create publishers for each motor
        self.motor_pubs = {}
        for name in self.motor_names:
            topic = f'/PiDog/{name}/set_position'
            self.motor_pubs[name] = self.create_publisher(Float64, topic, 10)

        # Subscribe to joint commands from gait generator
        self.create_subscription(JointState, 'motor_pos', self.motor_callback, 10)

        self.get_logger().info(f"PiDog Webots Bridge started - controlling {len(self.motor_names)} motors")

    def motor_callback(self, msg):
        """Receive joint state and publish to individual motor topics."""
        if len(msg.position) != len(self.motor_names):
            self.get_logger().warn(
                f"Expected {len(self.motor_names)} positions, got {len(msg.position)}"
            )
            return

        # Publish to each motor
        for i, name in enumerate(self.motor_names):
            cmd = Float64()
            cmd.data = float(msg.position[i])
            self.motor_pubs[name].publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PiDogWebotsBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
