#!/usr/bin/env python3
"""
Publish standing pose to keep PiDog upright.
Commands all 8 leg joints to standing position.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class StandingPosePublisher(Node):
    def __init__(self):
        super().__init__('standing_pose_publisher')

        # Publisher to position controller
        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        # Standing pose: legs spread out and bent to lift body
        # Joint order: BR_shoulder, BR_knee, FR_shoulder, FR_knee,
        #              BL_shoulder, BL_knee, FL_shoulder, FL_knee
        # Shoulders at 0 (straight out), knees at -0.8 rad (~45 deg down)
        self.standing_pose = [
            0.0, -0.8,  # Back Right leg
            0.0, -0.8,  # Front Right leg
            0.0, -0.8,  # Back Left leg
            0.0, -0.8,  # Front Left leg
        ]

        # Publish at 50Hz to maintain position
        self.timer = self.create_timer(0.02, self.publish_pose)

        self.get_logger().info('Standing pose publisher started - commanding position')

    def publish_pose(self):
        """Continuously publish standing pose to hold position."""
        msg = Float64MultiArray()
        msg.data = self.standing_pose
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StandingPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
