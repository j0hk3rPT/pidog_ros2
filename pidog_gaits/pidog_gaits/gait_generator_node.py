#!/usr/bin/env python3
"""
Gait Generator ROS2 Node

Generates various gaits and publishes motor positions.
Supports: walk, trot, sit, stand, and more.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from math import pi

from .walk_gait import Walk
from .trot_gait import Trot
from .inverse_kinematics import LegIK


class GaitGeneratorNode(Node):
    """
    ROS2 node for generating robot gaits.

    Publishes to /motor_pos topic with JointState messages.
    Subscribes to /gait_command to switch between gaits.
    """

    def __init__(self):
        super().__init__("gait_generator")

        # Parameters
        self.declare_parameter('frequency', 30)  # Hz
        self.declare_parameter('default_gait', 'stand')
        self.declare_parameter('transition_frames', 30)  # Frames for smooth transition (1 second at 30Hz)

        freq = self.get_parameter('frequency').value
        self.default_gait = self.get_parameter('default_gait').value
        self.transition_frames = self.get_parameter('transition_frames').value

        # Publishers and subscribers
        qos = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, "motor_pos", qos)
        self.gait_sub = self.create_subscription(
            String, "gait_command", self.gait_command_callback, qos
        )

        # Timer for publishing at fixed rate
        self.timer = self.create_timer(1.0 / freq, self.update)

        # Gait state
        self.current_gait = self.default_gait
        self.gait_coords = []
        self.frame_index = 0

        # Transition state for smooth gait switching
        self.transitioning = False
        self.transition_progress = 0
        self.transition_start_angles = None
        self.transition_target_angles = None

        # Pre-generate all gaits
        self.gaits = self._generate_all_gaits()

        # Static poses
        self.poses = self._create_poses()

        # Joint message template
        self.joint_state = JointState()
        self.joint_state.name = [f"motor_{i}" for i in range(12)]

        # Store current angles (initialized to stand pose)
        stand_angles = self.poses['stand']
        self.current_angles = stand_angles.copy()

        self.get_logger().info(f"Gait Generator started. Default gait: {self.default_gait}")
        self.get_logger().info(f"Available gaits: {list(self.gaits.keys())}")
        self.get_logger().info(f"Available poses: {list(self.poses.keys())}")
        self.get_logger().info(f"Smooth transitions enabled ({self.transition_frames} frames)")

        # Initialize with default gait
        self._switch_gait(self.default_gait)

    def _generate_all_gaits(self):
        """Pre-generate coordinate sequences for all gaits."""
        gaits = {}

        # Walking gaits
        gaits['walk_forward'] = Walk(fb=Walk.FORWARD, lr=Walk.STRAIGHT).get_coords()
        gaits['walk_backward'] = Walk(fb=Walk.BACKWARD, lr=Walk.STRAIGHT).get_coords()
        gaits['walk_left'] = Walk(fb=Walk.FORWARD, lr=Walk.LEFT).get_coords()
        gaits['walk_right'] = Walk(fb=Walk.FORWARD, lr=Walk.RIGHT).get_coords()

        # Trotting gaits
        gaits['trot_forward'] = Trot(fb=Trot.FORWARD, lr=Trot.STRAIGHT).get_coords()
        gaits['trot_backward'] = Trot(fb=Trot.BACKWARD, lr=Trot.STRAIGHT).get_coords()
        gaits['trot_left'] = Trot(fb=Trot.FORWARD, lr=Trot.LEFT).get_coords()
        gaits['trot_right'] = Trot(fb=Trot.FORWARD, lr=Trot.RIGHT).get_coords()

        return gaits

    def _create_poses(self):
        """Create static poses."""
        poses = {}

        # Stand pose - neutral standing position
        stand_coords = [[0, 80], [0, 80], [0, 80], [0, 80]]
        poses['stand'] = LegIK.legs_coords_to_angles(stand_coords)

        # Sit pose - back legs bent, front legs straight
        sit_coords = [[0, 80], [0, 80], [30, 60], [30, 60]]
        poses['sit'] = LegIK.legs_coords_to_angles(sit_coords)

        # Lie pose - all legs bent
        lie_coords = [[20, 40], [20, 40], [20, 40], [20, 40]]
        poses['lie'] = LegIK.legs_coords_to_angles(lie_coords)

        # Stretch pose - legs extended forward
        stretch_coords = [[40, 70], [40, 70], [-20, 90], [-20, 90]]
        poses['stretch'] = LegIK.legs_coords_to_angles(stretch_coords)

        return poses

    def gait_command_callback(self, msg):
        """Handle gait switching commands."""
        requested_gait = msg.data
        self._switch_gait(requested_gait)

    def _switch_gait(self, gait_name):
        """Switch to a new gait or pose with smooth transition."""
        if gait_name in self.gaits:
            # Get target angles for first frame of new gait
            coords = self.gaits[gait_name][0]
            target_angles = LegIK.legs_coords_to_angles(coords)

            # Start smooth transition
            self.transition_start_angles = self.current_angles.copy()
            self.transition_target_angles = target_angles
            self.transitioning = True
            self.transition_progress = 0

            # Set new gait (will be used after transition completes)
            self.current_gait = gait_name
            self.gait_coords = self.gaits[gait_name]
            self.frame_index = 0
            self.get_logger().info(f"Transitioning to gait: {gait_name} ({len(self.gait_coords)} frames)")

        elif gait_name in self.poses:
            # Get target angles for pose
            target_angles = self.poses[gait_name]

            # Start smooth transition
            self.transition_start_angles = self.current_angles.copy()
            self.transition_target_angles = target_angles
            self.transitioning = True
            self.transition_progress = 0

            # Set new pose (will be used after transition completes)
            self.current_gait = gait_name
            self.gait_coords = None  # Static pose
            self.get_logger().info(f"Transitioning to pose: {gait_name}")
        else:
            self.get_logger().warn(f"Unknown gait/pose: {gait_name}")

    def update(self):
        """Publish current motor positions."""
        now = self.get_clock().now()
        self.joint_state.header.stamp = now.to_msg()

        # Handle smooth transitions between gaits
        if self.transitioning:
            # Linear interpolation between start and target angles
            alpha = min(1.0, self.transition_progress / self.transition_frames)
            leg_angles = []
            for i in range(8):
                start = self.transition_start_angles[i]
                target = self.transition_target_angles[i]
                interpolated = start + (target - start) * alpha
                leg_angles.append(interpolated)

            self.transition_progress += 1

            # Transition complete
            if self.transition_progress >= self.transition_frames:
                self.transitioning = False
                self.get_logger().info(f"Transition complete, now executing {self.current_gait}")
        else:
            # Normal gait execution (no transition)
            if self.gait_coords is None:
                # Static pose
                leg_angles = self.poses[self.current_gait]
            else:
                # Dynamic gait
                coords = self.gait_coords[self.frame_index]
                leg_angles = LegIK.legs_coords_to_angles(coords)

                # Advance frame (loop back to start)
                self.frame_index = (self.frame_index + 1) % len(self.gait_coords)

        # Store current angles for next transition
        self.current_angles = leg_angles.copy()

        # Publish angles (already in radians from inverse_kinematics)
        # 8 leg angles + 4 zeros for tail/head
        angles_rad = leg_angles.copy()
        angles_rad.extend([0.0, 0.0, 0.0, 0.0])  # motors 8-11 (tail, head)

        self.joint_state.position = angles_rad
        self.joint_pub.publish(self.joint_state)


def main(args=None):
    rclpy.init(args=args)
    node = GaitGeneratorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
