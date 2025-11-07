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
from .inverse_kinematics import LegIK, HeadTailController


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
        self.gait_coords = None  # Leg coordinates (will be set by gaits)
        self.gait_head_tail = None  # Head/tail angles (will be set by gaits)
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

        # Store current angles (initialized to stand pose) - all 12 joints
        stand_angles = self.poses['stand']
        self.current_angles = stand_angles.copy()  # Should be 12 angles now

        self.get_logger().info(f"Gait Generator started. Default gait: {self.default_gait}")
        self.get_logger().info(f"Available gaits: {list(self.gaits.keys())}")
        self.get_logger().info(f"Available poses: {list(self.poses.keys())}")
        self.get_logger().info(f"Smooth transitions enabled ({self.transition_frames} frames)")

        # Initialize with default gait
        self._switch_gait(self.default_gait)

    def _generate_all_gaits(self):
        """Pre-generate coordinate sequences for all gaits with head/tail control."""
        gaits = {}

        # Walking gaits - now includes head/tail angles for balancing
        gaits['walk_forward'] = Walk(fb=Walk.FORWARD, lr=Walk.STRAIGHT).get_coords(include_head_tail=True)
        gaits['walk_backward'] = Walk(fb=Walk.BACKWARD, lr=Walk.STRAIGHT).get_coords(include_head_tail=True)
        gaits['walk_left'] = Walk(fb=Walk.FORWARD, lr=Walk.LEFT).get_coords(include_head_tail=True)
        gaits['walk_right'] = Walk(fb=Walk.FORWARD, lr=Walk.RIGHT).get_coords(include_head_tail=True)

        # Trotting gaits - now includes head/tail angles for balancing
        gaits['trot_forward'] = Trot(fb=Trot.FORWARD, lr=Trot.STRAIGHT).get_coords(include_head_tail=True)
        gaits['trot_backward'] = Trot(fb=Trot.BACKWARD, lr=Trot.STRAIGHT).get_coords(include_head_tail=True)
        gaits['trot_left'] = Trot(fb=Trot.FORWARD, lr=Trot.LEFT).get_coords(include_head_tail=True)
        gaits['trot_right'] = Trot(fb=Trot.FORWARD, lr=Trot.RIGHT).get_coords(include_head_tail=True)

        return gaits

    def _create_poses(self):
        """Create static poses with head/tail angles - use IK for consistency."""
        poses = {}

        # Generate stand pose from IK to match walking gait calculations
        # Standing height: y=0 (neutral), z=80mm (from walk_gait.py Z_ORIGIN)
        # All static poses have neutral head/tail
        neutral_head_tail = HeadTailController.neutral_pose()

        stand_coords = [
            [0, 80],  # FL: neutral position
            [0, 80],  # FR: neutral position
            [0, 80],  # BL: neutral position
            [0, 80],  # BR: neutral position
        ]
        leg_angles = LegIK.legs_coords_to_angles(stand_coords)
        poses['stand'] = leg_angles + neutral_head_tail  # 8 leg + 4 head/tail = 12 total

        # Sit pose - sitting with rear up, front down (using IK)
        # Head compensates with pitch down as in SunFounder (-30°)
        sit_coords = [
            [30, 65],   # FL: forward 30mm, lower 65mm
            [30, 65],   # FR: forward 30mm, lower 65mm
            [-20, 90],  # BL: back 20mm, higher 90mm
            [-20, 90],  # BR: back 20mm, higher 90mm
        ]
        leg_angles = LegIK.legs_coords_to_angles(sit_coords)
        # Head pitch compensation for sitting (SunFounder uses -30° = -0.524 rad)
        sit_head_tail = [0.0] + HeadTailController.head_rpy_to_angles(0.0, 0.0, 0.0, 0.0, -0.524)
        poses['sit'] = leg_angles + sit_head_tail

        # Lie pose - legs splayed out, body low (using IK)
        lie_coords = [
            [40, 60],   # FL: splayed out, low
            [40, 60],   # FR: splayed out, low
            [40, 60],   # BL: splayed out, low
            [40, 60],   # BR: splayed out, low
        ]
        leg_angles = LegIK.legs_coords_to_angles(lie_coords)
        poses['lie'] = leg_angles + neutral_head_tail

        # Stretch pose - front legs forward, back legs back (using IK)
        stretch_coords = [
            [40, 75],   # FL: forward, medium height
            [40, 75],   # FR: forward, medium height
            [-40, 85],  # BL: back, higher
            [-40, 85],  # BR: back, higher
        ]
        leg_angles = LegIK.legs_coords_to_angles(stretch_coords)
        poses['stretch'] = leg_angles + neutral_head_tail

        return poses

    def gait_command_callback(self, msg):
        """Handle gait switching commands."""
        requested_gait = msg.data
        self._switch_gait(requested_gait)

    def _switch_gait(self, gait_name):
        """Switch to a new gait or pose with smooth transition."""
        if gait_name in self.gaits:
            # Gaits now return (leg_coords, head_tail_angles) tuples
            leg_coords, head_tail_angles = self.gaits[gait_name]

            # Get target angles for first frame of new gait (8 leg + 4 head/tail = 12 total)
            leg_angles = LegIK.legs_coords_to_angles(leg_coords[0])
            target_angles = leg_angles + head_tail_angles[0]

            # Start smooth transition
            self.transition_start_angles = self.current_angles.copy()
            self.transition_target_angles = target_angles
            self.transitioning = True
            self.transition_progress = 0

            # Set new gait (will be used after transition completes)
            self.current_gait = gait_name
            self.gait_coords = leg_coords
            self.gait_head_tail = head_tail_angles
            self.frame_index = 0
            self.get_logger().info(f"Transitioning to gait: {gait_name} ({len(leg_coords)} frames with head/tail)")

        elif gait_name in self.poses:
            # Get target angles for pose (already includes all 12 joints)
            target_angles = self.poses[gait_name]

            # Start smooth transition
            self.transition_start_angles = self.current_angles.copy()
            self.transition_target_angles = target_angles
            self.transitioning = True
            self.transition_progress = 0

            # Set new pose (will be used after transition completes)
            self.current_gait = gait_name
            self.gait_coords = None  # Static pose
            self.gait_head_tail = None
            self.get_logger().info(f"Transitioning to pose: {gait_name} (12 joints)")
        else:
            self.get_logger().warn(f"Unknown gait/pose: {gait_name}")

    def update(self):
        """Publish current motor positions for all 12 joints."""
        now = self.get_clock().now()
        self.joint_state.header.stamp = now.to_msg()

        # Handle smooth transitions between gaits
        if self.transitioning:
            # Linear interpolation between start and target angles (all 12 joints)
            alpha = min(1.0, self.transition_progress / self.transition_frames)
            all_angles = []
            for i in range(12):  # All 12 joints now
                start = self.transition_start_angles[i]
                target = self.transition_target_angles[i]
                interpolated = start + (target - start) * alpha
                all_angles.append(interpolated)

            self.transition_progress += 1

            # Transition complete
            if self.transition_progress >= self.transition_frames:
                self.transitioning = False
                self.get_logger().info(f"Transition complete, now executing {self.current_gait}")
        else:
            # Normal gait execution (no transition)
            if self.gait_coords is None:
                # Static pose (already has all 12 joints)
                all_angles = self.poses[self.current_gait]
            else:
                # Dynamic gait - combine leg angles + head/tail angles
                coords = self.gait_coords[self.frame_index]
                head_tail = self.gait_head_tail[self.frame_index]

                leg_angles = LegIK.legs_coords_to_angles(coords)
                all_angles = leg_angles + head_tail  # 8 leg + 4 head/tail = 12 total

                # Advance frame (loop back to start)
                self.frame_index = (self.frame_index + 1) % len(self.gait_coords)

        # Store current angles for next transition (all 12 joints)
        self.current_angles = all_angles.copy()

        # DIAGNOSTIC: Log published array length and sample values
        if not hasattr(self, '_pub_log_count'):
            self._pub_log_count = 0
        self._pub_log_count += 1
        if self._pub_log_count <= 5 or self._pub_log_count % 100 == 0:
            self.get_logger().info(
                f'Publishing {len(all_angles)} angles: '
                f'legs[0-3]={all_angles[:4]} ... head/tail[8-11]={all_angles[8:]}'
            )

        self.joint_state.position = all_angles
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
