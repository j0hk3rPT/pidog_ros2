#!/usr/bin/env python3
"""
Data Collector for Neural Network Training

Records gait trajectories to create training dataset.
Saves data in format suitable for PyTorch training.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np
import json
import os
from datetime import datetime


class DataCollectorNode(Node):
    """
    Collects gait data for neural network training.

    Records:
        - Time step
        - Current gait type
        - Gait phase (0.0 to 1.0)
        - Direction (forward/backward)
        - Turn direction (left/straight/right)
        - 8 joint angles (output)
    """

    def __init__(self):
        super().__init__("data_collector")

        # Parameters
        self.declare_parameter('output_dir', './training_data')
        self.declare_parameter('collect_duration', 60.0)  # seconds per gait

        self.output_dir = self.get_parameter('output_dir').value
        self.duration = self.get_parameter('collect_duration').value

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Data storage
        self.data = []
        self.start_time = None
        self.current_gait = "unknown"

        # ROS2 setup
        qos = QoSProfile(depth=10)
        self.joint_sub = self.create_subscription(
            JointState, "motor_pos", self.joint_callback, qos
        )
        self.gait_sub = self.create_subscription(
            String, "gait_command", self.gait_callback, qos
        )

        # Gait encoding: convert gait names to numerical features
        self.gait_info = {
            # Walking gaits
            'walk_forward': {'gait_type': 0, 'direction': 1, 'turn': 0},
            'walk_backward': {'gait_type': 0, 'direction': -1, 'turn': 0},
            'walk_left': {'gait_type': 0, 'direction': 1, 'turn': -1},
            'walk_right': {'gait_type': 0, 'direction': 1, 'turn': 1},

            # Trotting gaits
            'trot_forward': {'gait_type': 1, 'direction': 1, 'turn': 0},
            'trot_backward': {'gait_type': 1, 'direction': -1, 'turn': 0},
            'trot_left': {'gait_type': 1, 'direction': 1, 'turn': -1},
            'trot_right': {'gait_type': 1, 'direction': 1, 'turn': 1},

            # Static poses
            'stand': {'gait_type': 2, 'direction': 0, 'turn': 0},
            'sit': {'gait_type': 2, 'direction': 0, 'turn': 0},
            'lie': {'gait_type': 2, 'direction': 0, 'turn': 0},
            'stretch': {'gait_type': 2, 'direction': 0, 'turn': 0},
        }

        self.frame_count = 0
        self.gait_start_frame = 0

        self.get_logger().info(f"Data Collector started. Saving to: {self.output_dir}")
        self.get_logger().info(f"Will collect {self.duration}s per gait")

    def gait_callback(self, msg):
        """Track current gait."""
        self.current_gait = msg.data
        self.gait_start_frame = self.frame_count
        self.get_logger().info(f"Now recording: {self.current_gait}")

    def joint_callback(self, msg):
        """Record joint state data."""
        if self.start_time is None:
            self.start_time = self.get_clock().now()

        # Calculate elapsed time
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        # Get gait features
        if self.current_gait not in self.gait_info:
            return  # Skip unknown gaits

        gait_features = self.gait_info[self.current_gait]

        # Calculate phase within gait cycle (approximate)
        # Phase resets when gait changes
        frames_in_gait = self.frame_count - self.gait_start_frame
        phase = (frames_in_gait % 100) / 100.0  # Assume ~100 frames per cycle

        # Extract leg angles only (first 8 motors)
        leg_angles = list(msg.position[:8])

        # Create data point
        data_point = {
            'timestamp': elapsed,
            'frame': self.frame_count,
            'gait_name': self.current_gait,
            'gait_type': gait_features['gait_type'],     # 0=walk, 1=trot, 2=pose
            'direction': gait_features['direction'],     # -1=back, 0=none, 1=forward
            'turn': gait_features['turn'],               # -1=left, 0=straight, 1=right
            'phase': phase,                               # 0.0 to 1.0
            'joint_angles': leg_angles                    # 8 angles in radians
        }

        self.data.append(data_point)
        self.frame_count += 1

        # Log progress every 100 frames
        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f"Collected {self.frame_count} frames, {len(self.data)} data points"
            )

    def save_data(self):
        """Save collected data to file."""
        if len(self.data) == 0:
            self.get_logger().warn("No data collected!")
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gait_data_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

        self.get_logger().info(f"Saved {len(self.data)} data points to {filepath}")

        # Also save as NumPy arrays for easier loading
        np_filename = filepath.replace('.json', '.npz')
        self._save_numpy(np_filename)

    def _save_numpy(self, filename):
        """Save data as NumPy arrays for faster loading."""
        # Convert to arrays
        inputs = []
        outputs = []

        for point in self.data:
            # Input features: [gait_type, direction, turn, phase]
            input_vec = [
                point['gait_type'],
                point['direction'],
                point['turn'],
                point['phase']
            ]
            inputs.append(input_vec)

            # Output: 8 joint angles
            outputs.append(point['joint_angles'])

        inputs = np.array(inputs, dtype=np.float32)
        outputs = np.array(outputs, dtype=np.float32)

        # Save
        np.savez(filename, inputs=inputs, outputs=outputs)
        self.get_logger().info(f"Saved NumPy arrays to {filename}")
        self.get_logger().info(f"  Input shape: {inputs.shape}")
        self.get_logger().info(f"  Output shape: {outputs.shape}")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save data on exit
        node.get_logger().info("Shutting down, saving data...")
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
