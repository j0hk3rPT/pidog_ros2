#!/usr/bin/env python3
"""
Enhanced Data Collector for Neural Network Training with Observation Noise

Records gait trajectories with added noise for robustness.
Prepares data for better sim-to-real transfer.
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


class DataCollectorEnhancedNode(Node):
    """
    Enhanced data collector with observation noise for robust training.

    Records:
        - Time step
        - Current gait type (with optional randomization)
        - Gait phase (0.0 to 1.0)
        - Direction (forward/backward)
        - Turn direction (left/straight/right)
        - 8 joint angles (output) with added noise
        - 8 joint velocities (optional, for LSTM)
    """

    def __init__(self):
        super().__init__("data_collector_enhanced")

        # Parameters
        self.declare_parameter('output_dir', './training_data')
        self.declare_parameter('collect_duration', 60.0)  # seconds per gait
        self.declare_parameter('add_noise', True)  # Enable observation noise
        self.declare_parameter('position_noise_std', 0.01)  # rad (~0.57°)
        self.declare_parameter('velocity_noise_std', 0.1)  # rad/s
        self.declare_parameter('collect_velocities', True)  # For LSTM training

        self.output_dir = self.get_parameter('output_dir').value
        self.duration = self.get_parameter('collect_duration').value
        self.add_noise = self.get_parameter('add_noise').value
        self.pos_noise_std = self.get_parameter('position_noise_std').value
        self.vel_noise_std = self.get_parameter('velocity_noise_std').value
        self.collect_vel = self.get_parameter('collect_velocities').value

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Data storage
        self.data = []
        self.start_time = None
        self.current_gait = "unknown"
        self.previous_positions = None
        self.previous_time = None

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

        self.get_logger().info(f"Enhanced Data Collector started. Saving to: {self.output_dir}")
        self.get_logger().info(f"Will collect {self.duration}s per gait")
        self.get_logger().info(f"Observation noise: {self.add_noise} (pos: {self.pos_noise_std} rad, vel: {self.vel_noise_std} rad/s)")
        self.get_logger().info(f"Collect velocities: {self.collect_vel}")

    def gait_callback(self, msg):
        """Track current gait."""
        self.current_gait = msg.data
        self.gait_start_frame = self.frame_count
        self.get_logger().info(f"Now recording: {self.current_gait}")

    def joint_callback(self, msg):
        """Record joint state data with observation noise."""
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
        leg_angles = np.array(msg.position[:8], dtype=np.float32)

        # Add observation noise to positions (simulates sensor noise)
        if self.add_noise:
            leg_angles_noisy = leg_angles + np.random.normal(
                0, self.pos_noise_std, size=8
            ).astype(np.float32)
        else:
            leg_angles_noisy = leg_angles

        # Calculate velocities if needed
        leg_velocities = None
        if self.collect_vel:
            if len(msg.velocity) >= 8:
                # Use provided velocities
                leg_velocities = np.array(msg.velocity[:8], dtype=np.float32)
            elif self.previous_positions is not None and self.previous_time is not None:
                # Compute velocities from position difference
                dt = elapsed - self.previous_time
                if dt > 0:
                    leg_velocities = (leg_angles - self.previous_positions) / dt
                else:
                    leg_velocities = np.zeros(8, dtype=np.float32)
            else:
                leg_velocities = np.zeros(8, dtype=np.float32)

            # Add observation noise to velocities
            if self.add_noise and leg_velocities is not None:
                leg_velocities = leg_velocities + np.random.normal(
                    0, self.vel_noise_std, size=8
                ).astype(np.float32)

        # Create data point
        data_point = {
            'timestamp': elapsed,
            'frame': self.frame_count,
            'gait_name': self.current_gait,
            'gait_type': gait_features['gait_type'],     # 0=walk, 1=trot, 2=pose
            'direction': gait_features['direction'],     # -1=back, 0=none, 1=forward
            'turn': gait_features['turn'],               # -1=left, 0=straight, 1=right
            'phase': phase,                               # 0.0 to 1.0
            'joint_angles': leg_angles_noisy.tolist()    # 8 angles in radians (with noise)
        }

        if leg_velocities is not None:
            data_point['joint_velocities'] = leg_velocities.tolist()

        self.data.append(data_point)
        self.frame_count += 1

        # Store for velocity calculation
        self.previous_positions = leg_angles.copy()
        self.previous_time = elapsed

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
        filename = f"gait_data_enhanced_{timestamp}.json"
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
        velocities = [] if self.collect_vel else None

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

            # Velocities (if collected)
            if self.collect_vel and 'joint_velocities' in point:
                velocities.append(point['joint_velocities'])

        inputs = np.array(inputs, dtype=np.float32)
        outputs = np.array(outputs, dtype=np.float32)

        # Save
        save_dict = {'inputs': inputs, 'outputs': outputs}
        if velocities:
            velocities = np.array(velocities, dtype=np.float32)
            save_dict['velocities'] = velocities

        np.savez(filename, **save_dict)
        self.get_logger().info(f"Saved NumPy arrays to {filename}")
        self.get_logger().info(f"  Input shape: {inputs.shape}")
        self.get_logger().info(f"  Output shape: {outputs.shape}")
        if velocities is not None:
            self.get_logger().info(f"  Velocity shape: {velocities.shape}")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorEnhancedNode()

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
