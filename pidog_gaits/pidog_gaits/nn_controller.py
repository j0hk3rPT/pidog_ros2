#!/usr/bin/env python3
"""
Neural Network Controller Node

Uses trained neural network to generate joint angles in real-time.
Replaces the traditional gait generator with learned behavior.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import torch
import numpy as np
from math import pi
import os

from .neural_network import GaitNet, GaitNetLarge, GaitNetSimpleLSTM, GaitNetLSTM


class NNControllerNode(Node):
    """
    ROS2 node that uses trained neural network for gait generation.

    Subscribes to /gait_command to switch gaits.
    Publishes to /motor_pos with predicted joint angles.
    """

    def __init__(self):
        super().__init__("nn_controller")

        # Parameters
        self.declare_parameter('model_path', './models/best_model.pth')
        self.declare_parameter('model_type', 'simple')  # 'simple', 'large', 'simple_lstm', or 'lstm'
        self.declare_parameter('frequency', 30)  # Hz
        self.declare_parameter('device', 'cpu')

        model_path = self.get_parameter('model_path').value
        model_type = self.get_parameter('model_type').value
        freq = self.get_parameter('frequency').value
        device = self.get_parameter('device').value

        # Load model
        self.device = torch.device(device)
        self.model = self._load_model(model_path, model_type)
        self.model.eval()  # Set to evaluation mode

        # Gait state
        self.current_gait = 'stand'
        self.phase = 0.0
        self.phase_increment = 1.0 / 100.0  # Complete cycle in ~100 frames

        # Gait encoding (same as data collector)
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

        # ROS2 setup
        qos = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, "motor_pos", qos)
        self.gait_sub = self.create_subscription(
            String, "gait_command", self.gait_callback, qos
        )

        # Timer for publishing
        self.timer = self.create_timer(1.0 / freq, self.update)

        # Joint message template
        self.joint_state = JointState()
        self.joint_state.name = [f"motor_{i}" for i in range(12)]

        self.get_logger().info("Neural Network Controller started")
        self.get_logger().info(f"Model: {model_type} from {model_path}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Available gaits: {list(self.gait_info.keys())}")

    def _load_model(self, model_path, model_type):
        """Load trained model from checkpoint."""
        # Create model
        if model_type == 'simple':
            model = GaitNet()
        elif model_type == 'large':
            model = GaitNetLarge()
        elif model_type == 'simple_lstm':
            model = GaitNetSimpleLSTM()
        elif model_type == 'lstm':
            model = GaitNetLSTM()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        self.get_logger().info(f"Loaded model from {model_path}")
        self.get_logger().info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        self.get_logger().info(f"  Val Loss: {checkpoint.get('val_loss', 'unknown'):.6f}")

        return model

    def gait_callback(self, msg):
        """Handle gait switching commands."""
        requested_gait = msg.data

        if requested_gait in self.gait_info:
            self.current_gait = requested_gait
            self.phase = 0.0  # Reset phase when switching gaits
            self.get_logger().info(f"Switched to: {requested_gait}")
        else:
            self.get_logger().warn(f"Unknown gait: {requested_gait}")

    def predict_angles(self):
        """
        Use neural network to predict joint angles.

        Returns:
            list: 12 joint angles in radians (8 legs + 4 head/tail for balance)
        """
        # Get current gait features
        gait_features = self.gait_info[self.current_gait]

        # Create input vector [gait_type, direction, turn, phase]
        input_vec = np.array([
            gait_features['gait_type'],
            gait_features['direction'],
            gait_features['turn'],
            self.phase
        ], dtype=np.float32)

        # Convert to tensor
        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Convert back to numpy
        angles = output_tensor.cpu().numpy()[0]

        return angles.tolist()

    def update(self):
        """Publish predicted motor positions."""
        now = self.get_clock().now()
        self.joint_state.header.stamp = now.to_msg()

        # Get predicted angles from neural network (all 12 motors)
        all_angles = self.predict_angles()

        self.joint_state.position = all_angles
        self.joint_pub.publish(self.joint_state)

        # Update phase (for cyclic gaits)
        gait_type = self.gait_info[self.current_gait]['gait_type']
        if gait_type != 2:  # Not a static pose
            self.phase += self.phase_increment
            if self.phase >= 1.0:
                self.phase = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = NNControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
