#!/usr/bin/env python3
"""
Virtual IMU Node - Synthesizes IMU data from Gazebo model state

This node creates realistic IMU sensor data from Gazebo's physics simulation,
allowing RL training to use the same sensor inputs as the real robot.

Subscribes to: /model/pidog/pose (from Gazebo)
Publishes to:  /imu (sensor_msgs/Imu)

For sim-to-real transfer:
- Training: Uses this virtual IMU computed from Gazebo physics
- Real robot: Disable this node, use real IMU hardware instead
- Network sees identical sensor_msgs/Imu format in both cases
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
import numpy as np
from scipy.spatial.transform import Rotation
import math


class VirtualIMUNode(Node):
    def __init__(self):
        super().__init__('virtual_imu_node')

        # Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)

        # Subscribers - TF messages from Gazebo
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

        # State tracking for computing derivatives
        self.last_orientation = None
        self.last_time = None
        self.last_velocity = None

        # Noise parameters matching URDF configuration
        self.angular_vel_noise_std = 0.01  # rad/s
        self.linear_acc_noise_std = 0.1    # m/s^2

        # Gravity vector (Gazebo's linear acceleration includes gravity)
        self.gravity = 9.81

        self.get_logger().info('Virtual IMU node started - synthesizing IMU from TF')
        self.get_logger().info('Publishing to /imu topic at 100Hz')

    def tf_callback(self, msg):
        """
        Compute IMU data from TF transform (body link).

        IMU outputs:
        - orientation: Direct from TF (quaternion)
        - angular_velocity: Computed from orientation derivative
        - linear_acceleration: Gravity transformed to body frame
        """
        # Find the body link transform
        body_transform = None
        for transform in msg.transforms:
            # Look for transform from world/odom to body link
            if transform.child_frame_id == 'body' or transform.child_frame_id == 'base_link':
                body_transform = transform
                break

        if body_transform is None:
            return  # Body transform not in this message

        current_time = self.get_clock().now()

        # Extract orientation quaternion
        q = body_transform.transform.rotation
        orientation_quat = np.array([q.x, q.y, q.z, q.w])

        # Compute angular velocity from orientation change
        angular_velocity = np.array([0.0, 0.0, 0.0])
        if self.last_orientation is not None and self.last_time is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                angular_velocity = self._compute_angular_velocity(
                    self.last_orientation,
                    orientation_quat,
                    dt
                )

        # For now, we don't have velocity from pose topic, so linear acceleration
        # will just be gravity transformed to body frame
        linear_acceleration = self._compute_linear_acceleration(orientation_quat)

        # Add realistic noise
        angular_velocity += np.random.normal(0, self.angular_vel_noise_std, 3)
        linear_acceleration += np.random.normal(0, self.linear_acc_noise_std, 3)

        # Create IMU message
        imu_msg = Imu()
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Orientation (quaternion)
        imu_msg.orientation.x = q.x
        imu_msg.orientation.y = q.y
        imu_msg.orientation.z = q.z
        imu_msg.orientation.w = q.w
        imu_msg.orientation_covariance = [
            0.01, 0, 0,
            0, 0.01, 0,
            0, 0, 0.01
        ]

        # Angular velocity (rad/s)
        imu_msg.angular_velocity.x = angular_velocity[0]
        imu_msg.angular_velocity.y = angular_velocity[1]
        imu_msg.angular_velocity.z = angular_velocity[2]
        imu_msg.angular_velocity_covariance = [
            0.0001, 0, 0,
            0, 0.0001, 0,
            0, 0, 0.0001
        ]

        # Linear acceleration (m/s^2)
        imu_msg.linear_acceleration.x = linear_acceleration[0]
        imu_msg.linear_acceleration.y = linear_acceleration[1]
        imu_msg.linear_acceleration.z = linear_acceleration[2]
        imu_msg.linear_acceleration_covariance = [
            0.01, 0, 0,
            0, 0.01, 0,
            0, 0, 0.01
        ]

        # Publish
        self.imu_pub.publish(imu_msg)

        # Update state for next iteration
        self.last_orientation = orientation_quat
        self.last_time = current_time

    def _compute_angular_velocity(self, q_prev, q_curr, dt):
        """
        Compute angular velocity from quaternion change.

        Uses quaternion derivative: ω = 2 * q_dot * q_conj
        """
        # Convert quaternions to scipy Rotation objects
        r_prev = Rotation.from_quat(q_prev)
        r_curr = Rotation.from_quat(q_curr)

        # Compute relative rotation
        r_delta = r_curr * r_prev.inv()

        # Convert to axis-angle to get angular velocity
        rotvec = r_delta.as_rotvec()  # Returns rotation vector (axis * angle)
        angular_velocity = rotvec / dt  # Convert to angular velocity

        return angular_velocity

    def _compute_linear_acceleration(self, orientation_quat):
        """
        Compute linear acceleration in body frame.

        Since we don't have velocity from pose topic, we return gravity
        transformed to body frame (what a real IMU measures when stationary).

        A real IMU measures: specific_force = acceleration - gravity_world
        When stationary: specific_force = -gravity_world
        Transformed to body frame using orientation.
        """
        # Gravity in world frame (pointing down)
        gravity_world = np.array([0.0, 0.0, -self.gravity])

        # Transform to body frame
        r = Rotation.from_quat(orientation_quat)
        gravity_body = r.inv().apply(gravity_world)

        # IMU measures specific force = -gravity when stationary
        linear_acceleration = -gravity_body

        return linear_acceleration


def main(args=None):
    rclpy.init(args=args)
    node = VirtualIMUNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
