#!/usr/bin/env python3
"""
Virtual IMU Node - Synthesizes IMU data from Gazebo model state

This node creates realistic IMU sensor data from Gazebo's physics simulation,
allowing RL training to use the same sensor inputs as the real robot.

Subscribes to: /gazebo/model_states (from Gazebo via ros_gz_bridge)
Publishes to:  /imu (sensor_msgs/Imu)

For sim-to-real transfer:
- Training: Uses this virtual IMU computed from Gazebo physics
- Real robot: Disable this node, use real IMU hardware instead
- Network sees identical sensor_msgs/Imu format in both cases
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelStates
import numpy as np
from scipy.spatial.transform import Rotation


class VirtualIMUNode(Node):
    def __init__(self):
        super().__init__('virtual_imu_node')

        # Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)

        # Subscribe to Gazebo model states
        self.model_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_callback,
            10
        )

        # State tracking for computing derivatives
        self.last_velocity = None
        self.last_time = None

        # Noise parameters matching URDF configuration
        self.angular_vel_noise_std = 0.01  # rad/s
        self.linear_acc_noise_std = 0.1    # m/s^2

        # Gravity vector
        self.gravity = 9.81

        # Model name to track (Gazebo spawns as this name)
        self.model_name = 'PiDog'

        self.get_logger().info('Virtual IMU node started - synthesizing IMU from Gazebo model states')
        self.get_logger().info('Publishing to /imu topic')

    def model_callback(self, msg):
        """
        Compute IMU data from Gazebo model states.

        IMU outputs:
        - orientation: Direct from model pose (quaternion)
        - angular_velocity: From model twist, transformed to body frame
        - linear_acceleration: Computed from velocity derivative + gravity compensation
        """
        # Find our robot in the model states
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            # Model not found, skip this message
            return

        current_time = self.get_clock().now()

        # Extract pose and twist
        pose = msg.pose[idx]
        twist = msg.twist[idx]

        # Extract orientation quaternion
        q = pose.orientation
        orientation_quat = np.array([q.x, q.y, q.z, q.w])

        # Compute angular velocity in body frame
        # Gazebo twist angular is in world frame, transform to body frame
        angular_velocity_world = np.array([
            twist.angular.x,
            twist.angular.y,
            twist.angular.z
        ])

        # Transform to body frame
        r = Rotation.from_quat(orientation_quat)
        angular_velocity = r.inv().apply(angular_velocity_world)

        # Compute linear acceleration
        # Get current linear velocity in world frame
        linear_velocity_world = np.array([
            twist.linear.x,
            twist.linear.y,
            twist.linear.z
        ])

        # Compute acceleration from velocity derivative
        linear_acc_world = np.array([0.0, 0.0, 0.0])
        if self.last_velocity is not None and self.last_time is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                linear_acc_world = (linear_velocity_world - self.last_velocity) / dt

        # Transform to body frame and add gravity compensation
        # IMU measures specific force = acceleration - gravity
        gravity_world = np.array([0.0, 0.0, -self.gravity])
        specific_force_world = linear_acc_world - gravity_world
        linear_acceleration = r.inv().apply(specific_force_world)

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
        self.last_velocity = linear_velocity_world
        self.last_time = current_time


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
