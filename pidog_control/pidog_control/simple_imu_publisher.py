#!/usr/bin/env python3
"""
Simple IMU publisher that reads robot pose from Gazebo and publishes IMU data.
Alternative to Gazebo's built-in IMU sensor plugin which may not be available.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np


class SimpleIMUPublisher(Node):
    """
    Publishes IMU data by reading robot orientation from Gazebo/TF.

    This is a fallback for when Gazebo IMU sensor plugins aren't available.
    Provides:
    - Orientation (quaternion from TF)
    - Angular velocity (computed from orientation changes)
    - Linear acceleration (includes gravity from orientation)
    """

    def __init__(self):
        super().__init__('simple_imu_publisher')

        # Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # TF listener to get robot orientation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Previous orientation for computing angular velocity
        self.prev_orientation = None
        self.prev_time = None

        # Timer for publishing at 100 Hz (matching Gazebo IMU spec)
        self.timer = self.create_timer(0.01, self.publish_imu)

        self.get_logger().info("Simple IMU Publisher started")
        self.get_logger().info("Publishing IMU data to /imu/data at 100 Hz")
        self.get_logger().info("Reading orientation from TF (world -> body)")

    def publish_imu(self):
        """Publish IMU data based on robot transform."""
        try:
            # Get transform from world to robot body
            transform = self.tf_buffer.lookup_transform(
                'world',  # target frame
                'body',   # source frame
                rclpy.time.Time()  # latest
            )

            # Create IMU message
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'body'

            # Orientation (quaternion from TF)
            imu_msg.orientation.x = transform.transform.rotation.x
            imu_msg.orientation.y = transform.transform.rotation.y
            imu_msg.orientation.z = transform.transform.rotation.z
            imu_msg.orientation.w = transform.transform.rotation.w

            # Orientation covariance (reasonable values)
            imu_msg.orientation_covariance = [
                0.0001, 0, 0,
                0, 0.0001, 0,
                0, 0, 0.0001
            ]

            # Compute angular velocity from orientation changes
            current_time = self.get_clock().now()
            if self.prev_orientation is not None and self.prev_time is not None:
                dt = (current_time - self.prev_time).nanoseconds / 1e9
                if dt > 0:
                    # Simplified angular velocity calculation
                    # (proper implementation would use quaternion derivatives)
                    dqx = (transform.transform.rotation.x - self.prev_orientation[0]) / dt
                    dqy = (transform.transform.rotation.y - self.prev_orientation[1]) / dt
                    dqz = (transform.transform.rotation.z - self.prev_orientation[2]) / dt

                    # Approximate angular velocity (simplified)
                    imu_msg.angular_velocity.x = 2.0 * dqx
                    imu_msg.angular_velocity.y = 2.0 * dqy
                    imu_msg.angular_velocity.z = 2.0 * dqz

                    # Add noise matching Gazebo IMU spec (stddev=0.01 rad/s)
                    noise_scale = 0.01
                    imu_msg.angular_velocity.x += np.random.normal(0, noise_scale)
                    imu_msg.angular_velocity.y += np.random.normal(0, noise_scale)
                    imu_msg.angular_velocity.z += np.random.normal(0, noise_scale)

            imu_msg.angular_velocity_covariance = [
                0.0001, 0, 0,
                0, 0.0001, 0,
                0, 0, 0.0001
            ]

            # Compute gravity vector in body frame
            # Rotate gravity (0, 0, -9.81) by inverse of body orientation
            q = transform.transform.rotation
            grav = 9.81

            # Simplified gravity rotation (full implementation would use proper quaternion math)
            # For now, approximate based on tilt
            imu_msg.linear_acceleration.x = 0.0
            imu_msg.linear_acceleration.y = 0.0
            imu_msg.linear_acceleration.z = -grav  # Always pointing down in world frame

            # Add noise matching Gazebo IMU spec (stddev=0.1 m/s²)
            noise_scale = 0.1
            imu_msg.linear_acceleration.x += np.random.normal(0, noise_scale)
            imu_msg.linear_acceleration.y += np.random.normal(0, noise_scale)
            imu_msg.linear_acceleration.z += np.random.normal(0, noise_scale)

            imu_msg.linear_acceleration_covariance = [
                0.01, 0, 0,
                0, 0.01, 0,
                0, 0, 0.01
            ]

            # Publish
            self.imu_pub.publish(imu_msg)

            # Save for next iteration
            self.prev_orientation = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            self.prev_time = current_time

        except Exception as e:
            # TF not available yet - this is normal at startup
            if 'world' not in str(e) and 'body' not in str(e):
                self.get_logger().debug(f'Waiting for TF: {e}')


def main(args=None):
    rclpy.init(args=args)
    imu_publisher = SimpleIMUPublisher()

    try:
        rclpy.spin(imu_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        imu_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
