#!/usr/bin/env python3
"""
Direct Webots external controller for PiDog.
Bypasses webots_ros2_driver - uses Webots Python API directly.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from controller import Robot


class PiDogDirectController(Node):
    """Direct Webots controller with ROS2 integration."""

    def __init__(self):
        super().__init__('pidog_direct_controller')

        # Initialize Webots robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get all motors
        self.motor_names = [
            'body_to_front_left_leg_b', 'front_left_leg_b_to_a',
            'body_to_front_right_leg_b', 'front_right_leg_b_to_a',
            'body_to_back_left_leg_b', 'back_left_leg_b_to_a',
            'body_to_back_right_leg_b', 'back_right_leg_b_to_a',
            'motor_8_to_tail',
            'neck1_to_motor_9', 'neck2_to_motor_10', 'neck3_to_motor_11'
        ]

        # Standing pose: shoulders straight, knees bent to lift body
        initial_positions = [
            0.0, -0.8,  # front_left: shoulder straight, knee bent 45° down
            0.0, -0.8,  # front_right: shoulder straight, knee bent 45° down
            0.0, -0.8,  # back_left: shoulder straight, knee bent 45° down
            0.0, -0.8,  # back_right: shoulder straight, knee bent 45° down
            0.0,        # tail neutral
            0.2, 0.2, 0.2  # neck joints (tilt head up)
        ]

        self.motors = []
        for i, name in enumerate(self.motor_names):
            motor = self.robot.getDevice(name)
            if motor is None:
                self.get_logger().error(f"Motor not found: {name}")
            else:
                motor.setPosition(initial_positions[i])
                self.motors.append(motor)

        self.get_logger().info(f"Initialized {len(self.motors)} motors")

        # Current joint positions (start with initial pose)
        self.joint_positions = initial_positions.copy()

        # Subscribe to joint commands
        self.create_subscription(JointState, 'motor_pos', self.motor_callback, 10)

        # Create timer for Webots step
        self.create_timer(self.timestep / 1000.0, self.step_callback)

        self.get_logger().info("PiDog Direct Controller started!")

    def motor_callback(self, msg):
        """Receive joint commands from gait generator."""
        if len(msg.position) == len(self.motors):
            self.joint_positions = list(msg.position)

    def step_callback(self):
        """Webots simulation step - apply motor commands."""
        # Apply positions to motors
        for i, motor in enumerate(self.motors):
            motor.setPosition(self.joint_positions[i])

        # Step the simulation
        if self.robot.step(self.timestep) == -1:
            self.get_logger().info("Simulation stopped")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    controller = PiDogDirectController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
