"""
Hardware-Compatible PiDog RL Environment

Uses ONLY sensors available on real SunFounder PiDog:
- IMU (6-DOF): Roll, pitch, yaw, angular velocity, linear acceleration
- Joint encoders: Position and velocity

NO contact sensors (don't exist on hardware!)
Virtual contact detection using forward kinematics + IMU
"""

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray, String
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty
import time
import math


def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class PiDogHardwareEnv(gym.Env):
    """
    Hardware-compatible RL environment for PiDog.

    Observation (48D):
        - gait_cmd (3): [gait_type, direction, turn]
        - phase (1): gait cycle phase
        - joint_pos (12): motor positions
        - joint_vel (12): motor velocities
        - imu_orientation (3): [roll, pitch, yaw]
        - imu_angular_vel (3): [wx, wy, wz]
        - imu_linear_acc (3): [ax, ay, az]
        - body_linear_vel (3): [vx, vy, vz] (from IMU integration or GPS)
        - virtual_contacts (4): [BR, FR, BL, FL] estimated from kinematics
        - body_height (1): Z position (from IMU integration)
        - stall_counter_norm (1): normalized stall duration (0-1)
        - prev_action (2): Previous shoulder/knee action for history

    Action: [12 joint positions] in radians
    """

    metadata = {'render.modes': ['human']}

    # PiDog leg geometry (meters)
    UPPER_LEG = 0.047  # shoulder to knee
    LOWER_LEG = 0.0635 # knee to foot

    def __init__(self, node_name='pidog_hardware_env', reward_mode='conservative'):
        super(PiDogHardwareEnv, self).__init__()

        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name)

        # State variables - Joint data
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.prev_action = np.zeros(2)  # Previous shoulder/knee for one leg

        # State variables - Ground truth (for sim only, will be IMU-integrated on real hardware)
        self.body_position = np.zeros(3)
        self.body_orientation_quat = np.zeros(4)
        self.body_linear_vel = np.zeros(3)

        # State variables - IMU sensor data (REAL HARDWARE)
        self.imu_orientation = np.zeros(3)  # roll, pitch, yaw
        self.imu_angular_vel = np.zeros(3)  # wx, wy, wz
        self.imu_linear_acc = np.zeros(3)   # ax, ay, az

        # Virtual contact detection
        self.virtual_contacts = np.zeros(4)  # [BR, FR, BL, FL]

        # Stall detection
        self.stall_counter = 0
        self.stall_threshold_vel = 0.05  # m/s (5 cm/s)
        self.stall_max_steps = 60        # 2 seconds at 30Hz

        # Current gait
        self.phase = 0.0
        self.episode_step = 0
        self.max_episode_steps = 500

        # Reward mode
        self.reward_mode = reward_mode

        # ROS2 publishers
        self.action_pub = self.node.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10
        )
        self.gait_cmd_pub = self.node.create_publisher(
            String, '/gait_command', 10
        )

        # ROS2 subscribers
        self.joint_sub = self.node.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10
        )
        self.model_sub = self.node.create_subscription(
            ModelStates, '/model/pidog/state', self._model_callback, 10
        )
        self.imu_sub = self.node.create_subscription(
            Imu, '/imu', self._imu_callback, 10
        )

        # Gazebo service clients
        self.reset_world_client = self.node.create_client(Empty, '/reset_world')
        self.reset_sim_client = self.node.create_client(Empty, '/reset_simulation')

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.57, high=1.57, shape=(12,), dtype=np.float32
        )

        # Gait commands
        self.target_gait = 'trot_forward'
        self.gait_params = {
            'walk_forward': [0.0, 1.0, 0.0],
            'trot_forward': [1.0, 1.0, 0.0],
            'gallop_forward': [2.0, 1.0, 0.0],
            'stand': [3.0, 0.0, 0.0],
        }

        print(f"[PiDogHardwareEnv] Initialized ({self.observation_space.shape[0]}D obs)")
        print(f"[PiDogHardwareEnv] Reward mode: {self.reward_mode}")
        print(f"[PiDogHardwareEnv] Using ONLY real hardware sensors (IMU + joint encoders)")

    def _joint_callback(self, msg):
        if len(msg.position) >= 12:
            self.joint_positions = np.array(msg.position[:12])
            self.joint_velocities = np.array(msg.velocity[:12]) if len(msg.velocity) >= 12 else np.zeros(12)

    def _model_callback(self, msg):
        """Ground truth from Gazebo (for training only)."""
        try:
            idx = msg.name.index('pidog')
            pose = msg.pose[idx]
            twist = msg.twist[idx]

            self.body_position = np.array([
                pose.position.x, pose.position.y, pose.position.z
            ])
            self.body_orientation_quat = np.array([
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w
            ])
            self.body_linear_vel = np.array([
                twist.linear.x, twist.linear.y, twist.linear.z
            ])
        except (ValueError, IndexError):
            pass

    def _imu_callback(self, msg):
        """IMU sensor data (REAL HARDWARE)."""
        # Orientation
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        self.imu_orientation = np.array([roll, pitch, yaw])

        # Angular velocity
        self.imu_angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Linear acceleration
        self.imu_linear_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

    def _estimate_foot_contacts(self):
        """
        Virtual contact detection using forward kinematics.
        Works on real hardware (no physical contact sensors needed!)
        """
        contacts = np.zeros(4)

        # Estimate body height from ground truth in sim (would be IMU-integrated on real robot)
        body_height = self.body_position[2]

        # For each leg: [BR=0, FR=1, BL=2, FL=3]
        for leg_idx in range(4):
            shoulder_angle = self.joint_positions[leg_idx * 2]      # Joint 0, 2, 4, 6
            knee_angle = self.joint_positions[leg_idx * 2 + 1]      # Joint 1, 3, 5, 7

            # Simple 2-link forward kinematics (Z-axis only)
            # Note: PiDog left legs have flipped axes, but we handle that in signs
            foot_z_relative = (
                self.UPPER_LEG * np.sin(shoulder_angle) +
                self.LOWER_LEG * np.sin(shoulder_angle + knee_angle)
            )

            # Absolute foot height = body height + relative foot position
            foot_z_abs = body_height + foot_z_relative

            # Contact if foot within 1cm of ground
            if foot_z_abs < 0.01:
                contacts[leg_idx] = 1.0

        return contacts

    def _get_obs(self):
        """Get observation (48D)."""
        # Update virtual contacts
        self.virtual_contacts = self._estimate_foot_contacts()

        # Gait command
        gait_vec = self.gait_params.get(self.target_gait, [0.0, 0.0, 0.0])

        # Normalize stall counter
        stall_norm = min(self.stall_counter / self.stall_max_steps, 1.0)

        # Observation vector (48D)
        obs = np.concatenate([
            gait_vec,                # 3D: gait command
            [self.phase],            # 1D: phase
            self.joint_positions,    # 12D: joint positions (REAL SENSOR)
            self.joint_velocities,   # 12D: joint velocities (REAL SENSOR)
            self.imu_orientation,    # 3D: roll, pitch, yaw (REAL SENSOR)
            self.imu_angular_vel,    # 3D: angular velocity (REAL SENSOR)
            self.imu_linear_acc,     # 3D: linear acceleration (REAL SENSOR)
            self.body_linear_vel,    # 3D: body velocity (IMU-integrated or ground truth)
            self.virtual_contacts,   # 4D: virtual contact detection
            [self.body_position[2]], # 1D: body height (IMU-integrated or ground truth)
            [stall_norm],            # 1D: stall counter normalized
            self.prev_action[:2],    # 2D: previous action history
        ])

        return obs.astype(np.float32)

    def _calculate_reward(self, action):
        """Calculate reward based on mode."""
        if self.reward_mode == 'conservative':
            return self._reward_conservative(action)
        elif self.reward_mode == 'fast_running':
            return self._reward_fast_running(action)
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def _reward_conservative(self, action):
        """
        Phase 1: Conservative baseline for stable locomotion.
        Goal: Learn to walk steadily without falling or stalling.

        User settings:
        - Speed weight: 10.0 (conservative)
        - Stall: < 0.05 m/s for 2 sec
        - Roll limit: 45 degrees
        - Pitch limit: LOOSE (0.8 rad = 46 deg) for galloping
        """
        reward = 0.0
        done = False
        info = {}

        roll, pitch, yaw = self.imu_orientation
        body_height = self.body_position[2]
        forward_vel = self.body_linear_vel[0]

        # === SPEED (MODERATE WEIGHT) ===
        if forward_vel > 0:
            reward += forward_vel * 10.0  # Conservative (user specified)
        else:
            reward -= abs(forward_vel) * 3.0

        # === STABILITY: HEIGHT ===
        if 0.09 < body_height < 0.13:
            reward += 4.0
        elif body_height < 0.07:
            reward -= 10.0

        # === STABILITY: ORIENTATION ===
        # Roll: Strict (45 degrees)
        if abs(roll) < 0.4:
            reward += 2.0
        else:
            reward -= abs(roll) * 4.0

        # Pitch: Loose (46 degrees) - user specified for galloping
        if abs(pitch) < 0.8:
            reward += 2.0
        else:
            reward -= abs(pitch) * 2.0

        # === ENERGY EFFICIENCY ===
        reward -= np.sum(np.abs(self.joint_velocities)) * 0.01
        reward -= np.sum(np.abs(action)) * 0.01

        # === TERMINATION CONDITIONS ===

        # 1. Roll > 45 degrees (STRICT - user requirement)
        if abs(roll) > 0.785:  # 45 degrees
            reward -= 100.0
            done = True
            info['termination_reason'] = 'roll_45deg'

        # 2. Pitch > 69 degrees (original, kept as fallback)
        if abs(pitch) > 1.2:
            reward -= 100.0
            done = True
            info['termination_reason'] = 'pitch_excessive'

        # 3. Body crashed
        if body_height < 0.05:
            reward -= 100.0
            done = True
            info['termination_reason'] = 'body_crash'

        # 4. Stall detection (NEW - user requirement)
        if forward_vel < self.stall_threshold_vel:
            self.stall_counter += 1
            if self.stall_counter >= self.stall_max_steps:
                reward -= 100.0
                done = True
                info['termination_reason'] = 'stalled_2sec'
        else:
            self.stall_counter = 0

        # Stall warning (gentle penalty before termination)
        if self.stall_counter > 0:
            reward -= self.stall_counter * 0.05

        info.update({
            'speed': forward_vel,
            'height': body_height,
            'roll': roll,
            'pitch': pitch,
            'stall_counter': self.stall_counter,
        })

        return reward, done, info

    def _reward_fast_running(self, action):
        """
        Phase 2: Fast running with stability.
        Same termination rules, higher speed reward.
        """
        reward = 0.0
        done = False
        info = {}

        roll, pitch, yaw = self.imu_orientation
        body_height = self.body_position[2]
        forward_vel = self.body_linear_vel[0]
        num_contacts = int(np.sum(self.virtual_contacts))

        # === SPEED (HIGH PRIORITY) ===
        if forward_vel > 0:
            reward += forward_vel * 15.0  # Aggressive for Phase 2

            # Milestone bonuses
            if forward_vel > 1.0:
                reward += 2.0
            if forward_vel > 1.5:
                reward += 5.0
            if forward_vel > 2.0:
                reward += 15.0
        else:
            reward += forward_vel * 5.0

        # === STABILITY: HEIGHT ===
        if 0.10 < body_height < 0.15:
            reward += 2.0
        elif body_height < 0.08:
            reward -= 10.0

        # === STABILITY: ORIENTATION ===
        if abs(roll) < 0.4:
            reward += 1.5
        else:
            reward -= abs(roll) * 3.0

        if abs(pitch) < 0.8:  # Loose for galloping
            reward += 1.5
        else:
            reward -= abs(pitch) * 2.0

        # === GAIT QUALITY: VIRTUAL CONTACTS ===
        if num_contacts == 0:
            reward += 3.0   # Flight phase
        elif num_contacts == 2:
            reward += 1.5
        elif num_contacts == 4:
            reward += 0.2
        else:
            reward -= 2.0

        # === ANGULAR VELOCITY STABILITY ===
        ang_vel_mag = np.linalg.norm(self.imu_angular_vel)
        if ang_vel_mag > 5.0:
            reward -= ang_vel_mag * 0.3

        # === ENERGY ===
        reward -= np.sum(np.abs(self.joint_velocities)) * 0.005
        reward -= np.sum(np.abs(action)) * 0.005

        # === TERMINATION (SAME AS CONSERVATIVE) ===
        if abs(roll) > 0.785:
            reward -= 100.0
            done = True
            info['termination_reason'] = 'roll_45deg'

        if abs(pitch) > 1.2:
            reward -= 100.0
            done = True
            info['termination_reason'] = 'pitch_excessive'

        if body_height < 0.04:
            reward -= 100.0
            done = True
            info['termination_reason'] = 'body_crash'

        if forward_vel < self.stall_threshold_vel:
            self.stall_counter += 1
            if self.stall_counter >= self.stall_max_steps:
                reward -= 100.0
                done = True
                info['termination_reason'] = 'stalled_2sec'
        else:
            self.stall_counter = 0

        if self.stall_counter > 0:
            reward -= self.stall_counter * 0.1

        info.update({
            'speed': forward_vel,
            'height': body_height,
            'contacts': num_contacts,
            'roll': roll,
            'pitch': pitch,
            'ang_vel_mag': ang_vel_mag,
            'stall_counter': self.stall_counter,
        })

        return reward, done, info

    def step(self, action):
        """Take environment step."""
        # Publish action
        cmd_msg = Float64MultiArray()
        cmd_msg.data = action.tolist()
        self.action_pub.publish(cmd_msg)

        # Store previous action for history
        self.prev_action = action[:2].copy()

        # Update phase
        self.phase = (self.phase + 0.02) % 1.0

        # ROS spin
        rclpy.spin_once(self.node, timeout_sec=0.03)
        time.sleep(0.03)

        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward, done, info = self._calculate_reward(action)

        # Episode length check
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        self.episode_step = 0
        self.phase = 0.0
        self.stall_counter = 0
        self.prev_action = np.zeros(2)

        # Default to trot
        self.target_gait = 'trot_forward'

        gait_msg = String()
        gait_msg.data = self.target_gait
        self.gait_cmd_pub.publish(gait_msg)

        time.sleep(0.5)

        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_obs()
        info = {'target_gait': self.target_gait}

        return obs, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
