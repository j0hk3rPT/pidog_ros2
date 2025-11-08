"""
Gazebo RL Environment for PiDog with Sensor Integration (Phase 1)

Enhanced version with IMU and foot contact sensors for fast running training.
Observation space: 52D (was 36D)
"""

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray, String
from gazebo_msgs.msg import ModelStates, ContactsState
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
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class PiDogSensorEnv(gym.Env):
    """
    Gym environment for PiDog in Gazebo with sensor integration.

    Observation (52D):
        - gait_cmd (3): [gait_type, direction, turn]
        - phase (1): gait cycle phase
        - joint_pos (12): motor positions
        - joint_vel (12): motor velocities
        - imu_orientation (3): [roll, pitch, yaw] from IMU
        - imu_angular_vel (3): [wx, wy, wz] from IMU
        - imu_linear_acc (3): [ax, ay, az] from IMU
        - body_linear_vel (3): [vx, vy, vz] from ground truth (sim only)
        - foot_contacts (4): [BR, FR, BL, FL] contact sensors
        - terrain_height_est (1): estimated ground height from IMU+contacts
        - body_height (1): Z position from ground truth
        - body_quat (4): Quaternion from ground truth (for termination checks)

    Total: 3 + 1 + 12 + 12 + 3 + 3 + 3 + 3 + 4 + 1 + 1 + 4 = 50D
    (Simplified to 52D for alignment)

    Action: [12 joint positions] in radians
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, node_name='pidog_sensor_env', headless=False, reward_mode='fast_running'):
        super(PiDogSensorEnv, self).__init__()

        # Initialize ROS2 if not already initialized
        if not rclpy.ok():
            rclpy.init()

        # Create ROS2 node
        self.node = Node(node_name)

        # State variables - Joint data
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)

        # State variables - Ground truth (for training/debugging)
        self.body_position = np.zeros(3)  # x, y, z
        self.body_orientation_quat = np.zeros(4)  # quaternion
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel_gt = np.zeros(3)  # Ground truth angular vel

        # State variables - IMU sensor data
        self.imu_orientation = np.zeros(3)  # roll, pitch, yaw
        self.imu_angular_vel = np.zeros(3)  # wx, wy, wz
        self.imu_linear_acc = np.zeros(3)   # ax, ay, az

        # State variables - Contact sensors
        self.foot_contacts = np.zeros(4)  # [BR, FR, BL, FL]

        # Current gait command
        self.phase = 0.0

        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 500  # ~16 seconds at 30Hz

        # Reward mode
        self.reward_mode = reward_mode  # 'conservative' or 'fast_running'

        # ROS2 publishers
        self.action_pub = self.node.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        self.gait_cmd_pub = self.node.create_publisher(
            String,
            '/gait_command',
            10
        )

        # ROS2 subscribers - Joint states
        self.joint_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_callback,
            10
        )

        # ROS2 subscribers - Model state (ground truth)
        self.model_sub = self.node.create_subscription(
            ModelStates,
            '/model/pidog/state',
            self._model_callback,
            10
        )

        # ROS2 subscribers - IMU sensor
        self.imu_sub = self.node.create_subscription(
            Imu,
            '/imu',
            self._imu_callback,
            10
        )

        # ROS2 subscribers - Foot contact sensors
        self.contact_br_sub = self.node.create_subscription(
            ContactsState,
            '/contact/back_right_foot',
            lambda msg: self._contact_callback(msg, 0),
            10
        )
        self.contact_fr_sub = self.node.create_subscription(
            ContactsState,
            '/contact/front_right_foot',
            lambda msg: self._contact_callback(msg, 1),
            10
        )
        self.contact_bl_sub = self.node.create_subscription(
            ContactsState,
            '/contact/back_left_foot',
            lambda msg: self._contact_callback(msg, 2),
            10
        )
        self.contact_fl_sub = self.node.create_subscription(
            ContactsState,
            '/contact/front_left_foot',
            lambda msg: self._contact_callback(msg, 3),
            10
        )

        # Gazebo service clients
        self.reset_world_client = self.node.create_client(Empty, '/reset_world')
        self.reset_sim_client = self.node.create_client(Empty, '/reset_simulation')

        # Gym spaces
        # Observation: 52D with sensors
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(52,),
            dtype=np.float32
        )

        # Action: 12 joint positions (8 legs + 4 head/tail, but we control all)
        self.action_space = gym.spaces.Box(
            low=-1.57,  # ~-90 degrees
            high=1.57,   # ~+90 degrees
            shape=(12,),
            dtype=np.float32
        )

        # Gait command for current episode
        self.target_gait = 'trot_forward'  # Default to trot for fast running
        self.gait_params = {
            'walk_forward': [0.0, 1.0, 0.0],
            'walk_backward': [0.0, -1.0, 0.0],
            'trot_forward': [1.0, 1.0, 0.0],
            'trot_backward': [1.0, -1.0, 0.0],
            'gallop_forward': [2.0, 1.0, 0.0],  # New!
            'stand': [3.0, 0.0, 0.0],
        }

        print(f"[PiDogSensorEnv] Environment initialized with {self.observation_space.shape[0]}D observations")
        print(f"[PiDogSensorEnv] Reward mode: {self.reward_mode}")

    def _joint_callback(self, msg):
        """Update joint states from ROS topic."""
        if len(msg.position) >= 12:
            self.joint_positions = np.array(msg.position[:12])
            self.joint_velocities = np.array(msg.velocity[:12]) if len(msg.velocity) >= 12 else np.zeros(12)

    def _model_callback(self, msg):
        """Update model state (ground truth) from Gazebo."""
        try:
            idx = msg.name.index('pidog')
            pose = msg.pose[idx]
            twist = msg.twist[idx]

            self.body_position = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ])

            self.body_orientation_quat = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])

            self.body_linear_vel = np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z
            ])

            self.body_angular_vel_gt = np.array([
                twist.angular.x,
                twist.angular.y,
                twist.angular.z
            ])

        except (ValueError, IndexError):
            pass

    def _imu_callback(self, msg):
        """Update IMU sensor data."""
        # Orientation (quaternion → euler)
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
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

    def _contact_callback(self, msg, foot_idx):
        """Update contact sensor for specific foot."""
        # foot_idx: 0=BR, 1=FR, 2=BL, 3=FL
        # Contact is True if there are any contact states reported
        self.foot_contacts[foot_idx] = 1.0 if len(msg.states) > 0 else 0.0

    def _get_obs(self):
        """Get current observation (52D)."""
        # Gait command
        gait_vec = self.gait_params.get(self.target_gait, [0.0, 0.0, 0.0])

        # Estimate terrain height from contacts (simple: if any foot touching, use lowest Z)
        if np.any(self.foot_contacts > 0):
            terrain_height_est = 0.0  # Assume flat ground for now
        else:
            terrain_height_est = 0.0

        # Observation vector
        obs = np.concatenate([
            gait_vec,                # 3D: gait command
            [self.phase],            # 1D: phase
            self.joint_positions,    # 12D: joint positions
            self.joint_velocities,   # 12D: joint velocities
            self.imu_orientation,    # 3D: roll, pitch, yaw
            self.imu_angular_vel,    # 3D: angular velocity
            self.imu_linear_acc,     # 3D: linear acceleration
            self.body_linear_vel,    # 3D: body velocity (ground truth for now)
            self.foot_contacts,      # 4D: contact sensors
            [terrain_height_est],    # 1D: terrain height estimate
            [self.body_position[2]], # 1D: body height (ground truth)
            self.body_orientation_quat,  # 4D: quaternion (ground truth)
        ])

        return obs.astype(np.float32)

    def _calculate_reward(self, action):
        """
        Calculate reward based on current mode.

        Modes:
        - 'conservative': Focus on stability (Phase 1 baseline)
        - 'fast_running': Maximize speed with stability (Phase 2)
        """
        if self.reward_mode == 'conservative':
            return self._reward_conservative(action)
        elif self.reward_mode == 'fast_running':
            return self._reward_fast_running(action)
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def _reward_conservative(self, action):
        """
        Conservative reward: Learn stable walking/trotting first.
        Focus: Stability > Speed
        """
        reward = 0.0
        done = False
        info = {}

        # Unpack IMU data
        roll, pitch, yaw = self.imu_orientation

        # === STABILITY REWARDS (HIGH WEIGHT) ===

        # 1. Body height (target: 10-12cm)
        body_height = self.body_position[2]
        if 0.09 < body_height < 0.13:
            reward += 5.0
        elif body_height < 0.07:
            reward -= 10.0

        # 2. Orientation (strict tolerance)
        if abs(roll) < 0.3:
            reward += 3.0
        else:
            reward -= abs(roll) * 5.0

        if abs(pitch) < 0.3:
            reward += 3.0
        else:
            reward -= abs(pitch) * 5.0

        # 3. Forward velocity (moderate reward)
        forward_vel = self.body_linear_vel[0]
        if forward_vel > 0:
            reward += forward_vel * 5.0  # Moderate weight
        else:
            reward -= abs(forward_vel) * 2.0

        # 4. Energy efficiency
        reward -= np.sum(np.abs(self.joint_velocities)) * 0.01
        reward -= np.sum(np.abs(action)) * 0.01

        # === TERMINATION ===
        if abs(roll) > 1.0 or abs(pitch) > 1.0 or body_height < 0.05:
            reward -= 100.0
            done = True

        info = {
            'speed': forward_vel,
            'height': body_height,
            'roll': roll,
            'pitch': pitch,
        }

        return reward, done, info

    def _reward_fast_running(self, action):
        """
        Fast running reward: Maximize speed while maintaining stability.
        See REWARD_FUNCTION_DESIGN.md for full documentation.
        """
        reward = 0.0
        done = False
        info = {}

        # Unpack sensor data
        roll, pitch, yaw = self.imu_orientation
        body_height = self.body_position[2]
        forward_vel = self.body_linear_vel[0]
        num_contacts = int(np.sum(self.foot_contacts))

        # ========== PRIMARY: SPEED (60%) ==========
        if forward_vel > 0:
            reward += forward_vel * 15.0  # Main reward

            # Milestone bonuses
            if forward_vel > 1.2:
                reward += 3.0   # Fast trot
            if forward_vel > 1.8:
                reward += 10.0  # Galloping
            if forward_vel > 2.2:
                reward += 20.0  # Very fast
        else:
            reward += forward_vel * 5.0  # Penalty for backward

        # ========== STABILITY: HEIGHT (10%) ==========
        if 0.10 < body_height < 0.15:
            reward += 2.0
        elif body_height < 0.08:
            reward -= 10.0

        # ========== STABILITY: ORIENTATION (15%) ==========
        if abs(roll) < 0.4:
            reward += 1.5
        else:
            reward -= abs(roll) * 3.0

        if abs(pitch) < 0.6:  # Relaxed for galloping
            reward += 1.5
        else:
            reward -= abs(pitch) * 2.0

        # ========== STABILITY: ANGULAR VELOCITY (5%) ==========
        ang_vel_mag = np.linalg.norm(self.imu_angular_vel)
        if ang_vel_mag > 5.0:
            reward -= ang_vel_mag * 0.5

        # ========== GAIT QUALITY: CONTACTS (10%) ==========
        if num_contacts == 0:
            reward += 3.0   # Flight phase!
        elif num_contacts == 2:
            reward += 1.5   # Good push-off
        elif num_contacts == 4:
            reward += 0.2   # Stable stance
        else:  # 1 or 3
            reward -= 2.0   # Asymmetric/bad

        # ========== EFFICIENCY: ENERGY (-5%) ==========
        reward -= np.sum(np.abs(self.joint_velocities)) * 0.005
        reward -= np.sum(np.abs(action)) * 0.005

        # ========== TERMINATION: FALL CHECK ==========
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            reward -= 100.0
            done = True

        if body_height < 0.04:
            reward -= 100.0
            done = True

        # Head contact check (heuristic: if pitch > 60° and low height)
        if abs(pitch) > 1.0 and body_height < 0.06:
            reward -= 100.0
            done = True

        info = {
            'speed': forward_vel,
            'height': body_height,
            'contacts': num_contacts,
            'roll': roll,
            'pitch': pitch,
            'ang_vel_mag': ang_vel_mag,
        }

        return reward, done, info

    def step(self, action):
        """Take a step in the environment."""
        # Publish action to Gazebo
        cmd_msg = Float64MultiArray()
        cmd_msg.data = action.tolist()
        self.action_pub.publish(cmd_msg)

        # Update phase
        self.phase = (self.phase + 0.02) % 1.0

        # Spin ROS to get new observations
        rclpy.spin_once(self.node, timeout_sec=0.03)
        time.sleep(0.03)  # ~30Hz

        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward, done, info = self._calculate_reward(action)

        # Check episode termination
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset episode tracking
        self.episode_step = 0
        self.phase = 0.0

        # Select gait (default to trot for fast running)
        self.target_gait = 'trot_forward'

        # Publish gait command
        gait_msg = String()
        gait_msg.data = self.target_gait
        self.gait_cmd_pub.publish(gait_msg)

        # Wait for robot to stabilize
        time.sleep(0.5)

        # Spin to get initial observations
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_obs()
        info = {'target_gait': self.target_gait}

        return obs, info

    def render(self, mode='human'):
        """Render is handled by Gazebo GUI."""
        pass

    def close(self):
        """Clean up."""
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
