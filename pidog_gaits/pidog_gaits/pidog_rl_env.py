"""
Gazebo RL Environment for PiDog

Gym-compatible environment for reinforcement learning training.
Interfaces with Gazebo simulation to provide observations and apply actions.
"""

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates, ContactsState
from gazebo_msgs.srv import SetModelState, GetModelState
from std_srvs.srv import Empty
import time
import math


class PiDogGazeboEnv(gym.Env):
    """
    Gym environment for PiDog in Gazebo.

    Observation: [gait_cmd (4), joint_pos (12), joint_vel (12), body_pose (7), contact (1)]
                 Total: 36 dimensions

    Action: [12 joint positions] in radians

    Rewards:
        - Stability (upright, head not touching ground)
        - Task completion (walking forward, turning, etc.)
        - Energy efficiency (small penalties for large movements)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, node_name='pidog_rl_env', headless=False):
        super(PiDogGazeboEnv, self).__init__()

        # Initialize ROS2 if not already initialized
        if not rclpy.ok():
            rclpy.init()

        # Create ROS2 node
        self.node = Node(node_name)

        # State variables
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.body_position = np.zeros(3)  # x, y, z
        self.body_orientation = np.zeros(4)  # quaternion
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.head_contact = False

        # Current gait command
        self.current_gait = [0.0, 0.0, 0.0, 0.0]  # [gait_type, direction, turn, phase]
        self.phase = 0.0

        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 500  # ~16 seconds at 30Hz

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

        # ROS2 subscribers
        self.joint_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_callback,
            10
        )

        self.model_sub = self.node.create_subscription(
            ModelStates,
            '/model/pidog/state',
            self._model_callback,
            10
        )

        # Gazebo service clients
        self.reset_world_client = self.node.create_client(Empty, '/reset_world')
        self.reset_sim_client = self.node.create_client(Empty, '/reset_simulation')

        # Gym spaces
        # Observation: [gait_cmd(4), joint_pos(12), joint_vel(12), body_pose(7), phase(1)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(36,),
            dtype=np.float32
        )

        # Action: 12 joint positions
        self.action_space = gym.spaces.Box(
            low=-1.57,  # ~-90 degrees
            high=1.57,   # ~+90 degrees
            shape=(12,),
            dtype=np.float32
        )

        # Gait command for current episode
        self.target_gait = 'walk_forward'
        self.gait_params = {
            'walk_forward': [0.0, 1.0, 0.0],   # [gait_type, direction, turn]
            'walk_backward': [0.0, -1.0, 0.0],
            'walk_left': [0.0, 0.0, -1.0],
            'walk_right': [0.0, 0.0, 1.0],
            'trot_forward': [1.0, 1.0, 0.0],
            'trot_backward': [1.0, -1.0, 0.0],
            'stand': [2.0, 0.0, 0.0],
            'sit': [2.0, 0.0, 0.0],
        }

        print(f"[PiDogEnv] Environment initialized")

    def _joint_callback(self, msg):
        """Update joint states from ROS topic."""
        if len(msg.position) >= 12:
            self.joint_positions = np.array(msg.position[:12])
            self.joint_velocities = np.array(msg.velocity[:12]) if len(msg.velocity) >= 12 else np.zeros(12)

    def _model_callback(self, msg):
        """Update model state (body position/orientation) from Gazebo."""
        try:
            # Find PiDog model in the list
            idx = msg.name.index('pidog')
            pose = msg.pose[idx]
            twist = msg.twist[idx]

            self.body_position = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ])

            self.body_orientation = np.array([
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

            self.body_angular_vel = np.array([
                twist.angular.x,
                twist.angular.y,
                twist.angular.z
            ])

            # Check if head is touching ground (simple heuristic)
            # Assuming head is at body + some offset
            head_z = self.body_position[2] + 0.05  # Rough estimate
            self.head_contact = head_z < 0.02  # 2cm threshold

        except (ValueError, IndexError):
            pass

    def _get_obs(self):
        """Get current observation."""
        # Get current gait parameters
        gait_vec = self.gait_params.get(self.target_gait, [0.0, 0.0, 0.0])

        # Construct observation
        obs = np.concatenate([
            gait_vec,              # Gait command (3)
            [self.phase],          # Phase (1)
            self.joint_positions,  # Joint positions (12)
            self.joint_velocities, # Joint velocities (12)
            self.body_position,    # Body position (3)
            self.body_orientation, # Body orientation (4)
            [float(self.head_contact)],  # Head contact (1)
        ])

        return obs.astype(np.float32)

    def _calculate_reward(self, action):
        """
        Calculate reward based on robot state.

        Rewards:
        - Stability: upright, not falling
        - Task: moving in commanded direction
        - Efficiency: smooth movements
        """
        reward = 0.0
        done = False
        info = {}

        # === STABILITY REWARDS ===

        # 1. Upright reward (body Z height)
        target_height = 0.10  # 10cm standing height
        height_error = abs(self.body_position[2] - target_height)
        if self.body_position[2] > 0.08:  # At least 8cm high
            reward += 1.0 - height_error * 5.0  # Scale down error
        else:
            reward -= 2.0  # Penalty for being too low

        # 2. Head not touching ground (IMPORTANT!)
        if self.head_contact:
            reward -= 5.0  # Big penalty
            info['head_contact'] = True
        else:
            reward += 0.5

        # 3. Body orientation (pitch and roll should be small)
        # Convert quaternion to euler angles
        qx, qy, qz, qw = self.body_orientation
        roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = math.asin(2*(qw*qy - qz*qx))

        if abs(roll) < 0.3 and abs(pitch) < 0.3:  # Within ~17 degrees
            reward += 1.0
        else:
            reward -= abs(roll) + abs(pitch)  # Penalty for tilting

        # Check if fallen over
        if abs(roll) > 1.0 or abs(pitch) > 1.0 or self.body_position[2] < 0.05:
            reward -= 10.0
            done = True
            info['fallen'] = True

        # === TASK REWARDS ===

        if self.target_gait in ['walk_forward', 'trot_forward']:
            # Reward forward velocity
            reward += self.body_linear_vel[0] * 2.0  # Forward = +X

        elif self.target_gait in ['walk_backward', 'trot_backward']:
            # Reward backward velocity
            reward -= self.body_linear_vel[0] * 2.0  # Backward = -X

        elif self.target_gait in ['walk_left']:
            # Reward left velocity
            reward += self.body_linear_vel[1] * 2.0  # Left = +Y

        elif self.target_gait in ['walk_right']:
            # Reward right velocity
            reward -= self.body_linear_vel[1] * 2.0  # Right = -Y

        elif self.target_gait in ['stand', 'sit']:
            # Reward staying still
            velocity_mag = np.linalg.norm(self.body_linear_vel)
            reward += max(0, 0.5 - velocity_mag)

        # === EFFICIENCY PENALTIES ===

        # Penalize large joint velocities (energy efficiency)
        joint_vel_penalty = np.sum(np.abs(self.joint_velocities)) * 0.01
        reward -= joint_vel_penalty

        # Penalize large actions (smooth control)
        action_penalty = np.sum(np.abs(action)) * 0.01
        reward -= action_penalty

        # === INFO ===
        info.update({
            'body_z': self.body_position[2],
            'roll': roll,
            'pitch': pitch,
            'forward_vel': self.body_linear_vel[0],
            'episode_step': self.episode_step,
        })

        return reward, done, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Array of 12 joint positions

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Publish action to Gazebo
        cmd_msg = Float64MultiArray()
        cmd_msg.data = action.tolist()
        self.action_pub.publish(cmd_msg)

        # Update phase (for gait cycle)
        self.phase = (self.phase + 0.02) % 1.0  # Increment by ~2% per step

        # Spin ROS to get new observations
        rclpy.spin_once(self.node, timeout_sec=0.03)  # ~30Hz update
        time.sleep(0.03)

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

        # Randomly select a gait for this episode
        gait_options = ['walk_forward', 'trot_forward', 'stand']
        self.target_gait = np.random.choice(gait_options)

        # Publish gait command
        gait_msg = String()
        gait_msg.data = self.target_gait
        self.gait_cmd_pub.publish(gait_msg)

        # Reset Gazebo simulation (optional - can be slow)
        # For now, just wait for robot to stabilize
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
