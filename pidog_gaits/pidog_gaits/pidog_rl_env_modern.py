"""
Gazebo RL Environment for PiDog - Modern Gazebo (Jetty) Compatible

Gym-compatible environment for reinforcement learning training.
Works with Gazebo Jetty and ROS Rolling using TF2 for pose tracking.
"""

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener, Buffer, TransformException
import time
import math
from scipy.spatial.transform import Rotation


class PiDogGazeboEnv(gym.Env):
    """
    Gym environment for PiDog in Gazebo (Modern Gazebo Compatible).

    Observation: [gait_cmd (4), joint_pos (12), joint_vel (12), body_pose (7), imu (7)]
                 Total: 42 dimensions

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

        # TF2 for getting body pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # State variables
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.body_position = np.zeros(3)  # x, y, z
        self.body_orientation = np.zeros(4)  # quaternion (x,y,z,w)
        self.imu_orientation = np.zeros(4)  # from IMU
        self.imu_angular_vel = np.zeros(3)
        self.last_joint_positions = np.zeros(12)

        # Current gait command
        self.current_gait = [0.0, 0.0, 0.0, 0.0]  # [gait_type, direction, turn, phase]
        self.phase = 0.0

        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 500  # ~16 seconds at 30Hz
        self.control_freq = 30.0  # Hz
        self.dt = 1.0 / self.control_freq

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

        self.imu_sub = self.node.create_subscription(
            Imu,
            '/imu',
            self._imu_callback,
            10
        )

        # Gym spaces
        # Observation: [gait_cmd(4), joint_pos(12), joint_vel(12), body_pos(3), body_orient(4), imu_orient(4), imu_angvel(3)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(42,),
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

        print(f"[PiDogEnv] Environment initialized (Modern Gazebo)")

    def _joint_callback(self, msg):
        """Update joint states from ROS topic."""
        if len(msg.position) >= 12:
            self.last_joint_positions = self.joint_positions.copy()
            self.joint_positions = np.array(msg.position[:12])
            self.joint_velocities = np.array(msg.velocity[:12]) if len(msg.velocity) >= 12 else np.zeros(12)

    def _imu_callback(self, msg):
        """Update IMU data."""
        self.imu_orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        self.imu_angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def _update_body_pose_from_tf(self):
        """Get body pose from TF tree."""
        try:
            # Get transform from world to body
            transform = self.tf_buffer.lookup_transform(
                'world',
                'body',  # body link name
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            self.body_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            self.body_orientation = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])

        except (TransformException, Exception) as e:
            # If TF lookup fails, keep previous values
            # self.node.get_logger().warn(f"TF lookup failed: {e}")
            pass

    def _get_obs(self):
        """Get current observation."""
        # Update body pose from TF
        self._update_body_pose_from_tf()

        # Get current gait parameters
        gait_vec = self.gait_params.get(self.target_gait, [0.0, 0.0, 0.0])

        # Construct observation
        obs = np.concatenate([
            gait_vec,                # Gait command (3)
            [self.phase],            # Phase (1)
            self.joint_positions,    # Joint positions (12)
            self.joint_velocities,   # Joint velocities (12)
            self.body_position,      # Body position (3)
            self.body_orientation,   # Body orientation (4)
            self.imu_orientation,    # IMU orientation (4)
            self.imu_angular_vel,    # IMU angular velocity (3)
        ])

        return obs.astype(np.float32)

    def _calculate_reward(self, action):
        """
        Calculate reward based on robot state.

        Rewards:
        - Stability: upright, not falling
        - Task: moving in commanded direction
        - Efficiency: smooth movements
        - Head/Neck: Proper head position for balance
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

        # 2. Head not touching ground (use body Z as proxy)
        # Head is approximately body_z + 0.05m (5cm above body)
        head_z = self.body_position[2] + 0.05
        if head_z < 0.02:  # Head touching ground
            reward -= 5.0  # Big penalty
            info['head_contact'] = True
        else:
            reward += 0.5

        # 3. Body orientation from IMU (pitch and roll should be small)
        # Convert quaternion to euler angles
        qx, qy, qz, qw = self.imu_orientation
        # Use scipy for robust conversion
        try:
            r = Rotation.from_quat([qx, qy, qz, qw])
            euler = r.as_euler('xyz')
            roll, pitch, yaw = euler
        except:
            # Fallback simple conversion
            roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
            pitch = math.asin(np.clip(2*(qw*qy - qz*qx), -1, 1))

        if abs(roll) < 0.3 and abs(pitch) < 0.3:  # Within ~17 degrees
            reward += 1.0
        else:
            reward -= abs(roll) + abs(pitch)  # Penalty for tilting

        # 4. Head/neck position contributes to stability
        # Neck joints (motors 8-11) should be controlled for balance
        # Penalize extreme neck positions
        neck_joints = self.joint_positions[8:12]
        neck_penalty = np.sum(np.abs(neck_joints)) * 0.1
        reward -= neck_penalty

        # Check if fallen over
        if abs(roll) > 1.0 or abs(pitch) > 1.0 or self.body_position[2] < 0.05:
            reward -= 10.0
            done = True
            info['fallen'] = True

        # === TASK REWARDS ===

        # Estimate velocity from body position change
        # (In real implementation, you'd get this from TF or velocity topic)
        body_vel_x = 0.0  # Placeholder

        if self.target_gait in ['walk_forward', 'trot_forward']:
            # Reward forward movement (positive body_vel_x)
            reward += body_vel_x * 2.0

        elif self.target_gait in ['walk_backward', 'trot_backward']:
            # Reward backward movement (negative body_vel_x)
            reward -= body_vel_x * 2.0

        elif self.target_gait in ['walk_left', 'walk_right']:
            # Reward lateral movement
            # body_vel_y would be used here
            pass

        # === EFFICIENCY REWARDS ===

        # Penalize large joint velocity changes (smooth control)
        joint_accel = self.joint_positions - self.last_joint_positions
        energy_cost = np.sum(np.square(joint_accel)) * 0.01
        reward -= energy_cost

        # Penalize staying too long without progress
        if self.episode_step > self.max_episode_steps:
            done = True
            info['timeout'] = True

        return reward, done, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Joint positions [12]

        Returns:
            observation, reward, done, truncated, info
        """
        # Apply action (publish to controller)
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)

        # Spin node to process callbacks (no sleep - run as fast as possible)
        # For RL training, we want max speed. Gazebo will limit the rate.
        rclpy.spin_once(self.node, timeout_sec=0.001)

        # Update phase
        self.phase = (self.phase + 0.01) % 1.0
        self.episode_step += 1

        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward, done, info = self._calculate_reward(action)

        # Gymnasium API uses separate 'terminated' and 'truncated'
        terminated = done
        truncated = self.episode_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Reset episode tracking
        self.episode_step = 0
        self.phase = 0.0

        # Reset state
        self.last_joint_positions = np.zeros(12)

        # Randomize target gait (optional)
        # self.target_gait = np.random.choice(list(self.gait_params.keys()))

        # Spin a few times to get fresh state (no long sleep)
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.001)

        obs = self._get_obs()
        info = {}

        return obs, info

    def close(self):
        """Clean up resources."""
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
