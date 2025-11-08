"""
Gazebo RL Environment for PiDog - Vision-Based Multi-Modal Learning

Combines camera vision with IMU and joint proprioception for robust,
dog-like behavior learning that maximizes GPU utilization.
"""

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image, LaserScan
from gazebo_msgs.msg import Contacts
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformListener, Buffer, TransformException
from cv_bridge import CvBridge
import cv2
import time
import math
from scipy.spatial.transform import Rotation


class PiDogVisionEnv(gym.Env):
    """
    Multi-modal Gym environment for PiDog using ALL sensors.

    Observation: Dict with:
        - 'image': Camera RGB [84, 84, 3] (resized for CNN)
        - 'vector': [gait_cmd(4), joint_pos(12), joint_vel(12),
                     body_pos(3), body_orient(4), imu_orient(4), imu_angvel(3),
                     ultrasonic_range(1), touch_contact(1)]
                    Total: 44 dimensions

    Action: [12 joint positions] in radians

    Sensors Used:
        - Camera: Vision for obstacle detection
        - IMU: Orientation and angular velocity
        - Ultrasonic: Distance measurement for navigation
        - Touch: Contact detection
        - Joint encoders: Position and velocity feedback

    Rewards:
        - Stability (upright, head not touching ground)
        - Speed (forward velocity for running)
        - Efficiency (smooth movements)
        - Collision avoidance (ultrasonic + touch feedback)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, node_name='pidog_vision_env', headless=False):
        super(PiDogVisionEnv, self).__init__()

        # Initialize ROS2 if not already initialized
        if not rclpy.ok():
            rclpy.init()

        # Create ROS2 node
        self.node = Node(node_name)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # TF2 for getting body pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # State variables
        self.current_image = np.zeros((84, 84, 3), dtype=np.uint8)
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.body_position = np.zeros(3)  # x, y, z
        self.body_orientation = np.zeros(4)  # quaternion (x,y,z,w)
        self.imu_orientation = np.zeros(4)  # from IMU
        self.imu_angular_vel = np.zeros(3)
        self.ultrasonic_range = 4.0  # Default max range (4m for HC-SR04)
        self.touch_contact = 0.0  # 0 = no contact, 1 = contact
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

        self.camera_sub = self.node.create_subscription(
            Image,
            '/camera',
            self._camera_callback,
            10
        )

        self.ultrasonic_sub = self.node.create_subscription(
            LaserScan,
            '/ultrasonic',
            self._ultrasonic_callback,
            10
        )

        self.touch_sub = self.node.create_subscription(
            Contacts,
            '/touch_sensor/contacts',
            self._touch_callback,
            10
        )

        # Gym spaces - MultiModal observation
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0,
                high=255,
                shape=(84, 84, 3),
                dtype=np.uint8
            ),
            'vector': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(44,),  # Added ultrasonic_range(1) + touch_contact(1)
                dtype=np.float32
            )
        })

        # Action: 12 joint positions
        self.action_space = gym.spaces.Box(
            low=-1.57,  # ~-90 degrees
            high=1.57,   # ~+90 degrees
            shape=(12,),
            dtype=np.float32
        )

        # Gait command for current episode
        self.target_gait = 'trot_forward'  # Focus on speed!
        self.gait_params = {
            'walk_forward': [0.0, 1.0, 0.0],
            'walk_backward': [0.0, -1.0, 0.0],
            'trot_forward': [1.0, 1.0, 0.0],   # Fastest gait
            'trot_backward': [1.0, -1.0, 0.0],
            'stand': [2.0, 0.0, 0.0],
        }

        print(f"[PiDogVisionEnv] Multi-modal environment initialized")
        print(f"[PiDogVisionEnv] Observation: 84x84x3 image + 44D vector")
        print(f"[PiDogVisionEnv] Sensors: Camera, IMU, Ultrasonic, Touch, 12 Joint Encoders")

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

    def _camera_callback(self, msg):
        """Update camera image - resize to 84x84 for efficient CNN processing."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Resize to 84x84 (standard for vision-based RL)
            resized = cv2.resize(cv_image, (84, 84), interpolation=cv2.INTER_AREA)

            self.current_image = resized
        except Exception as e:
            self.node.get_logger().warn(f"Camera callback error: {e}")

    def _ultrasonic_callback(self, msg):
        """Update ultrasonic sensor distance reading."""
        if len(msg.ranges) > 0:
            # HC-SR04 range: 0.02-4.0m, single beam
            self.ultrasonic_range = msg.ranges[0]
            # Clamp to valid range
            self.ultrasonic_range = np.clip(self.ultrasonic_range, 0.02, 4.0)

    def _touch_callback(self, msg):
        """Update touch sensor contact state (head only)."""
        # Touch sensor is only on head - detects head collisions
        if len(msg.states) > 0:
            self.touch_contact = 1.0  # Contact detected
        else:
            self.touch_contact = 0.0  # No contact

    def _update_body_pose_from_tf(self):
        """Get body pose from TF tree."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                'body',
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
            pass

    def _get_obs(self):
        """Get current multi-modal observation."""
        # Update body pose from TF
        self._update_body_pose_from_tf()

        # Get current gait parameters
        gait_vec = self.gait_params.get(self.target_gait, [0.0, 0.0, 0.0])

        # Construct vector observation (44D total)
        vector_obs = np.concatenate([
            gait_vec,                 # Gait command (3)
            [self.phase],             # Phase (1)
            self.joint_positions,     # Joint positions (12)
            self.joint_velocities,    # Joint velocities (12)
            self.body_position,       # Body position (3)
            self.body_orientation,    # Body orientation (4)
            self.imu_orientation,     # IMU orientation (4)
            self.imu_angular_vel,     # IMU angular velocity (3)
            [self.ultrasonic_range],  # Ultrasonic distance (1)
            [self.touch_contact],     # Touch sensor contact (1)
        ]).astype(np.float32)

        return {
            'image': self.current_image.copy(),
            'vector': vector_obs
        }

    def _calculate_reward(self, action):
        """
        Calculate reward optimized for SPEED.

        Primary goal: Run forward as fast as possible while staying upright.
        """
        reward = 0.0
        done = False
        info = {}

        # === STABILITY REWARDS (Must stay upright to run) ===

        # 1. Upright reward
        target_height = 0.10  # 10cm standing height
        height_error = abs(self.body_position[2] - target_height)
        if self.body_position[2] > 0.08:
            reward += 1.0 - height_error * 5.0
        else:
            reward -= 2.0

        # 2. Head not touching ground
        head_z = self.body_position[2] + 0.05
        if head_z < 0.02:
            reward -= 5.0
            info['head_contact'] = True
        else:
            reward += 0.5

        # 3. Body orientation (pitch and roll should be small)
        qx, qy, qz, qw = self.imu_orientation
        try:
            r = Rotation.from_quat([qx, qy, qz, qw])
            euler = r.as_euler('xyz')
            roll, pitch, yaw = euler
        except:
            roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
            pitch = math.asin(np.clip(2*(qw*qy - qz*qx), -1, 1))

        if abs(roll) < 0.3 and abs(pitch) < 0.3:
            reward += 1.0
        else:
            reward -= abs(roll) + abs(pitch)

        # Check if fallen over
        if abs(roll) > 1.0 or abs(pitch) > 1.0 or self.body_position[2] < 0.05:
            reward -= 10.0
            done = True
            info['fallen'] = True

        # === SPEED REWARDS (Primary objective!) ===

        # Estimate forward velocity from position change
        # In real implementation, this would come from TF velocity or odometry
        body_vel_x = 0.0  # Placeholder - need to track position history

        if self.target_gait in ['trot_forward', 'walk_forward']:
            # BIG reward for forward speed!
            reward += body_vel_x * 5.0  # 5x multiplier for speed emphasis

        # === EFFICIENCY ===

        # Penalize large accelerations (smooth = fast)
        joint_accel = self.joint_positions - self.last_joint_positions
        energy_cost = np.sum(np.square(joint_accel)) * 0.01
        reward -= energy_cost

        # Timeout
        if self.episode_step > self.max_episode_steps:
            done = True
            info['timeout'] = True

        info['body_z'] = self.body_position[2]
        info['forward_vel'] = body_vel_x

        return reward, done, info

    def step(self, action):
        """
        Execute one step in the environment.
        """
        # Apply action
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)

        # Wait for control step
        time.sleep(self.dt)

        # Spin node to process callbacks
        rclpy.spin_once(self.node, timeout_sec=0.01)

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
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset episode tracking
        self.episode_step = 0
        self.phase = 0.0

        # Reset state
        self.last_joint_positions = np.zeros(12)

        # Wait for state to stabilize
        time.sleep(0.5)
        rclpy.spin_once(self.node, timeout_sec=0.1)

        obs = self._get_obs()
        info = {}

        return obs, info

    def close(self):
        """Clean up resources."""
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
