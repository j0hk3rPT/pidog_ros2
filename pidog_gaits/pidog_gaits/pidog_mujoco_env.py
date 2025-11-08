"""
PiDog MuJoCo Gymnasium Environment for RL Training.

This environment uses MuJoCo physics simulation with native IMU sensors
for training quadruped locomotion policies.
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer  # Must import viewer explicitly!
import numpy as np
from typing import Optional, Tuple, Dict, Any


class PiDogMuJoCoEnv(gym.Env):
    """PiDog quadruped environment using MuJoCo physics.

    Observation Space (28D):
        - Joint positions (8): shoulder and knee angles for 4 legs
        - Joint velocities (8): angular velocities
        - IMU orientation (4): quaternion [x, y, z, w]
        - IMU gyroscope (3): angular velocity in body frame
        - IMU accelerometer (3): linear acceleration in body frame
        - Body height (1): Z position of body
        - Body velocity (1): forward velocity

    Action Space (8D):
        - Target joint positions for 8 leg joints (normalized -1 to 1)
        - Maps to actual joint range (-1.57 to 1.57 radians)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        model_path: str = "pidog_description/mjcf/pidog_simple.xml",  # Use pidog_with_meshes.xml for realistic appearance
        frame_skip: int = 20,  # 50 Hz control from 1000 Hz physics
        max_episode_steps: int = 1000,
    ):
        """Initialize PiDog MuJoCo environment.

        Args:
            render_mode: Rendering mode ("human", "rgb_array", or None)
            model_path: Path to MuJoCo XML model
            frame_skip: Number of physics steps per control step
            max_episode_steps: Maximum episode length
        """
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Get sensor IDs
        self._setup_sensor_ids()

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Rendering
        self.render_mode = render_mode
        self.viewer = None

        # Target standing pose (used for reward computation)
        self.target_height = 0.10  # Target body height (10cm)

    def _setup_sensor_ids(self):
        """Get sensor IDs and addresses for fast lookup."""
        # IMU sensors
        self.imu_orient_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_orientation'
        )
        self.imu_gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro'
        )
        self.imu_accel_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_accel'
        )

        # Sensor data addresses
        self.orient_addr = self.model.sensor_adr[self.imu_orient_id]
        self.gyro_addr = self.model.sensor_adr[self.imu_gyro_id]
        self.accel_addr = self.model.sensor_adr[self.imu_accel_id]

    def _get_obs(self) -> np.ndarray:
        """Get current observation from MuJoCo sensors.

        Returns:
            28D observation vector
        """
        # Joint positions and velocities (skip free joint: first 7 qpos, 6 qvel)
        qpos = self.data.qpos[7:15]  # 8 leg joints
        qvel = self.data.qvel[6:14]  # 8 leg joints

        # IMU data from sensors
        orientation = self.data.sensordata[self.orient_addr:self.orient_addr+4]
        gyro = self.data.sensordata[self.gyro_addr:self.gyro_addr+3]
        accel = self.data.sensordata[self.accel_addr:self.accel_addr+3]

        # Body state from free joint
        body_height = self.data.qpos[2]  # Z position
        body_vel_x = self.data.qvel[0]   # Forward velocity

        # Concatenate observation
        obs = np.concatenate([
            qpos,           # 8 joint positions
            qvel,           # 8 joint velocities
            orientation,    # 4 IMU quaternion
            gyro,           # 3 IMU gyroscope
            accel,          # 3 IMU accelerometer
            [body_height],  # 1 body height
            [body_vel_x],   # 1 forward velocity
        ])

        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for logging."""
        body_height = self.data.qpos[2]
        orientation_w = self.data.sensordata[self.orient_addr + 3]

        return {
            "body_height": body_height,
            "upright": orientation_w,  # Should be ~1.0 when upright
            "episode_step": self.current_step,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set initial standing pose with small random noise
        standing_pose = np.array([-1.2, 0.18, -1.2, 0.18, 1.2, -0.18, 1.2, -0.18])
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.uniform(-0.1, 0.1, size=8)
        self.data.qpos[7:15] = standing_pose + noise

        # Set body slightly above ground
        self.data.qpos[2] = 0.12  # 12cm initial height

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Let robot settle for a few steps
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.

        Args:
            action: 8D array of target joint positions (normalized -1 to 1)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action from [-1, 1] to joint range [-1.57, 1.57]
        action_scaled = action * 1.57

        # Set control targets
        self.data.ctrl[:8] = action_scaled

        # Step physics multiple times (frame_skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps

        # Get info
        info = self._get_info()

        self.current_step += 1

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        """Compute reward for current state.

        Reward components:
            - Height reward: Encourage maintaining target height
            - Upright reward: Encourage staying upright
            - Velocity reward: Encourage forward motion (optional)
            - Energy penalty: Penalize large control efforts
        """
        # Body height (target: 0.10m)
        body_height = self.data.qpos[2]
        height_reward = 1.0 if body_height > 0.08 else 0.0

        # Upright orientation (quaternion w should be ~1.0)
        orientation_w = self.data.sensordata[self.orient_addr + 3]
        upright_reward = max(0, orientation_w)  # 0 to 1

        # Forward velocity (encourage walking forward)
        vel_x = self.data.qvel[0]
        velocity_reward = min(vel_x, 0.5)  # Cap at 0.5 m/s

        # Energy penalty (encourage smooth movements)
        control_cost = 0.001 * np.sum(np.square(self.data.ctrl[:8]))

        # Total reward
        reward = (
            2.0 * height_reward +
            1.0 * upright_reward +
            1.0 * velocity_reward -
            control_cost
        )

        return reward

    def _is_terminated(self) -> bool:
        """Check if episode should terminate (robot fell).

        Returns:
            True if robot fell over
        """
        body_height = self.data.qpos[2]
        fallen = body_height < 0.05  # Below 5cm = fallen

        # Check if robot flipped over (w component negative)
        orientation_w = self.data.sensordata[self.orient_addr + 3]
        flipped = orientation_w < 0.0

        return fallen or flipped

    def render(self):
        """Render environment (if render_mode is set)."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            if self.viewer is not None:
                self.viewer.sync()

        elif self.render_mode == "rgb_array":
            # TODO: Implement offscreen rendering for video recording
            pass

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Register environment with Gymnasium
try:
    from gymnasium.envs.registration import register

    register(
        id='PiDogMuJoCo-v0',
        entry_point='pidog_gaits.pidog_mujoco_env:PiDogMuJoCoEnv',
        max_episode_steps=1000,
    )
except Exception:
    pass  # Already registered or not in package context
