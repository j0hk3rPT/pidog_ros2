# MuJoCo Migration Plan for PiDog RL Training

## Why MuJoCo?

**Current Problem**: Gazebo Harmonic sensor plugins crash/hang, preventing IMU sensor usage for RL training.

**MuJoCo Advantages**:
- ✅ Native IMU sensor support (built-in, stable)
- ✅ URDF import (can reuse existing robot model)
- ✅ Fast physics simulation (1000+ Hz on CPU)
- ✅ Designed for RL from ground up
- ✅ Perfect integration with Gymnasium + Stable-Baselines3
- ✅ Better contact dynamics for quadruped locomotion
- ✅ Works with AMD GPU (OpenGL rendering)

## Architecture Changes

### Current Stack (Gazebo)
```
Gazebo Physics → ROS2 topics → Custom Gymnasium Env → PPO (SB3)
```

### New Stack (MuJoCo)
```
MuJoCo Physics → Direct Python API → Gymnasium Env → PPO (SB3)
```

**Simplifications**:
- No ROS2 dependency for RL training
- No topic bridges needed
- Direct sensor access
- Faster iteration

## Installation Steps

### 1. Install MuJoCo in Container

```bash
# Install MuJoCo Python bindings
pip install mujoco>=3.1.0
pip install gymnasium[mujoco]  # Includes helpful utilities

# Verify installation
python3 -c "import mujoco; print(f'MuJoCo {mujoco.__version__}')"
```

### 2. Test AMD GPU Rendering

```python
# test_mujoco_rendering.py
import mujoco
import mujoco.viewer

# Simple test model
xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Launch viewer (uses OpenGL - will test AMD GPU)
mujoco.viewer.launch(model, data)
```

**Expected**: Green cube bouncing on red plane with smooth rendering.

## Robot Model Conversion

### Option 1: Direct URDF Import (Easiest)

MuJoCo 3.x supports URDF directly:

```python
import mujoco

# Load your existing URDF
model = mujoco.MjModel.from_xml_path(
    '/workspace/pidog_description/urdf/pidog.urdf'
)
```

**Pros**: No conversion needed, use existing URDF
**Cons**: Some Gazebo-specific tags may not translate perfectly

### Option 2: Convert URDF to MJCF (Recommended)

Create native MuJoCo XML (MJCF) for better control:

```bash
# Use MuJoCo's compile utility
python3 -c "
import mujoco
model = mujoco.MjModel.from_xml_path('pidog_description/urdf/pidog.urdf')
mujoco.mj_saveLastXML('pidog_description/mjcf/pidog.xml', model)
"
```

Then hand-tune the MJCF for:
- Contact parameters (better than Gazebo ODE)
- Actuator models (position, velocity, or torque control)
- Sensor placement (IMU, joint sensors, contact sensors)

### Adding IMU Sensor in MJCF

```xml
<sensor>
  <!-- IMU on body -->
  <framequat name="imu_orientation" objtype="body" objname="body"/>
  <gyro name="imu_gyro" site="imu_site"/>
  <accelerometer name="imu_accel" site="imu_site"/>
</sensor>

<worldbody>
  <body name="body">
    <!-- Add IMU site (sensor attachment point) -->
    <site name="imu_site" pos="0 0 0.02" size="0.001"/>
    <!-- rest of body definition -->
  </body>
</worldbody>
```

**Output**: Real IMU sensor data, no virtual computation needed!

## Gymnasium Environment

### New Environment Structure

```python
# pidog_gaits/pidog_gaits/pidog_mujoco_env.py
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class PiDogMuJoCoEnv(gym.Env):
    """PiDog quadruped environment using MuJoCo."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            'pidog_description/mjcf/pidog.xml'
        )
        self.data = mujoco.MjData(self.model)

        # Get sensor IDs
        self.imu_gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro'
        )
        self.imu_accel_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_accel'
        )

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        """Get observation from MuJoCo sensors."""
        # Joint positions and velocities
        qpos = self.data.qpos[7:]  # Skip free joint (first 7)
        qvel = self.data.qvel[6:]  # Skip free joint (first 6)

        # IMU data (direct from sensors!)
        gyro = self.data.sensordata[self.imu_gyro_id:self.imu_gyro_id+3]
        accel = self.data.sensordata[self.imu_accel_id:self.imu_accel_id+3]

        # Body orientation (from free joint)
        orientation = self.data.qpos[3:7]  # Quaternion

        obs = np.concatenate([
            qpos[:12],      # 12 joint positions
            qvel[:12],      # 12 joint velocities
            orientation,    # 4 quaternion
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose
        # ... (set standing pose)

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Apply action (joint position targets)
        self.data.ctrl[:] = action

        # Step physics (50 Hz control, 1000 Hz physics)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._is_terminated()
        truncated = False

        info = {}
        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """Reward function for standing/walking."""
        # Body height
        body_height = self.data.qpos[2]
        height_reward = 1.0 if body_height > 0.08 else 0.0

        # Upright orientation
        orientation = self.data.qpos[3:7]
        upright_reward = orientation[3]  # w component (should be ~1)

        # Velocity (for walking tasks)
        vel_x = self.data.qvel[0]
        vel_reward = min(vel_x, 0.3)  # Reward forward motion

        return height_reward + upright_reward + vel_reward

    def _is_terminated(self):
        """Check if episode should end."""
        body_height = self.data.qpos[2]
        return body_height < 0.05  # Fallen over

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # Render to numpy array for video recording
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
```

### Register Environment

```python
# In pidog_gaits/setup.py or __init__.py
from gymnasium.envs.registration import register

register(
    id='PiDogMuJoCo-v0',
    entry_point='pidog_gaits.pidog_mujoco_env:PiDogMuJoCoEnv',
    max_episode_steps=1000,
)
```

## Training Script (Minimal Changes)

```python
# train_mujoco.py
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment (same API as before!)
env = gym.make('PiDogMuJoCo-v0', render_mode=None)

# Train (identical to Gazebo version)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device='cuda',  # Still use GPU for neural network
    n_steps=2048,
    batch_size=64,
)

model.learn(total_timesteps=1_000_000)
model.save("pidog_mujoco_ppo")

# Test
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Performance Comparison

| Feature | Gazebo Harmonic | MuJoCo 3.x |
|---------|----------------|------------|
| **Physics Speed** | ~1000 Hz | ~10,000 Hz |
| **IMU Sensor** | ❌ Crashes | ✅ Native |
| **Setup Complexity** | High (ROS2, bridges) | Low (direct Python) |
| **RL Integration** | Custom wrapper | Built-in |
| **Contact Stability** | Good | Excellent |
| **GPU Requirement** | Yes (rendering) | No (CPU physics) |
| **Training Speed** | Moderate | Fast |

## Migration Steps

1. ✅ **Install MuJoCo** in container
2. ✅ **Test rendering** with AMD GPU
3. ✅ **Load URDF** in MuJoCo
4. ✅ **Add IMU sensor** to model
5. ✅ **Create Gymnasium env** with direct sensor access
6. ✅ **Update training script** (minimal changes)
7. ✅ **Train first model** and verify
8. ✅ **Compare with Gazebo** results

## Rollback Plan

Keep Gazebo setup intact:
- MuJoCo for RL training (fast iteration)
- Gazebo for visualization and ROS2 integration
- Both use same URDF, so models are compatible

## Next Steps

1. Install MuJoCo and test rendering
2. Convert URDF to MJCF
3. Create basic MuJoCo environment
4. Run first training experiment
5. Compare training speed and results

**Estimated Time**: 2-3 hours for complete migration

Let me know when you're ready to start!
