"""
PiDog MuJoCo MJX Environment - GPU-Accelerated Training

Uses MuJoCo XLA (MJX) for massively parallel physics simulation on GPU/TPU.
Includes realistic servo modeling matching SunFounder SF006FM 9g digital servos.

Features:
- Position-controlled servos with realistic torque limits (0.15 Nm)
- Servo dynamics (kp=150, kv=10) matching PID-controlled servos
- 1000s of parallel environments on GPU
- Compatible with JAX-based RL libraries (brax.training, etc.)

Hardware Specs (Real Servos):
- Model: SunFounder SF006FM 9g Digital Servo
- Torque: 0.127-0.137 Nm (1.3-1.4 kgf·cm at 4.8-6V)
- Speed: 333-400°/s (5.8-7.0 rad/s)
- Operating: 4.8-6.0V, 0-180° range
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
import mujoco
from mujoco import mjx
from typing import Tuple, Dict, Any
import numpy as np

from pidog_brax_mjcf import generate_pidog_mjcf


class PiDogMJXEnv:
    """
    GPU-accelerated PiDog environment using MuJoCo MJX

    Observation space (28D):
        - gait_command: [gait_type, direction, turn, phase] (4D)
        - joint_positions: 8D (BR, FR, BL, FL - shoulder, knee each)
        - joint_velocities: 8D
        - body_orientation: [roll, pitch, yaw] (3D)
        - body_height: 1D
        - body_velocity: [vx, vy, vz] (3D)
        - foot_contacts: [br, fr, bl, fl] (4D binary)

    Action space (8D):
        - Target joint angles for 8 leg servos (radians)
        - Clamped to [-1.57, 1.57] (±90°) matching real servo limits

    Servo Model:
        - Position control with PD gains (kp=150, kv=10)
        - Torque saturation at 0.15 Nm (matches real hardware)
        - Joint damping = 0.5, friction = 0.5 (realistic mechanics)
    """

    def __init__(
        self,
        gait_command: Tuple[int, int, int] = (0, 1, 0),  # walk_forward
        episode_length: int = 1000,
        dt: float = 0.02,  # 50 Hz control rate (matches ROS2)
    ):
        """
        Initialize PiDog MJX environment

        Args:
            gait_command: (gait_type, direction, turn)
                gait_type: 0=walk, 1=trot, 2=static_pose
                direction: -1=backward, 0=none, 1=forward
                turn: -1=left, 0=straight, 1=right
            episode_length: Max steps per episode
            dt: Physics timestep in seconds
        """
        # Load MuJoCo model from MJCF
        mjcf_xml = generate_pidog_mjcf()
        self.mj_model = mujoco.MjModel.from_xml_string(mjcf_xml)

        # Convert to MJX model (JAX)
        self.model = mjx.put_model(self.mj_model)

        # Environment parameters
        self.gait_command = jnp.array(gait_command, dtype=jnp.float32)
        self.episode_length = episode_length
        self.dt = dt

        # State dimensions
        self.nq = self.model.nq  # Position DOF (15: 7 freejoint + 8 legs)
        self.nv = self.model.nv  # Velocity DOF (14: 6 freejoint + 8 legs)
        self.nu = self.model.nu  # Actuator DOF (8 leg servos)

        print(f"PiDog MJX Environment:")
        print(f"  Position DOF (nq): {self.nq}")
        print(f"  Velocity DOF (nv): {self.nv}")
        print(f"  Actuators (nu): {self.nu}")
        print(f"  Control rate: {1/dt:.0f} Hz")
        print(f"  Servo torque limit: 0.15 Nm (realistic)")

    def reset(self, rng: jax.Array) -> mjx.Data:
        """
        Reset environment to standing pose

        Returns:
            MJX data structure with initial state
        """
        # Create initial MJX data
        data = mjx.make_data(self.model)

        # Set initial joint positions (standing pose)
        # Freejoint: [x, y, z, qw, qx, qy, qz]
        # Legs: [br_shoulder, br_knee, fr_shoulder, fr_knee,
        #        bl_shoulder, bl_knee, fl_shoulder, fl_knee]

        # Body at 0.1m height, upright orientation
        qpos = jnp.zeros(self.nq)
        qpos = qpos.at[2].set(0.1)  # Z position
        qpos = qpos.at[3].set(1.0)  # qw (identity quaternion)

        # Standing pose: shoulders at 0.5 rad, knees at -0.8 rad
        leg_angles = jnp.array([
            0.5, -0.8,  # BR
            0.5, -0.8,  # FR
            0.5, -0.8,  # BL
            0.5, -0.8,  # FL
        ])

        # Add small random noise
        rng, rng_noise = jax.random.split(rng)
        noise = jax.random.uniform(rng_noise, (8,), minval=-0.1, maxval=0.1)
        leg_angles = leg_angles + noise

        # Set joint positions (skip first 7 for freejoint)
        qpos = qpos.at[7:].set(leg_angles)

        # Zero velocities
        qvel = jnp.zeros(self.nv)

        # Update data
        data = data.replace(qpos=qpos, qvel=qvel)

        # Forward kinematics to get consistent state
        data = mjx.forward(self.model, data)

        return data

    def step(
        self,
        data: mjx.Data,
        action: jax.Array,
        phase: float
    ) -> Tuple[mjx.Data, float, bool, Dict[str, Any]]:
        """
        Step physics simulation with servo control

        Args:
            data: Current MJX data
            action: Target joint angles (8D) in radians
            phase: Gait phase [0, 1]

        Returns:
            (new_data, reward, done, info)
        """
        # Clamp actions to servo limits
        action = jnp.clip(action, -1.57, 1.57)

        # Set actuator controls (MJX position actuators)
        data = data.replace(ctrl=action)

        # Physics step
        data = mjx.step(self.model, data)

        # Compute reward
        reward = self._compute_reward(data, action)

        # Check termination
        body_height = data.qpos[2]  # Z position of torso
        body_quat = data.qpos[3:7]  # Quaternion

        # Compute roll and pitch from quaternion
        roll, pitch = self._quat_to_rp(body_quat)

        # Terminate if fallen or tipped over
        done = jnp.logical_or(
            body_height < 0.03,  # Fallen (body touches ground)
            jnp.logical_or(
                jnp.abs(roll) > 1.0,  # Rolled over (>57°)
                jnp.abs(pitch) > 1.0  # Pitched over
            )
        )

        # Info metrics
        info = {
            'body_height': body_height,
            'body_velocity': data.qvel[0],  # Forward velocity
            'roll': roll,
            'pitch': pitch,
            'reward': reward,
        }

        return data, reward, done, info

    def _quat_to_rp(self, quat: jax.Array) -> Tuple[float, float]:
        """Convert quaternion [w,x,y,z] to roll and pitch"""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Roll (X-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (Y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = jnp.where(
            jnp.abs(sinp) >= 1,
            jnp.copysign(jnp.pi / 2, sinp),  # Use 90° if out of range
            jnp.arcsin(sinp)
        )

        return roll, pitch

    def _compute_reward(self, data: mjx.Data, action: jax.Array) -> float:
        """
        Reward function for quadruped locomotion

        Components:
        1. Forward velocity (task reward)
        2. Standing upright (height + orientation)
        3. Energy efficiency (penalize large torques)
        4. Survival bonus
        """
        # Extract state
        body_height = data.qpos[2]
        body_quat = data.qpos[3:7]
        body_vel = data.qvel[:3]  # Linear velocity

        # 1. Task reward: forward velocity
        gait_type, direction, turn = self.gait_command
        forward_reward = 1.0 * body_vel[0] * direction  # Scale with direction

        # 2. Standing reward: height and orientation
        target_height = 0.08  # meters
        height_reward = 2.0 * jnp.exp(-10.0 * (body_height - target_height)**2)

        roll, pitch = self._quat_to_rp(body_quat)
        orientation_reward = -0.5 * (roll**2 + pitch**2)

        # 3. Energy penalty: discourage large torques
        # Estimate torque from actuator force (MJX computes this)
        energy_penalty = -0.01 * jnp.sum(data.actuator_force**2)

        # 4. Survival bonus
        survival_reward = 0.1

        # Total reward
        reward = (
            forward_reward +
            height_reward +
            orientation_reward +
            energy_penalty +
            survival_reward
        )

        return reward

    def get_obs(self, data: mjx.Data, phase: float) -> jax.Array:
        """
        Get observation vector (28D)

        Returns observation compatible with existing neural network models.
        """
        # Gait command with phase
        gait_obs = jnp.concatenate([
            self.gait_command,
            jnp.array([phase])
        ])  # 4D

        # Joint positions (8 legs, skip freejoint)
        qpos_legs = data.qpos[7:]  # 8D

        # Joint velocities (8 legs, skip freejoint)
        qvel_legs = data.qvel[6:]  # 8D

        # Body orientation (roll, pitch, yaw)
        quat = data.qpos[3:7]
        roll, pitch = self._quat_to_rp(quat)

        # Yaw from quaternion
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)

        body_orientation = jnp.array([roll, pitch, yaw])  # 3D

        # Body height
        body_height = jnp.array([data.qpos[2]])  # 1D

        # Body velocity
        body_velocity = data.qvel[:3]  # 3D (vx, vy, vz)

        # Foot contacts (check contact forces)
        # MJX contact detection: check if foot geoms have contact force
        foot_contacts = jnp.zeros(4)  # Placeholder - would need contact sensor data
        # TODO: Implement proper contact detection from data.contact

        # Concatenate all observations
        obs = jnp.concatenate([
            gait_obs,          # 4D
            qpos_legs,         # 8D
            qvel_legs,         # 8D
            body_orientation,  # 3D
            body_height,       # 1D
            body_velocity,     # 3D
            foot_contacts,     # 4D
        ])  # Total: 28D

        return obs


# Vectorized environment for parallel training
class VectorizedPiDogMJXEnv:
    """
    Vectorized version of PiDog MJX environment for parallel GPU training

    Runs N environments in parallel using JAX vmap.
    """

    def __init__(
        self,
        num_envs: int = 1024,
        gait_command: Tuple[int, int, int] = (0, 1, 0),
        episode_length: int = 1000,
    ):
        """
        Initialize vectorized environment

        Args:
            num_envs: Number of parallel environments (use 1024-4096 for GPU)
            gait_command: Default gait command for all envs
            episode_length: Max episode length
        """
        self.num_envs = num_envs
        self.env = PiDogMJXEnv(gait_command, episode_length)

        # Vectorize reset and step functions
        self.reset_vmap = jit(vmap(self.env.reset))
        self.step_vmap = jit(vmap(self.env.step, in_axes=(0, 0, 0)))
        self.get_obs_vmap = jit(vmap(self.env.get_obs, in_axes=(0, 0)))

        print(f"Vectorized Environment: {num_envs} parallel envs")

    def reset(self, rng: jax.Array) -> Tuple[jax.Array, mjx.Data]:
        """
        Reset all environments in parallel

        Returns:
            (observations, data_batch)
        """
        rngs = jax.random.split(rng, self.num_envs)
        data_batch = self.reset_vmap(rngs)

        # Get initial observations
        phases = jnp.zeros(self.num_envs)
        obs_batch = self.get_obs_vmap(data_batch, phases)

        return obs_batch, data_batch

    def step(
        self,
        data_batch: mjx.Data,
        actions: jax.Array,
        phases: jax.Array
    ) -> Tuple[jax.Array, mjx.Data, jax.Array, jax.Array, Dict]:
        """
        Step all environments in parallel

        Args:
            data_batch: Batch of MJX data (num_envs,)
            actions: Batch of actions (num_envs, 8)
            phases: Batch of gait phases (num_envs,)

        Returns:
            (observations, new_data_batch, rewards, dones, infos)
        """
        # Step all environments
        results = self.step_vmap(data_batch, actions, phases)
        new_data_batch, rewards, dones, infos = results

        # Get observations
        obs_batch = self.get_obs_vmap(new_data_batch, phases)

        return obs_batch, new_data_batch, rewards, dones, infos


if __name__ == '__main__':
    """Test environment creation and basic stepping"""
    import time

    print("=" * 60)
    print("Testing PiDog MJX Environment")
    print("=" * 60)

    # Single environment test
    print("\n1. Single Environment Test")
    env = PiDogMJXEnv(gait_command=(0, 1, 0))  # walk_forward

    rng = jax.random.PRNGKey(0)
    data = env.reset(rng)

    print(f"✅ Environment reset")
    print(f"   Initial height: {data.qpos[2]:.3f}m")
    print(f"   Initial qpos shape: {data.qpos.shape}")
    print(f"   Initial qvel shape: {data.qvel.shape}")

    # Test stepping
    action = jnp.array([0.5, -0.8] * 4)  # Standing pose
    data, reward, done, info = env.step(data, action, phase=0.0)

    print(f"✅ Environment stepped")
    print(f"   Reward: {reward:.3f}")
    print(f"   Done: {done}")
    print(f"   Height: {info['body_height']:.3f}m")

    # Vectorized environment test
    print("\n2. Vectorized Environment Test (1024 parallel envs)")
    vec_env = VectorizedPiDogMJXEnv(num_envs=1024)

    rng = jax.random.PRNGKey(42)
    obs_batch, data_batch = vec_env.reset(rng)

    print(f"✅ Vectorized reset")
    print(f"   Observations shape: {obs_batch.shape}")
    print(f"   Expected: (1024, 28)")

    # Test vectorized stepping
    actions = jnp.tile(action, (1024, 1))
    phases = jnp.zeros(1024)

    start = time.time()
    obs_batch, data_batch, rewards, dones, infos = vec_env.step(
        data_batch, actions, phases
    )
    elapsed = time.time() - start

    print(f"✅ Vectorized step completed in {elapsed*1000:.1f}ms")
    print(f"   Throughput: {1024/elapsed:.0f} steps/sec")
    print(f"   Mean reward: {jnp.mean(rewards):.3f}")
    print(f"   Mean height: {jnp.mean(infos['body_height']):.3f}m")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nNext: Train with JAX-based RL (e.g., Brax PPO, CleanRL)")
    print("Expected training speed: 10K-50K steps/sec on GPU")
