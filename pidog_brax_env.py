"""
PiDog Brax Gymnasium Environment for GPU-accelerated RL training

This environment wraps the Brax physics simulator for massively parallel training
on AMD/NVIDIA GPUs using JAX.

Usage:
    env = PiDogBraxEnv()
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf
from brax import base
import numpy as np
from typing import Any, Dict, Tuple

from pidog_brax_mjcf import generate_pidog_mjcf


class PiDogBraxEnv(PipelineEnv):
    """
    PiDog quadruped environment for Brax

    Observation space (28D - matches GaitNetLSTM input):
        - gait_command: [gait_type, direction, turn, phase] (4D)
        - joint_positions: 8D (shoulder, knee for each leg)
        - joint_velocities: 8D
        - body_orientation: [roll, pitch, yaw] (3D)
        - body_height: 1D
        - body_velocity: [vx, vy, vz] (3D)
        - foot_contacts: [br, fr, bl, fl] (4D binary)

    Action space (8D):
        - Target joint angles for 8 actuators (radians)
        - [br_shoulder, br_knee, fr_shoulder, fr_knee,
           bl_shoulder, bl_knee, fl_shoulder, fl_knee]

    Reward function:
        - Forward velocity bonus (for walk_forward)
        - Standing upright (body height > 0.08m)
        - Stable orientation (low roll/pitch)
        - Energy penalty (discourages large movements)
        - Head contact penalty (head shouldn't touch ground)
    """

    def __init__(
        self,
        gait_command: Tuple[int, int, int] = (0, 1, 0),  # walk_forward default
        episode_length: int = 1000,
        action_repeat: int = 1,
        **kwargs
    ):
        """
        Initialize PiDog Brax environment

        Args:
            gait_command: (gait_type, direction, turn)
                gait_type: 0=walk, 1=trot, 2=static_pose
                direction: -1=backward, 0=none, 1=forward
                turn: -1=left, 0=straight, 1=right
            episode_length: Max steps per episode
            action_repeat: Number of physics steps per action
        """
        # Generate MJCF XML and load via MuJoCo model object
        # Note: Brax's mjcf.loads() has XML parsing issues
        # Workaround: Load with MuJoCo first, then convert to Brax system
        mjcf_xml = generate_pidog_mjcf()
        import mujoco
        mj_model = mujoco.MjModel.from_xml_string(mjcf_xml)
        sys = mjcf.load_model(mj_model)

        self._gait_command = jnp.array(gait_command, dtype=jnp.float32)
        self._phase = 0.0
        self._episode_length = episode_length

        # Physics backend (generalized, spring, or positional)
        backend = 'generalized'  # Best for complex multi-body systems

        super().__init__(sys, backend=backend, n_frames=action_repeat)

    def reset(self, rng: jax.Array) -> State:
        """
        Reset environment to initial state

        Standing pose: body height ~0.08m, legs extended downward
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Initial state for freejoint + 8 leg joints
        # nq = 15: [x, y, z, qw, qx, qy, qz, joint1, ..., joint8]
        # nv = 14: [vx, vy, vz, wx, wy, wz, joint_vel1, ..., joint_vel8]

        # Freejoint position: [x, y, z] + quaternion [w, x, y, z]
        # Start with body at 0.1m height, identity rotation
        freejoint_pos = jnp.array([0.0, 0.0, 0.1])  # XYZ position
        freejoint_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (w, x, y, z)

        # Leg joint positions (standing pose)
        # Shoulder angles: ~0.5 rad (legs angled outward)
        # Knee angles: ~-0.8 rad (knees bent to support weight)
        leg_joints = jnp.array([
            0.5, -0.8,  # BR
            0.5, -0.8,  # FR
            0.5, -0.8,  # BL
            0.5, -0.8,  # FL
        ])

        # Add small random noise to leg joints
        leg_joints = leg_joints + jax.random.uniform(rng1, (8,), minval=-0.1, maxval=0.1)

        # Combine into full q_init (15 DOF)
        q_init = jnp.concatenate([freejoint_pos, freejoint_quat, leg_joints])

        # Initial velocities (14 DOF: 6 for freejoint + 8 for legs)
        qd_init = jnp.zeros(14)

        pipeline_state = self.pipeline_init(q_init, qd_init)

        obs = self._get_obs(pipeline_state, jnp.zeros(8))
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'forward_velocity': zero,
            'body_height': zero,
            'orientation_penalty': zero,
            'energy_penalty': zero,
        }

        self._phase = 0.0

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """
        Step environment forward one timestep

        Args:
            state: Current state
            action: Joint angle targets (8D)

        Returns:
            New state with updated obs, reward, done
        """
        # Clip actions to valid joint limits
        action = jnp.clip(action, -1.57, 1.57)

        # Step physics simulation
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Get observation
        obs = self._get_obs(pipeline_state, action)

        # Compute reward
        reward = self._compute_reward(pipeline_state, action)

        # Episode termination conditions
        body_height = pipeline_state.x.pos[0, 2]  # Body Z position
        body_roll = pipeline_state.x.rot[0, 0]
        body_pitch = pipeline_state.x.rot[0, 1]

        done = jnp.where(body_height < 0.03, 1.0, 0.0)  # Fallen over
        done = jnp.where(jnp.abs(body_roll) > 1.0, 1.0, done)  # Tipped over
        done = jnp.where(jnp.abs(body_pitch) > 1.0, 1.0, done)

        # Update phase (for gait cycle tracking)
        self._phase = (self._phase + 1.0 / self._episode_length) % 1.0

        # Metrics for logging
        metrics = {
            'forward_velocity': pipeline_state.x.vel[0, 0],
            'body_height': body_height,
            'orientation_penalty': jnp.abs(body_roll) + jnp.abs(body_pitch),
            'energy_penalty': jnp.sum(jnp.square(action)),
        }

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics
        )

    def _get_obs(self, pipeline_state: Any, action: jax.Array) -> jax.Array:
        """
        Construct observation vector (28D)

        Matches GaitNetLSTM input format for transfer learning
        """
        # Gait command with current phase
        gait_obs = jnp.concatenate([
            self._gait_command,
            jnp.array([self._phase])
        ])  # 4D

        # Joint positions and velocities
        qpos = pipeline_state.q  # 8D
        qvel = pipeline_state.qd  # 8D

        # Body state
        body_pos = pipeline_state.x.pos[0]  # XYZ position
        body_rot = pipeline_state.x.rot[0]  # Roll, pitch, yaw
        body_vel = pipeline_state.x.vel[0]  # XYZ velocity

        # Foot contact detection (check if foot Z < threshold)
        foot_indices = [3, 6, 9, 12]  # Indices of foot bodies
        foot_contacts = jnp.array([
            pipeline_state.x.pos[i, 2] < 0.01  # Contact if Z < 1cm
            for i in foot_indices
        ], dtype=jnp.float32)  # 4D

        # Concatenate all observations
        obs = jnp.concatenate([
            gait_obs,          # 4D
            qpos,              # 8D
            qvel,              # 8D
            body_rot,          # 3D (roll, pitch, yaw)
            body_pos[2:3],     # 1D (height only)
            body_vel,          # 3D
            foot_contacts,     # 4D
        ])  # Total: 28D

        return obs

    def _compute_reward(self, pipeline_state: Any, action: jax.Array) -> float:
        """
        Compute reward based on task and robot state

        Reward components:
        1. Task reward: Forward velocity for walk_forward
        2. Standing reward: Bonus for body height > 0.08m
        3. Stability: Penalty for large roll/pitch angles
        4. Energy: Penalty for large joint movements
        5. Survival: Small bonus per timestep alive
        """
        # Extract state
        body_pos = pipeline_state.x.pos[0]
        body_rot = pipeline_state.x.rot[0]
        body_vel = pipeline_state.x.vel[0]

        # 1. Task reward (forward velocity)
        gait_type, direction, turn = self._gait_command

        # Forward velocity bonus
        forward_reward = 1.0 * body_vel[0] * direction  # Scale: ±1.0

        # 2. Standing upright reward
        height_target = 0.08  # meters
        height_reward = 2.0 * jnp.exp(-10.0 * (body_pos[2] - height_target)**2)

        # 3. Stability reward (low roll/pitch)
        roll, pitch, yaw = body_rot
        orientation_penalty = -0.5 * (jnp.square(roll) + jnp.square(pitch))

        # 4. Energy penalty (discourage large movements)
        energy_penalty = -0.01 * jnp.sum(jnp.square(action))

        # 5. Survival bonus
        survival_reward = 0.1

        # Total reward
        reward = (
            forward_reward +
            height_reward +
            orientation_penalty +
            energy_penalty +
            survival_reward
        )

        return reward


# Register environment with Brax
envs.register_environment('pidog', PiDogBraxEnv)


if __name__ == '__main__':
    """Test environment creation and basic stepping"""
    import time

    print("Creating PiDog Brax environment...")
    env = PiDogBraxEnv(gait_command=(0, 1, 0))  # walk_forward

    print(f"Observation space: {env.observation_size}D")
    print(f"Action space: {env.action_size}D")

    # Test reset
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print(f"Initial obs shape: {state.obs.shape}")
    print(f"Initial reward: {state.reward}")

    # Test stepping
    print("\nTesting 100 random steps...")
    start = time.time()
    for _ in range(100):
        rng, rng_act = jax.random.split(rng)
        action = jax.random.uniform(rng_act, (8,), minval=-1.0, maxval=1.0)
        state = env.step(state, action)

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.3f}s ({100/elapsed:.0f} steps/sec)")
    print(f"Final body height: {state.metrics['body_height']:.3f}m")
    print(f"Final forward velocity: {state.metrics['forward_velocity']:.3f}m/s")

    print("\n✅ Environment created successfully!")
    print("Next: Train with Brax PPO using train_brax_ppo.py")
