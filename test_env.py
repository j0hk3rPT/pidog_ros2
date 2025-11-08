#!/usr/bin/env python3
"""Quick test of PiDog MuJoCo Gymnasium environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pidog_gaits'))

from pidog_gaits.pidog_mujoco_env import PiDogMuJoCoEnv
import numpy as np

print("=" * 60)
print("Testing PiDog MuJoCo Gymnasium Environment")
print("=" * 60)

# Create environment
print("\n1. Creating environment...")
env = PiDogMuJoCoEnv(render_mode=None)
print(f"   ✅ Environment created")
print(f"   - Observation space: {env.observation_space.shape}")
print(f"   - Action space: {env.action_space.shape}")

# Reset environment
print("\n2. Resetting environment...")
obs, info = env.reset(seed=42)
print(f"   ✅ Environment reset")
print(f"   - Observation shape: {obs.shape}")
print(f"   - Initial height: {info['body_height']:.3f}m")
print(f"   - Initial upright: {info['upright']:.3f}")

# Take random actions
print("\n3. Taking 10 random steps...")
total_reward = 0
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if i == 0:
        print(f"   Step {i+1}:")
        print(f"   - Action shape: {action.shape}")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Reward: {reward:.3f}")
        print(f"   - Terminated: {terminated}")
        print(f"   - Truncated: {truncated}")

    if terminated or truncated:
        print(f"   Episode ended at step {i+1}")
        break

print(f"\n   Total reward: {total_reward:.3f}")
print(f"   Final height: {info['body_height']:.3f}m")

# Test observation components
print("\n4. Checking observation components...")
print(f"   - Joint positions (8): {obs[0:8]}")
print(f"   - Joint velocities (8): {obs[8:16]}")
print(f"   - IMU orientation (4): {obs[16:20]}")
print(f"   - IMU gyro (3): {obs[20:23]}")
print(f"   - IMU accel (3): {obs[23:26]}")
print(f"   - Body height (1): {obs[26]}")
print(f"   - Body velocity (1): {obs[27]}")
print(f"   ✅ All observation components present")

# Test reset consistency
print("\n5. Testing reset consistency...")
obs1, _ = env.reset(seed=123)
obs2, _ = env.reset(seed=123)
if np.allclose(obs1, obs2, atol=1e-5):
    print(f"   ✅ Reset is deterministic with seed")
else:
    print(f"   ⚠️  Reset has some randomness despite seed")

env.close()

print("\n" + "=" * 60)
print("✅ Environment test passed!")
print("=" * 60)
print("\nEnvironment is ready for RL training!")
print("Run: python3 train_mujoco.py")
