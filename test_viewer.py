#!/usr/bin/env python3
"""
Visualize PiDog in MuJoCo with interactive viewer.
Test that rendering works before starting training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pidog_gaits'))

from pidog_gaits.pidog_mujoco_env import PiDogMuJoCoEnv
import numpy as np
import time

print("=" * 70)
print("PiDog MuJoCo Visualization Test")
print("=" * 70)

# Check DISPLAY environment variable
display = os.environ.get('DISPLAY')
if not display:
    print("\n⚠️  WARNING: DISPLAY environment variable not set!")
    print("   Viewer might not work. To fix:")
    print("   1. On host: xhost +local:docker")
    print("   2. In container: export DISPLAY=:0")
    print("\nAttempting to start viewer anyway...\n")
else:
    print(f"\n✅ DISPLAY is set: {display}")

# Create environment with rendering
print("\n1. Creating environment with rendering...")
try:
    env = PiDogMuJoCoEnv(render_mode="human")
    print("   ✅ Environment created with render_mode='human'")
except Exception as e:
    print(f"   ❌ Failed to create environment: {e}")
    print("\nTrying without viewer (headless mode)...")
    env = PiDogMuJoCoEnv(render_mode=None)
    print("   ✅ Headless mode works (training will still work)")
    env.close()
    sys.exit(1)

# Reset environment
print("\n2. Resetting environment...")
obs, info = env.reset(seed=42)
print(f"   ✅ Robot initialized")
print(f"   - Initial height: {info['body_height']:.3f}m")
print(f"   - Initial upright: {info['upright']:.3f}")

# Test with standing pose
print("\n3. Testing standing pose with viewer...")
print("   🎮 MuJoCo Viewer Controls:")
print("   - Left mouse drag: Rotate view")
print("   - Right mouse drag: Pan view")
print("   - Scroll wheel: Zoom")
print("   - ESC: Close viewer")
print("\n   Robot will try to stand for 5 seconds...")

# Standing pose action
standing_action = np.array([-0.76, 0.11, -0.76, 0.11, 0.76, -0.11, 0.76, -0.11])

try:
    for i in range(250):  # 5 seconds at 50 Hz
        obs, reward, terminated, truncated, info = env.step(standing_action)
        env.render()  # Update viewer
        time.sleep(0.02)  # 50 Hz

        if i % 50 == 0:
            print(f"   Step {i}: height={info['body_height']:.3f}m, reward={reward:.2f}")

        if terminated or truncated:
            print(f"\n   Episode ended at step {i}")
            break

    print("\n✅ Viewer test successful!")

    # Test with random actions
    print("\n4. Testing random actions...")
    print("   Robot will move randomly for 3 seconds...")

    obs, info = env.reset()

    for i in range(150):  # 3 seconds
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            print(f"\n   Episode ended (robot fell)")
            break

    print("\n✅ Random action test complete!")

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")

finally:
    env.close()
    print("\n" + "=" * 70)
    print("✅ Visualization test complete!")
    print("=" * 70)
    print("\nIf viewer worked:")
    print("  ✅ You're ready to visualize training")
    print("  ✅ Run: python3 train_mujoco.py --render")
    print("\nIf viewer failed:")
    print("  ✅ Headless training still works (faster!)")
    print("  ✅ Run: python3 train_mujoco.py")
