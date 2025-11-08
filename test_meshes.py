#!/usr/bin/env python3
"""Test PiDog with actual mesh files loaded."""

import mujoco
import mujoco.viewer
import numpy as np
import time

print("=" * 70)
print("Testing PiDog with Actual Mesh Files")
print("=" * 70)

# Load MJCF with meshes
mjcf_path = 'pidog_description/mjcf/pidog_with_meshes.xml'
print(f"\n1. Loading MJCF with meshes from: {mjcf_path}")

try:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    print(f"✅ Model with meshes loaded successfully!")

    # Print mesh info
    print(f"\n2. Mesh Information:")
    print(f"   - Number of meshes: {model.nmesh}")
    for i in range(model.nmesh):
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        if mesh_name:
            print(f"   - {mesh_name}")

    # Set standing pose
    print(f"\n3. Setting standing pose...")
    standing_pose = [-1.2, 0.18, -1.2, 0.18, 1.2, -0.18, 1.2, -0.18]
    data.ctrl[:8] = standing_pose

    # Let physics settle
    for _ in range(500):
        mujoco.mj_step(model, data)

    body_height = data.qpos[2]
    print(f"   ✅ Robot standing (body height: {body_height:.3f}m)")

    # Launch viewer
    print(f"\n4. Launching viewer with actual PiDog meshes...")
    print(f"   This should look like your real robot!")
    print(f"\n   🎮 Controls:")
    print(f"   - Left mouse drag: Rotate view")
    print(f"   - Right mouse drag: Pan view")
    print(f"   - Scroll wheel: Zoom")
    print(f"   - ESC: Close\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Keep robot standing
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 30:
            data.ctrl[:8] = standing_pose
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print(f"\n✅ Mesh visualization complete!")

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print(f"   Make sure you're running from repository root")
    print(f"   Current directory: {os.getcwd()}")

except Exception as e:
    print(f"\n❌ Error loading meshes:")
    print(f"   {type(e).__name__}: {e}")
    print(f"\nPossible issues:")
    print(f"   - Mesh files might be in unsupported format")
    print(f"   - File paths might be wrong")
    print(f"\nNote: Simple model (no meshes) still works for training!")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 70)
print(f"Note: Meshes are for visualization only.")
print(f"Physics uses simple collision shapes (faster simulation).")
print(f"=" * 70)
