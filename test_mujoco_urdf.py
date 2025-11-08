#!/usr/bin/env python3
"""Test loading PiDog URDF in MuJoCo."""

import mujoco
import numpy as np

print("=" * 60)
print("Testing PiDog URDF in MuJoCo")
print("=" * 60)

# Try to load URDF (adjust path based on where script is run)
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, 'pidog_description/urdf/pidog.urdf')

# If running from container at /workspace
if not os.path.exists(urdf_path):
    urdf_path = '/workspace/pidog_description/urdf/pidog.urdf'

print(f"\n1. Loading URDF from: {urdf_path}")

try:
    model = mujoco.MjModel.from_xml_path(urdf_path)
    print(f"✅ URDF loaded successfully!")

    # Print model info
    print(f"\n2. Model Information:")
    print(f"   - Bodies: {model.nbody}")
    print(f"   - Joints: {model.njnt}")
    print(f"   - Actuators: {model.nu}")
    print(f"   - Sensors: {model.nsensor}")
    print(f"   - Timestep: {model.opt.timestep}s")

    # List all bodies
    print(f"\n3. Robot Bodies:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            print(f"   - {body_name}")

    # List all joints
    print(f"\n4. Robot Joints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            joint_type = model.jnt_type[i]
            type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
            print(f"   - {joint_name} ({type_names.get(joint_type, joint_type)})")

    # List all sensors
    print(f"\n5. Sensors:")
    if model.nsensor > 0:
        for i in range(model.nsensor):
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_type = model.sensor_type[i]
            if sensor_name:
                print(f"   - {sensor_name} (type {sensor_type})")
    else:
        print("   ⚠️  No sensors found in model")
        print("   → Need to add IMU sensors to MJCF")

    # Create data structure
    data = mujoco.MjData(model)
    print(f"\n6. Creating simulation data...")
    print(f"   ✅ MjData created successfully")

    # Test forward kinematics
    print(f"\n7. Testing physics simulation...")
    mujoco.mj_forward(model, data)
    print(f"   ✅ Forward kinematics computed")

    # Print robot pose
    if model.njnt > 0:
        print(f"\n8. Initial Robot State:")
        print(f"   - Base position: {data.qpos[:3]}")
        print(f"   - Base quaternion: {data.qpos[3:7]}")
        if len(data.qpos) > 7:
            print(f"   - Joint positions: {data.qpos[7:]}")

    # Test single physics step
    print(f"\n9. Running single physics step...")
    mujoco.mj_step(model, data)
    print(f"   ✅ Physics step successful")

    print(f"\n" + "=" * 60)
    print("✅ SUCCESS: PiDog URDF is fully compatible with MuJoCo!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Add IMU sensors to create MJCF version")
    print(f"2. Create Gymnasium environment")
    print(f"3. Start RL training")

except Exception as e:
    print(f"\n❌ Error loading URDF:")
    print(f"   {type(e).__name__}: {e}")
    print(f"\nThis might be due to:")
    print(f"- Gazebo-specific URDF tags")
    print(f"- Missing mesh files")
    print(f"- Incompatible physics parameters")
    print(f"\nWe can convert to native MJCF format to fix this.")
    import traceback
    traceback.print_exc()
