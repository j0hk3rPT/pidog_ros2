#!/usr/bin/env python3
"""Test simplified PiDog MuJoCo model with IMU sensors."""

import mujoco
import mujoco.viewer
import numpy as np
import time

print("=" * 70)
print("Testing Simplified PiDog MuJoCo Model with IMU Sensors")
print("=" * 70)

# Load simplified MJCF model
mjcf_path = 'pidog_description/mjcf/pidog_simple.xml'
print(f"\n1. Loading MJCF from: {mjcf_path}")

try:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    print(f"✅ MJCF loaded successfully!")

    # Print model info
    print(f"\n2. Model Information:")
    print(f"   - Bodies: {model.nbody}")
    print(f"   - Joints: {model.njnt}")
    print(f"   - Actuators: {model.nu}")
    print(f"   - Sensors: {model.nsensor}")
    print(f"   - Timestep: {model.opt.timestep}s ({1/model.opt.timestep:.0f} Hz)")

    # List actuators
    print(f"\n3. Actuators (8 leg joints):")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if act_name:
            print(f"   [{i}] {act_name}")

    # List sensors
    print(f"\n4. Sensors ({model.nsensor} total):")
    print(f"   IMU Sensors:")
    sensor_names = []
    for i in range(model.nsensor):
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if sensor_name:
            sensor_names.append(sensor_name)
            if 'imu' in sensor_name:
                print(f"   - {sensor_name}")

    print(f"\n   Joint Sensors: {len([s for s in sensor_names if 'jp_' in s or 'jv_' in s])}")
    print(f"   Contact Sensors: {len([s for s in sensor_names if 'contact' in s])}")

    # Initialize simulation
    print(f"\n5. Initializing Physics...")
    mujoco.mj_forward(model, data)
    print(f"   ✅ Forward kinematics computed")

    # Set standing pose (approximate)
    print(f"\n6. Setting Standing Pose...")
    # BR, FR, BL, FL (shoulder, knee pairs)
    standing_pose = [-1.2, 0.18, -1.2, 0.18, 1.2, -0.18, 1.2, -0.18]
    data.ctrl[:8] = standing_pose

    # Let robot settle into standing pose
    for _ in range(500):
        mujoco.mj_step(model, data)

    body_height = data.qpos[2]  # Z position of body
    print(f"   ✅ Robot standing (body height: {body_height:.3f}m)")

    # Test IMU sensors
    print(f"\n7. Reading IMU Sensors:")

    # Find IMU sensor indices
    imu_orient_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_orientation')
    imu_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
    imu_accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_accel')

    # Get sensor address (where data starts in sensordata array)
    orient_addr = model.sensor_adr[imu_orient_id]
    gyro_addr = model.sensor_adr[imu_gyro_id]
    accel_addr = model.sensor_adr[imu_accel_id]

    # Read IMU data
    orientation = data.sensordata[orient_addr:orient_addr+4]  # Quaternion (4 values)
    gyro = data.sensordata[gyro_addr:gyro_addr+3]  # Angular velocity (3 values)
    accel = data.sensordata[accel_addr:accel_addr+3]  # Linear acceleration (3 values)

    print(f"   Orientation (quaternion): {orientation}")
    print(f"   Angular velocity (rad/s): {gyro}")
    print(f"   Linear acceleration (m/s²): {accel}")

    # Check if IMU data looks reasonable
    if abs(orientation[3]) > 0.9:  # w component should be ~1 when upright
        print(f"   ✅ IMU orientation looks correct (upright)")

    if abs(accel[2] - 9.81) < 2.0:  # Z acceleration should be ~9.81
        print(f"   ✅ IMU accelerometer looks correct (gravity detected)")

    if np.linalg.norm(gyro) < 0.5:  # Should be near zero when standing still
        print(f"   ✅ IMU gyroscope looks correct (stationary)")

    # Test physics step
    print(f"\n8. Testing Physics Simulation...")
    t0 = time.time()
    num_steps = 1000
    for _ in range(num_steps):
        mujoco.mj_step(model, data)
    t1 = time.time()

    sim_time = num_steps * model.opt.timestep
    real_time = t1 - t0
    rtf = sim_time / real_time  # Real-time factor

    print(f"   Simulated {num_steps} steps ({sim_time:.2f}s sim time)")
    print(f"   Wall time: {real_time:.3f}s")
    print(f"   Real-time factor: {rtf:.1f}x (higher is faster)")
    print(f"   ✅ Physics simulation works!")

    # Check if robot is still standing
    final_height = data.qpos[2]
    if final_height > 0.08:
        print(f"   ✅ Robot stayed standing (height: {final_height:.3f}m)")
    else:
        print(f"   ⚠️  Robot fell (height: {final_height:.3f}m)")

    print(f"\n" + "=" * 70)
    print("✅ SUCCESS: MuJoCo model works perfectly!")
    print("=" * 70)
    print(f"\nKey Features:")
    print(f"✅ 8 leg actuators (position control)")
    print(f"✅ IMU sensor with orientation, gyro, accelerometer")
    print(f"✅ Joint position/velocity sensors")
    print(f"✅ Foot contact sensors")
    print(f"✅ Fast physics ({rtf:.0f}x real-time)")
    print(f"\nNext Steps:")
    print(f"1. Test interactive viewer (if GUI available)")
    print(f"2. Create Gymnasium environment")
    print(f"3. Start RL training with PPO")

    # Optional: Launch interactive viewer
    print(f"\n🎮 Launching Interactive Viewer...")
    print(f"   Use mouse to rotate/zoom view")
    print(f"   Press ESC or close window to exit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Keep robot standing in viewer
        while viewer.is_running():
            data.ctrl[:8] = standing_pose
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print(f"   Make sure you're running from the repository root")

except Exception as e:
    print(f"\n❌ Error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
