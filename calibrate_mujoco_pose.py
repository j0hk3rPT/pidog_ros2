#!/usr/bin/env python3
"""
Interactive calibration tool for MuJoCo standing pose.
Adjust joint angles and see the robot in real-time.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

print("=" * 70)
print("MuJoCo Standing Pose Calibration Tool")
print("=" * 70)

# Load model
model = mujoco.MjModel.from_xml_path('pidog_description/mjcf/pidog_simple.xml')
data = mujoco.MjData(model)

# Starting pose from Gazebo calibration
# Order: BR_shoulder, BR_knee, FR_shoulder, FR_knee, BL_shoulder, BL_knee, FL_shoulder, FL_knee
pose = {
    'br_shoulder': -1.208,
    'br_knee': +0.180,
    'fr_shoulder': -1.208,
    'fr_knee': +0.180,
    'bl_shoulder': +1.208,
    'bl_knee': -0.180,
    'fl_shoulder': +1.208,
    'fl_knee': -0.180,
}

def apply_pose(data, pose_dict):
    """Apply pose to MuJoCo actuators."""
    data.ctrl[0] = pose_dict['br_shoulder']
    data.ctrl[1] = pose_dict['br_knee']
    data.ctrl[2] = pose_dict['fr_shoulder']
    data.ctrl[3] = pose_dict['fr_knee']
    data.ctrl[4] = pose_dict['bl_shoulder']
    data.ctrl[5] = pose_dict['bl_knee']
    data.ctrl[6] = pose_dict['fl_shoulder']
    data.ctrl[7] = pose_dict['fl_knee']

def get_robot_state(model, data):
    """Get current robot state."""
    body_height = data.qpos[2]
    quat = data.sensordata[0:4]

    # Get foot positions
    feet = {}
    for name in ['back_right_foot', 'front_right_foot', 'back_left_foot', 'front_left_foot']:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        feet[name] = data.xpos[body_id].copy()

    return {
        'height': body_height,
        'quat': quat,
        'feet': feet
    }

def print_state(state):
    """Print robot state."""
    print(f"\n{'='*70}")
    print(f"Body height: {state['height']:.4f}m")
    print(f"Orientation: w={state['quat'][0]:.3f}, x={state['quat'][1]:.3f}, y={state['quat'][2]:.3f}, z={state['quat'][3]:.3f}")
    print(f"\nFoot heights:")
    for name, pos in state['feet'].items():
        on_ground = "✓" if pos[2] < 0.015 else "✗"
        print(f"  {name:20s}: z={pos[2]:.4f}m {on_ground}")

    # Check if standing
    all_on_ground = all(pos[2] < 0.015 for pos in state['feet'].values())
    upright = abs(state['quat'][0]) > 0.99
    good_height = 0.10 < state['height'] < 0.14

    print(f"\n{'='*70}")
    print(f"Standing check:")
    print(f"  All feet on ground: {'✓' if all_on_ground else '✗'}")
    print(f"  Body upright: {'✓' if upright else '✗'}")
    print(f"  Good height: {'✓' if good_height else '✗'}")
    print(f"  STANDING CORRECTLY: {'✅ YES' if (all_on_ground and upright and good_height) else '❌ NO'}")

def print_current_pose(pose_dict):
    """Print current pose as code."""
    print(f"\n{'='*70}")
    print("Current pose (copy to code):")
    print(f"standing_pose = [")
    print(f"    {pose_dict['br_shoulder']:+.3f}, {pose_dict['br_knee']:+.3f},  # Back Right")
    print(f"    {pose_dict['fr_shoulder']:+.3f}, {pose_dict['fr_knee']:+.3f},  # Front Right")
    print(f"    {pose_dict['bl_shoulder']:+.3f}, {pose_dict['bl_knee']:+.3f},  # Back Left")
    print(f"    {pose_dict['fl_shoulder']:+.3f}, {pose_dict['fl_knee']:+.3f},  # Front Left")
    print(f"]")

# Apply initial pose and settle
apply_pose(data, pose)
for _ in range(1000):
    mujoco.mj_step(model, data)

# Print initial state
print("\nInitial state with Gazebo calibrated pose:")
state = get_robot_state(model, data)
print_state(state)
print_current_pose(pose)

print(f"\n{'='*70}")
print("Interactive Adjustment Mode")
print("=" * 70)
print("\nCommands:")
print("  Adjust joint: <joint> <delta>")
print("    Joints: br_shoulder, br_knee, fr_shoulder, fr_knee,")
print("            bl_shoulder, bl_knee, fl_shoulder, fl_knee")
print("    Example: bl_shoulder -0.1")
print("  'reset' - Reset to Gazebo pose")
print("  'print' - Print current pose")
print("  'state' - Print robot state")
print("  'quit' - Exit")
print("\nTip: Adjust in small increments (0.05-0.1 rad)")
print("=" * 70)

while True:
    try:
        cmd = input("\n> ").strip().lower()

        if not cmd:
            continue

        if cmd == 'quit' or cmd == 'exit':
            break

        if cmd == 'reset':
            pose = {
                'br_shoulder': -1.208,
                'br_knee': +0.180,
                'fr_shoulder': -1.208,
                'fr_knee': +0.180,
                'bl_shoulder': +1.208,
                'bl_knee': -0.180,
                'fl_shoulder': +1.208,
                'fl_knee': -0.180,
            }
            print("Reset to Gazebo pose")
            apply_pose(data, pose)
            for _ in range(500):
                mujoco.mj_step(model, data)
            state = get_robot_state(model, data)
            print_state(state)
            continue

        if cmd == 'print':
            print_current_pose(pose)
            continue

        if cmd == 'state':
            state = get_robot_state(model, data)
            print_state(state)
            continue

        # Parse adjustment command
        parts = cmd.split()
        if len(parts) == 2:
            joint = parts[0]
            try:
                delta = float(parts[1])
            except ValueError:
                print(f"Invalid delta: {parts[1]}")
                continue

            if joint not in pose:
                print(f"Unknown joint: {joint}")
                print(f"Available: {', '.join(pose.keys())}")
                continue

            # Apply adjustment
            pose[joint] += delta
            print(f"Adjusted {joint} by {delta:+.3f} → {pose[joint]:+.3f}")

            # Apply and settle
            apply_pose(data, pose)
            for _ in range(500):
                mujoco.mj_step(model, data)

            # Show result
            state = get_robot_state(model, data)
            print_state(state)
        else:
            print("Invalid command. Type 'quit' to exit or '<joint> <delta>' to adjust.")

    except KeyboardInterrupt:
        print("\n\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 70)
print("Final calibrated pose:")
print_current_pose(pose)
print("=" * 70)
