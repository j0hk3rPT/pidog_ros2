#!/usr/bin/env python3
"""
View PiDog in Stand Position

Shows the PiDog robot model in MuJoCo viewer with correct stand pose.
Based on SunFounder PiDog specifications.

Usage:
    python3 view_pidog_stand.py

Controls:
    - Left mouse: Rotate camera
    - Right mouse: Translate camera
    - Scroll: Zoom
    - SPACEBAR: Pause/resume
    - ESC: Exit
"""

import sys
import mujoco
import mujoco.viewer
from pidog_brax_mjcf import generate_pidog_mjcf
from pidog_gaits.inverse_kinematics import LegIK
import numpy as np


def get_stand_position():
    """
    Calculate stand position angles using SunFounder parameters.

    SunFounder stand position:
    - Front legs: [y, z] = [-15, 95] mm
    - Hind legs: [y, z] = [5, 90] mm (shifted for stability)

    Returns:
        numpy.array: 8 joint angles in controller order [BR, FR, BL, FL]
    """
    # Stand coordinates from SunFounder actions_dictionary.py:
    # x (barycenter) = -15, y (height) = 95
    # Front legs: [x, y] = [-15, 95]
    # Hind legs: [x+20, y-5] = [5, 90]

    # Convert to meters (IK expects mm, but we'll use mm)
    front_coord = [-15.0, 95.0]  # FL, FR
    hind_coord = [5.0, 90.0]     # BL, BR

    # Leg coordinates in gait order: FL, FR, BL, BR
    leg_coords = [
        front_coord,  # FL (Front Left)
        front_coord,  # FR (Front Right)
        hind_coord,   # BL (Back Left)
        hind_coord,   # BR (Back Right)
    ]

    # Convert to joint angles (returns controller order: BR, FR, BL, FL)
    angles = LegIK.legs_coords_to_angles(leg_coords)

    return np.array(angles)


def main():
    """Launch MuJoCo viewer with PiDog in stand position"""

    print("=" * 60)
    print("PiDog Stand Position Viewer")
    print("=" * 60)
    print("\nLoading PiDog model...")

    # Generate MJCF model
    mjcf_xml = generate_pidog_mjcf()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    print(f"✓ Model loaded: {model.nq} DOF, {model.nu} actuators")

    # Get stand position angles
    stand_angles = get_stand_position()

    print("\nStand Position Joint Angles (radians):")
    print("=" * 60)
    joint_names = [
        "BR shoulder", "BR knee",
        "FR shoulder", "FR knee",
        "BL shoulder", "BL knee",
        "FL shoulder", "FL knee"
    ]
    for i, (name, angle) in enumerate(zip(joint_names, stand_angles)):
        print(f"  {name:12s}: {angle:7.3f} rad ({np.degrees(angle):7.2f}°)")

    # Set initial position
    # qpos structure: [px, py, pz, qw, qx, qy, qz, leg_joints...]
    # Start with body at 80mm height (Z_ORIGIN from SunFounder)
    data.qpos[0] = 0.0       # X position
    data.qpos[1] = 0.0       # Y position
    data.qpos[2] = 0.08      # Z position (80mm)
    data.qpos[3] = 1.0       # Quaternion W
    data.qpos[4] = 0.0       # Quaternion X
    data.qpos[5] = 0.0       # Quaternion Y
    data.qpos[6] = 0.0       # Quaternion Z

    # Set leg joint angles (qpos[7:15])
    data.qpos[7:15] = stand_angles

    # Set control targets to match position
    data.ctrl[:8] = stand_angles

    # Forward kinematics to settle
    mujoco.mj_forward(model, data)

    print("\nBody Position:")
    print("=" * 60)
    print(f"  Height: {data.qpos[2]*1000:.1f} mm")
    print(f"  Position: ({data.qpos[0]*1000:.1f}, {data.qpos[1]*1000:.1f}, {data.qpos[2]*1000:.1f}) mm")

    print("\nPhysical Dimensions (SunFounder Reference):")
    print("=" * 60)
    print(f"  Upper leg (LEG):  42 mm")
    print(f"  Lower leg (FOOT): 76 mm")
    print(f"  Body length:      117 mm")
    print(f"  Body width:       98 mm")
    print(f"  Body height:      80 mm (standing)")

    print("\nLaunching MuJoCo viewer...")
    print("=" * 60)
    print("Controls:")
    print("  - Left mouse drag: Rotate camera")
    print("  - Right mouse drag: Translate camera")
    print("  - Scroll wheel: Zoom")
    print("  - SPACEBAR: Pause/resume simulation")
    print("  - ESC: Exit viewer")
    print("=" * 60)

    # Launch passive viewer (user can interact)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera to good viewing angle
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0, 0, 0.08]

        # Keep viewer open
        while viewer.is_running():
            # Apply control to maintain stand position
            data.ctrl[:8] = stand_angles

            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer (30 FPS)
            viewer.sync()

    print("\nViewer closed.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
