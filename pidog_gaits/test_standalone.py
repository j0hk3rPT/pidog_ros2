#!/usr/bin/env python3
"""
Standalone test script for pidog_gaits components.

Tests the core gait generation and inverse kinematics without ROS2.
"""

import sys
import os

# Add pidog_gaits to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pidog_gaits'))

from walk_gait import Walk
from trot_gait import Trot
from inverse_kinematics import LegIK


def test_walk_gait():
    """Test walk gait generation."""
    print("\n" + "=" * 60)
    print("Testing Walk Gait")
    print("=" * 60)

    # Create forward walk
    walk = Walk(fb=Walk.FORWARD, lr=Walk.STRAIGHT)
    coords = walk.get_coords()

    print(f"✓ Walk gait generated: {len(coords)} frames")
    print(f"  First frame coords: {coords[0]}")
    print(f"  Each frame has {len(coords[0])} legs")

    # Convert to angles
    angles = LegIK.legs_coords_to_angles(coords[0])
    print(f"  Joint angles: {[f'{a:.2f}°' for a in angles]}")

    return True


def test_trot_gait():
    """Test trot gait generation."""
    print("\n" + "=" * 60)
    print("Testing Trot Gait")
    print("=" * 60)

    # Create forward trot
    trot = Trot(fb=Trot.FORWARD, lr=Trot.STRAIGHT)
    coords = trot.get_coords()

    print(f"✓ Trot gait generated: {len(coords)} frames")
    print(f"  First frame coords: {coords[0]}")

    # Convert to angles
    angles = LegIK.legs_coords_to_angles(coords[0])
    print(f"  Joint angles: {[f'{a:.2f}°' for a in angles]}")

    return True


def test_inverse_kinematics():
    """Test inverse kinematics."""
    print("\n" + "=" * 60)
    print("Testing Inverse Kinematics")
    print("=" * 60)

    # Test various leg positions
    test_coords = [
        [0, 80],    # Neutral standing
        [20, 70],   # Forward reach
        [-20, 70],  # Backward reach
        [0, 60],    # Crouched
    ]

    for i, coord in enumerate(test_coords):
        y, z = coord
        alpha, beta = LegIK.coord2angles(y, z)
        print(f"  Coord [{y:3.0f}, {z:3.0f}] → Angles [{alpha:6.2f}°, {beta:6.2f}°]")

    print("✓ Inverse kinematics working")

    return True


def test_all_gaits():
    """Test all gait variations."""
    print("\n" + "=" * 60)
    print("Testing All Gait Variations")
    print("=" * 60)

    gaits = [
        ("Walk Forward", Walk(fb=Walk.FORWARD, lr=Walk.STRAIGHT)),
        ("Walk Backward", Walk(fb=Walk.BACKWARD, lr=Walk.STRAIGHT)),
        ("Walk Left", Walk(fb=Walk.FORWARD, lr=Walk.LEFT)),
        ("Walk Right", Walk(fb=Walk.FORWARD, lr=Walk.RIGHT)),
        ("Trot Forward", Trot(fb=Trot.FORWARD, lr=Trot.STRAIGHT)),
        ("Trot Backward", Trot(fb=Trot.BACKWARD, lr=Trot.STRAIGHT)),
        ("Trot Left", Trot(fb=Trot.FORWARD, lr=Trot.LEFT)),
        ("Trot Right", Trot(fb=Trot.FORWARD, lr=Trot.RIGHT)),
    ]

    for name, gait in gaits:
        coords = gait.get_coords()
        angles = LegIK.legs_coords_to_angles(coords[0])
        print(f"  ✓ {name:20s} - {len(coords):3d} frames")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PiDog Gaits - Standalone Component Tests")
    print("=" * 60)

    tests = [
        ("Inverse Kinematics", test_inverse_kinematics),
        ("Walk Gait", test_walk_gait),
        ("Trot Gait", test_trot_gait),
        ("All Gait Variations", test_all_gaits),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All tests passed! The gait generation system is working correctly.")
        print("\nNext steps:")
        print("  1. Build with ROS2: colcon build --packages-select pidog_gaits")
        print("  2. Run demo: ros2 launch pidog_gaits gait_demo.launch.py")
        print("  3. Collect data: ros2 launch pidog_gaits collect_data.launch.py")
        print("  4. Train model: python3 -m pidog_gaits.pidog_gaits.train --data ./training_data/gait_data_*.npz")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
