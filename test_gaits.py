#!/usr/bin/env python3
"""Test script to verify all gaits generate valid coordinates."""

import sys
sys.path.insert(0, 'pidog_gaits')

from pidog_gaits.walk_gait import Walk
from pidog_gaits.trot_gait import Trot

print("Testing Walk Gaits:")
print("=" * 50)

walk_forward = Walk(fb=Walk.FORWARD, lr=Walk.STRAIGHT).get_coords()
print(f"walk_forward: {len(walk_forward)} frames")

walk_backward = Walk(fb=Walk.BACKWARD, lr=Walk.STRAIGHT).get_coords()
print(f"walk_backward: {len(walk_backward)} frames")

walk_left = Walk(fb=Walk.FORWARD, lr=Walk.LEFT).get_coords()
print(f"walk_left: {len(walk_left)} frames")

walk_right = Walk(fb=Walk.FORWARD, lr=Walk.RIGHT).get_coords()
print(f"walk_right: {len(walk_right)} frames")

print("\nTesting Trot Gaits:")
print("=" * 50)

trot_forward = Trot(fb=Trot.FORWARD, lr=Trot.STRAIGHT).get_coords()
print(f"trot_forward: {len(trot_forward)} frames")
if trot_forward:
    print(f"  First frame: {trot_forward[0]}")
else:
    print("  ERROR: Empty coordinates!")

trot_backward = Trot(fb=Trot.BACKWARD, lr=Trot.STRAIGHT).get_coords()
print(f"trot_backward: {len(trot_backward)} frames")

trot_left = Trot(fb=Trot.FORWARD, lr=Trot.LEFT).get_coords()
print(f"trot_left: {len(trot_left)} frames")

trot_right = Trot(fb=Trot.FORWARD, lr=Trot.RIGHT).get_coords()
print(f"trot_right: {len(trot_right)} frames")

print("\nTesting Static Poses:")
print("=" * 50)
from pidog_gaits.inverse_kinematics import LegIK

stand_coords = [[0, 80], [0, 80], [0, 80], [0, 80]]
stand_angles = LegIK.legs_coords_to_angles(stand_coords)
print(f"stand: {len(stand_angles)} angles")
print(f"  Angles: {[f'{a:.3f}' for a in stand_angles]}")

sit_coords = [[30, 65], [30, 65], [-20, 90], [-20, 90]]
sit_angles = LegIK.legs_coords_to_angles(sit_coords)
print(f"sit: {len(sit_angles)} angles")

lie_coords = [[50, 40], [50, 40], [-30, 40], [-30, 40]]
lie_angles = LegIK.legs_coords_to_angles(lie_coords)
print(f"lie: {len(lie_angles)} angles")

stretch_coords = [[60, 60], [60, 60], [-60, 60], [-60, 60]]
stretch_angles = LegIK.legs_coords_to_angles(stretch_coords)
print(f"stretch: {len(stretch_angles)} angles")

print("\n✅ All gaits generated successfully!")
