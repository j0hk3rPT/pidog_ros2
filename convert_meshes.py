#!/usr/bin/env python3
"""Convert DAE meshes to OBJ format for MuJoCo."""

import trimesh
import os
from pathlib import Path

mesh_dir = Path("pidog_description/meshes")
dae_files = list(mesh_dir.glob("*.dae"))

print("=" * 70)
print("Converting DAE Meshes to OBJ for MuJoCo")
print("=" * 70)

if not dae_files:
    print("No DAE files found in pidog_description/meshes/")
    exit(1)

print(f"\nFound {len(dae_files)} DAE files to convert\n")

converted = []
failed = []

for dae_file in sorted(dae_files):
    obj_file = dae_file.with_suffix('.obj')

    print(f"Converting: {dae_file.name} → {obj_file.name}...", end=" ")

    try:
        # Load DAE mesh
        mesh = trimesh.load(str(dae_file))

        # Handle scene vs single mesh
        if isinstance(mesh, trimesh.Scene):
            # Merge all geometries in scene
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in mesh.geometry.values())
            )

        # Export to OBJ
        mesh.export(str(obj_file))

        # Get file sizes
        dae_size = dae_file.stat().st_size / 1024  # KB
        obj_size = obj_file.stat().st_size / 1024  # KB

        print(f"✅ ({dae_size:.1f}KB → {obj_size:.1f}KB)")
        converted.append(obj_file.name)

    except Exception as e:
        print(f"❌ FAILED: {e}")
        failed.append((dae_file.name, str(e)))

print("\n" + "=" * 70)
print("Conversion Summary")
print("=" * 70)
print(f"✅ Successfully converted: {len(converted)}")
for name in converted:
    print(f"   - {name}")

if failed:
    print(f"\n❌ Failed: {len(failed)}")
    for name, error in failed:
        print(f"   - {name}: {error}")

print("\n" + "=" * 70)
print("Next Steps:")
print("=" * 70)
if converted:
    print("1. OBJ files are now in pidog_description/meshes/")
    print("2. Update pidog_with_meshes.xml to use .obj instead of .dae")
    print("3. Test with: python3 test_meshes.py")
print("=" * 70)
