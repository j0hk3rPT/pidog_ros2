#!/usr/bin/env python3
"""
Collect training data from MJX simulation using traditional gaits

This script runs the existing gait generators (walk, trot) in MJX simulation
and records state-action pairs for imitation learning.

Output: .npz file with inputs (gait commands) and outputs (joint angles)
"""

import jax
import jax.numpy as jnp
import numpy as np
from pidog_mjx_env import PiDogMJXEnv
from datetime import datetime
from pathlib import Path
import argparse

# Import existing gait generators
import sys
sys.path.append(str(Path(__file__).parent / 'pidog_gaits' / 'pidog_gaits'))
from walk_gait import WalkGait
from trot_gait import TrotGait
from inverse_kinematics import LegIK


class MJXDataCollector:
    """Collect training data from MJX simulation"""

    def __init__(self, env: PiDogMJXEnv):
        self.env = env
        self.walk_gait = WalkGait()
        self.trot_gait = TrotGait()
        self.ik = LegIK()

        # Data storage
        self.inputs = []  # [gait_type, direction, turn, phase]
        self.outputs = []  # [8 joint angles]

    def collect_gait(
        self,
        gait_name: str,
        gait_type: int,
        direction: int,
        turn: int,
        duration_steps: int = 100,
        rng: jax.Array = None
    ):
        """
        Collect data for one gait variant

        Args:
            gait_name: "walk" or "trot"
            gait_type: 0=walk, 1=trot
            direction: -1=backward, 0=none, 1=forward
            turn: -1=left, 0=straight, 1=right
            duration_steps: Number of steps to collect
        """
        print(f"  Collecting {gait_name} (dir={direction}, turn={turn})...")

        # Reset environment
        if rng is None:
            rng = jax.random.PRNGKey(0)
        data = self.env.reset(rng)

        # Select gait generator
        if gait_name == "walk":
            gait_gen = self.walk_gait
        elif gait_name == "trot":
            gait_gen = self.trot_gait
        else:
            raise ValueError(f"Unknown gait: {gait_name}")

        # Generate gait trajectory
        for step in range(duration_steps):
            # Compute gait phase [0, 1]
            phase = (step % 100) / 100.0

            # Get target foot positions from gait generator
            # This calls the existing walk/trot gait logic
            target_coords = self._get_target_coords(
                gait_gen, direction, turn, phase
            )

            # Convert to joint angles via IK
            joint_angles = self.ik.legs_coords_to_angles(target_coords)

            # Extract only leg joints (8 DOF, skip head/tail)
            leg_angles = jnp.array(joint_angles[:8])

            # Store input-output pair
            input_vec = jnp.array([gait_type, direction, turn, phase])
            self.inputs.append(input_vec)
            self.outputs.append(leg_angles)

            # Step simulation
            data, reward, done, info = self.env.step(data, leg_angles, phase)

            if done:
                # Reset if robot falls
                data = self.env.reset(rng)

        print(f"    Collected {duration_steps} samples")

    def _get_target_coords(self, gait_gen, direction, turn, phase):
        """
        Get target foot coordinates from gait generator

        Returns: [[y1,z1], [y2,z2], [y3,z3], [y4,z4]] for FL, FR, BL, BR
        """
        # Call gait generator's coordinate function
        # The existing gaits use step index, so convert phase to step
        step = int(phase * 100)

        # Get coordinates for all 4 legs
        if hasattr(gait_gen, 'generate_coordinates'):
            coords = gait_gen.generate_coordinates(step, direction, turn)
        else:
            # Fallback: use default standing pose
            coords = [[0, 80], [0, 80], [0, 80], [0, 80]]  # Standing

        return coords

    def save_data(self, output_path: str):
        """Save collected data to .npz file"""
        inputs = np.array(self.inputs)
        outputs = np.array(self.outputs)

        print(f"\n💾 Saving data to {output_path}")
        print(f"   Inputs shape: {inputs.shape}")
        print(f"   Outputs shape: {outputs.shape}")

        np.savez_compressed(
            output_path,
            inputs=inputs,
            outputs=outputs
        )

        print(f"✅ Data saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Collect training data from MJX simulation'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=100,
        help='Steps per gait variant (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: training_data/gait_data_mjx_TIMESTAMP.npz)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path('./training_data')
    output_dir.mkdir(exist_ok=True)

    # Default output path with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = str(output_dir / f'gait_data_mjx_{timestamp}.npz')

    print("=" * 70)
    print("MJX Data Collection")
    print("=" * 70)
    print(f"Duration per gait: {args.duration} steps")
    print(f"Output file: {args.output}")
    print()

    # Create environment
    env = PiDogMJXEnv()
    collector = MJXDataCollector(env)

    # Collect data for different gaits
    rng = jax.random.PRNGKey(42)

    # Walk gaits
    print("Collecting WALK gaits...")
    collector.collect_gait("walk", gait_type=0, direction=1, turn=0,
                          duration_steps=args.duration, rng=rng)  # forward
    collector.collect_gait("walk", gait_type=0, direction=-1, turn=0,
                          duration_steps=args.duration, rng=rng)  # backward
    collector.collect_gait("walk", gait_type=0, direction=1, turn=1,
                          duration_steps=args.duration, rng=rng)  # right turn
    collector.collect_gait("walk", gait_type=0, direction=1, turn=-1,
                          duration_steps=args.duration, rng=rng)  # left turn

    # Trot gaits
    print("\nCollecting TROT gaits...")
    collector.collect_gait("trot", gait_type=1, direction=1, turn=0,
                          duration_steps=args.duration, rng=rng)  # forward
    collector.collect_gait("trot", gait_type=1, direction=-1, turn=0,
                          duration_steps=args.duration, rng=rng)  # backward
    collector.collect_gait("trot", gait_type=1, direction=1, turn=1,
                          duration_steps=args.duration, rng=rng)  # right turn
    collector.collect_gait("trot", gait_type=1, direction=1, turn=-1,
                          duration_steps=args.duration, rng=rng)  # left turn

    # Save data
    collector.save_data(args.output)

    print("\n" + "=" * 70)
    print("✅ Data collection complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Train imitation model:")
    print(f"   python3 train_imitation.py --data {args.output}")
    print(f"2. Fine-tune with RL:")
    print(f"   python3 train_mjx_ppo.py --pretrained models/imitation_model.pth")


if __name__ == '__main__':
    main()
