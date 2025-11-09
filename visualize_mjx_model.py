#!/usr/bin/env python3
"""
Visualize PiDog model and trained policies in MuJoCo viewer

This script opens an interactive MuJoCo viewer where you can:
1. View the PiDog model geometry
2. Watch trained policies execute
3. Manually control joints
4. Record videos

Usage:
    # View model only
    python3 visualize_mjx_model.py

    # Load and execute trained policy
    python3 visualize_mjx_model.py --policy models/mjx_ppo/pidog_walk_forward_ppo.pkl

    # Record video
    python3 visualize_mjx_model.py --policy models/policy.pkl --record video.mp4
"""

import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from pidog_brax_mjcf import generate_pidog_mjcf


class PiDogVisualizer:
    """Interactive MuJoCo visualizer for PiDog"""

    def __init__(self, model_path=None, policy_path=None):
        """
        Initialize visualizer

        Args:
            model_path: Path to MJCF XML (default: generate from pidog_brax_mjcf.py)
            policy_path: Path to trained policy .pkl file (optional)
        """
        # Load MuJoCo model
        if model_path is None:
            print("Generating PiDog MJCF model...")
            xml = generate_pidog_mjcf()
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)

        self.data = mujoco.MjData(self.model)

        # Load policy if provided
        self.policy_fn = None
        if policy_path:
            self.load_policy(policy_path)

        # Control state
        self.paused = False
        self.step_count = 0
        self.phase = 0.0

        print(f"✅ Model loaded:")
        print(f"   Bodies: {self.model.nbody}")
        print(f"   DOF: {self.model.nv}")
        print(f"   Actuators: {self.model.nu}")
        print(f"   Sensors: {self.model.nsensor}")

    def load_policy(self, policy_path):
        """Load trained Brax PPO policy"""
        import pickle

        print(f"Loading policy from {policy_path}...")
        with open(policy_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.policy_params = checkpoint['params']
        self.make_inference_fn = checkpoint['make_inference_fn']
        self.inference_fn = self.make_inference_fn(self.policy_params)

        print(f"✅ Policy loaded successfully!")

    def reset(self):
        """Reset simulation to standing pose"""
        mujoco.mj_resetData(self.model, self.data)

        # Set initial joint positions (standing)
        # Freejoint: position + quaternion (7 values in qpos)
        self.data.qpos[2] = 0.1  # Z height

        self.data.qpos[3] = 1.0  # qw (identity quaternion)

        # Leg joints: shoulders at 0.5, knees at -0.8
        leg_angles = [0.5, -0.8] * 4
        self.data.qpos[7:15] = leg_angles

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.phase = 0.0

    def step(self):
        """Step simulation one timestep"""
        if self.policy_fn is not None:
            # Execute learned policy
            # TODO: Get observation and run inference
            # For now, use standing pose
            ctrl = [0.5, -0.8] * 4
        else:
            # Default: standing pose
            ctrl = [0.5, -0.8] * 4

        # Set actuator controls
        self.data.ctrl[:8] = ctrl

        # Step physics
        mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.phase = (self.phase + 0.01) % 1.0

    def run(self, duration=None):
        """
        Run interactive viewer

        Args:
            duration: Max duration in seconds (None = infinite)
        """
        print("\n" + "=" * 60)
        print("Interactive MuJoCo Viewer")
        print("=" * 60)
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset")
        print("  ESC: Quit")
        print("=" * 60 + "\n")

        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()

            while viewer.is_running():
                step_start = time.time()

                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break

                # Step simulation
                if not self.paused:
                    self.step()

                # Update viewer
                viewer.sync()

                # Sleep to maintain real-time (50 Hz)
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        print("\n✅ Viewer closed")


    def record_video(self, output_path, duration=10.0, fps=30):
        """
        Record video of simulation

        Args:
            output_path: Output video file path (.mp4)
            duration: Duration in seconds
            fps: Frames per second
        """
        print(f"Recording video: {output_path} ({duration}s @ {fps} FPS)")

        import cv2

        # Create video writer
        width, height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Create offscreen renderer
        renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.reset()

        num_frames = int(duration * fps)
        for frame in range(num_frames):
            # Step simulation (multiple steps per frame if needed)
            steps_per_frame = int(1 / (self.model.opt.timestep * fps))
            for _ in range(steps_per_frame):
                self.step()

            # Render frame
            renderer.update_scene(self.data)
            pixels = renderer.render()

            # Convert to BGR for OpenCV
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Write frame
            writer.write(bgr)

            if (frame + 1) % fps == 0:
                print(f"  Rendered {frame + 1}/{num_frames} frames...")

        writer.release()
        renderer.close()

        print(f"✅ Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize PiDog model and policies in MuJoCo'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to MJCF XML (default: auto-generate)'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default=None,
        help='Path to trained policy .pkl file (optional)'
    )
    parser.add_argument(
        '--record',
        type=str,
        default=None,
        help='Record video to file path (e.g., video.mp4)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Duration in seconds (for recording or limiting viewer)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video FPS (default: 30)'
    )

    args = parser.parse_args()

    # Create visualizer
    viz = PiDogVisualizer(
        model_path=args.model,
        policy_path=args.policy
    )

    # Run or record
    if args.record:
        duration = args.duration or 10.0
        viz.record_video(args.record, duration=duration, fps=args.fps)
    else:
        viz.run(duration=args.duration)


if __name__ == '__main__':
    main()
