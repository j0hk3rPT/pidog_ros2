#!/usr/bin/env python3
"""
Render PiDog Brax simulation to HTML viewer

This creates an HTML file with embedded 3D visualization that you can
open in a web browser. Useful for viewing on remote machines without display.
"""

import jax
from brax.io import html
from pidog_mjx_env import PiDogMJXEnv
import argparse


def render_to_html(output_path='pidog_simulation.html', duration_steps=500):
    """
    Render simulation to interactive HTML

    Args:
        output_path: Output HTML file path
        duration_steps: Number of simulation steps to render
    """
    print(f"Creating Brax environment...")

    # Create environment
    env = PiDogMJXEnv(gait_command=(0, 1, 0))  # walk_forward

    print(f"Running {duration_steps} simulation steps...")

    # Initialize
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    # Collect trajectory
    states = [state]

    for step in range(duration_steps):
        # Simple standing action for now
        action = jax.numpy.array([0.5, -0.8] * 4)  # Standing pose
        phase = (step % 100) / 100.0

        state, reward, done, info = env.step(state, action, phase)
        states.append(state)

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{duration_steps}")

    print(f"\nRendering to HTML...")

    # Render to HTML with interactive viewer
    html_content = html.render(env.model.tree_replace({'opt.timestep': env.dt}), states)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✅ Saved interactive HTML: {output_path}")
    print(f"\nOpen in browser:")
    print(f"  firefox {output_path}")
    print(f"  # or")
    print(f"  chromium {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Render Brax simulation to HTML')
    parser.add_argument('--output', type=str, default='pidog_simulation.html',
                       help='Output HTML file path')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of simulation steps')

    args = parser.parse_args()

    render_to_html(args.output, args.steps)


if __name__ == '__main__':
    main()
