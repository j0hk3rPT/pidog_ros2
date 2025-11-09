#!/usr/bin/env python3
"""
Test a trained RL model in Gazebo simulation

Usage:
    python3 test_rl_model.py --model models/rl_vision_fast/final_model.zip --episodes 10
"""

import argparse
import numpy as np
import time
from stable_baselines3 import PPO
import rclpy

# Import the environment
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'install/pidog_gaits/lib/python3.12/site-packages'))

from pidog_gaits.pidog_rl_env_vision import PiDogVisionEnv


def test_model(model_path, num_episodes=10, render=True):
    """
    Test a trained RL model.

    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to run
        render: Whether to render (not used in headless mode)
    """

    print("=" * 60)
    print("Testing Vision-Based RL Model")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    print()

    # Load the trained model
    print("Loading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded successfully")
    print()

    # Create environment
    print("Creating environment...")
    env = PiDogVisionEnv(node_name='test_rl_env')
    print("✓ Environment created")
    print()

    # Statistics
    episode_rewards = []
    episode_lengths = []

    try:
        for episode in range(num_episodes):
            print(f"[Episode {episode + 1}/{num_episodes}]")

            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            step = 0

            while not (done or truncated):
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, done, truncated, info = env.step(action)

                episode_reward += reward
                step += 1

                # Print progress every 50 steps
                if step % 50 == 0:
                    print(f"  Step {step}: reward={reward:.2f}, total={episode_reward:.2f}")
                    if 'forward_vel' in info:
                        print(f"    Forward velocity: {info['forward_vel']:.3f} m/s")
                    if 'body_z' in info:
                        print(f"    Body height: {info['body_z']:.3f} m")

                # Check termination reasons
                if done:
                    if info.get('fallen', False):
                        print(f"  ✗ Episode ended: Robot fell (step {step})")
                    elif info.get('head_contact', False):
                        print(f"  ✗ Episode ended: Head contact (step {step})")
                    else:
                        print(f"  Episode ended (step {step})")

                if truncated:
                    print(f"  ✓ Episode completed full duration (step {step})")

            # Episode summary
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

            print(f"  Episode reward: {episode_reward:.2f}")
            print(f"  Episode length: {step} steps")

            # Show final state info
            if 'forward_vel' in info:
                print(f"  Final forward velocity: {info['forward_vel']:.3f} m/s")
            if 'speed' in info:
                print(f"  Final speed: {info['speed']:.3f} m/s")
            if 'ultrasonic_range' in info:
                print(f"  Final obstacle distance: {info['ultrasonic_range']:.2f} m")

            print()

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    finally:
        env.close()

    # Print statistics
    if episode_rewards:
        print("=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Best reward: {np.max(episode_rewards):.2f}")
        print(f"Worst reward: {np.min(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
        print("=" * 60)

    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description='Test trained RL model in Gazebo')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes (default: 10)')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Render the environment (default: True)')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("\nAvailable models:")
        if os.path.exists('models'):
            for root, dirs, files in os.walk('models'):
                for f in files:
                    if f.endswith('.zip'):
                        print(f"  - {os.path.join(root, f)}")
        return 1

    # Run test
    test_model(args.model, args.episodes, args.render)

    return 0


if __name__ == '__main__':
    exit(main())
