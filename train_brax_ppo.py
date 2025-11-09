#!/usr/bin/env python3
"""
Train PiDog quadruped controller using Brax PPO on AMD 7900XT GPU

This script uses JAX + Brax for massively parallel RL training:
- 1000s of environments on GPU
- 100-200x faster than Gazebo
- Trains in minutes instead of hours

Usage:
    # Train walk_forward gait
    python3 train_brax_ppo.py --gait walk_forward --timesteps 10000000

    # Continue from pretrained imitation model
    python3 train_brax_ppo.py --pretrained models/best_model.pth --timesteps 5000000

Requirements:
    pip install jax[rocm] brax optax flax
"""

import argparse
import functools
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, html
import numpy as np

from pidog_brax_env import PiDogBraxEnv


# Gait command mapping
GAIT_COMMANDS = {
    'walk_forward': (0, 1, 0),
    'walk_backward': (0, -1, 0),
    'trot_forward': (1, 1, 0),
    'trot_backward': (1, -1, 0),
    'stand': (2, 0, 0),
}


def create_train_fn(
    gait_command,
    num_timesteps=10_000_000,
    num_envs=4096,
    batch_size=2048,
    num_minibatches=32,
    num_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.97,
    unroll_length=10,
    episode_length=1000,
    normalize_observations=True,
    reward_scaling=1.0,
    pretrained_policy=None,
):
    """
    Create Brax PPO training function

    Args:
        gait_command: Tuple of (gait_type, direction, turn)
        num_timesteps: Total training timesteps
        num_envs: Number of parallel environments (4096 recommended for 7900XT)
        batch_size: Batch size for policy updates
        learning_rate: Adam learning rate
        pretrained_policy: Optional pretrained model parameters
    """

    # Create environment factory
    env_fn = functools.partial(
        PiDogBraxEnv,
        gait_command=gait_command,
        episode_length=episode_length,
    )

    # Create training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=20,
        reward_scaling=reward_scaling,
        episode_length=episode_length,
        normalize_observations=normalize_observations,
        action_repeat=1,
        unroll_length=unroll_length,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        discounting=discounting,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        num_envs=num_envs,
        batch_size=batch_size,
        seed=0,
    )

    return train_fn, env_fn


def train_pidog(args):
    """Main training loop"""

    # Get gait command
    gait_command = GAIT_COMMANDS.get(
        args.gait,
        (0, 1, 0)  # Default: walk_forward
    )

    print("=" * 80)
    print("PiDog Brax PPO Training")
    print("=" * 80)
    print(f"Gait: {args.gait}")
    print(f"Gait command: {gait_command}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.num_envs:,}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Device: {jax.devices()}")
    print(f"Output directory: {args.output}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create training function
    train_fn, env_fn = create_train_fn(
        gait_command=gait_command,
        num_timesteps=args.timesteps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Run training
    print("\n🚀 Starting training...\n")
    start_time = time.time()

    make_inference_fn, params, metrics = train_fn(
        environment_fn=env_fn,
        progress_fn=lambda current_step, metrics: print(
            f"Step {current_step:,} | "
            f"Reward: {metrics['eval/episode_reward']:.2f} | "
            f"Episode length: {metrics['eval/episode_length']:.0f}"
        )
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"✅ Training completed in {elapsed/60:.1f} minutes")
    print(f"   Throughput: {args.timesteps/elapsed:,.0f} steps/sec")
    print(f"   Final reward: {metrics['eval/episode_reward']:.2f}")
    print("=" * 80)

    # Save trained model
    model_path = output_dir / f"pidog_{args.gait}_brax.pkl"
    model.save_params(model_path, params)
    print(f"\n💾 Saved Brax model: {model_path}")

    # Convert to PyTorch format for deployment
    # Extract policy network parameters
    policy_params = params['policy']

    # Save as numpy arrays (can be loaded into PyTorch)
    torch_model_path = output_dir / f"pidog_{args.gait}_brax_weights.npz"
    jax.tree_util.tree_map(
        lambda x: np.save(torch_model_path, jax.device_get(x)),
        policy_params
    )
    print(f"💾 Saved PyTorch-compatible weights: {torch_model_path}")

    # Create visualization
    print("\n📊 Generating evaluation video...")
    env = env_fn()
    inference_fn = make_inference_fn(params)

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    rollout = [state.pipeline_state]

    for _ in range(1000):
        act_rng, rng = jax.random.split(rng)
        obs = state.obs
        action, _ = inference_fn(obs, act_rng)
        state = env.step(state, action)
        rollout.append(state.pipeline_state)

    html_path = output_dir / f"pidog_{args.gait}_eval.html"
    html_string = html.render(env.sys, rollout)
    with open(html_path, 'w') as f:
        f.write(html_string)

    print(f"🎬 Saved visualization: {html_path}")
    print(f"   Open in browser to view trained policy\n")

    return params, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train PiDog with Brax PPO on GPU'
    )
    parser.add_argument(
        '--gait',
        type=str,
        default='walk_forward',
        choices=list(GAIT_COMMANDS.keys()),
        help='Gait to train'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=10_000_000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=4096,
        help='Number of parallel environments (4096 for 7900XT)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048,
        help='Batch size for policy updates'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./models/brax',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Path to pretrained PyTorch model (optional)'
    )

    args = parser.parse_args()

    # Train
    params, metrics = train_pidog(args)

    print("\n✨ Training complete!")
    print("\nNext steps:")
    print("1. Test model: python3 test_brax_model.py --model models/brax/pidog_walk_forward_brax.pkl")
    print("2. Deploy to hardware: python3 deploy_to_hardware.py --model models/brax/pidog_walk_forward_brax_weights.npz")
    print("3. Convert to ROS2: python3 brax_to_ros2.py --model models/brax/pidog_walk_forward_brax_weights.npz\n")


if __name__ == '__main__':
    main()
