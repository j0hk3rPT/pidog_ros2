#!/usr/bin/env python3
"""
Train PiDog for Fast Running using Reinforcement Learning

Pure RL approach - no demonstrations needed!
Directly optimizes for maximum forward speed.

Hardware: GPU (CUDA/ROCm) recommended
GPU Memory: ~4-8GB for 16 parallel environments
Training Time: ~20-45 minutes depending on phase
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add ROS2 workspace to path
sys.path.append('/home/user/pidog_ros2/install/pidog_gaits/lib/python3.10/site-packages')

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ROCm available: {torch.version.hip is not None if hasattr(torch.version, 'hip') else False}")

# Import RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# Import our hardware-compatible environment
from pidog_gaits.pidog_rl_env_hardware import PiDogHardwareEnv


def make_env(rank, reward_mode='conservative', seed=0):
    """
    Create a single environment instance.

    Args:
        rank: Environment ID for parallel training
        reward_mode: 'conservative' (Phase 1) or 'fast_running' (Phase 2)
        seed: Random seed
    """
    def _init():
        env = PiDogHardwareEnv(
            node_name=f'pidog_rl_env_{rank}',
            reward_mode=reward_mode
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train PiDog for fast running')

    # Training phase
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase: 1=conservative baseline, 2=fast running')

    # Environment settings
    parser.add_argument('--num-envs', type=int, default=16,
                        help='Number of parallel environments (use GPU VRAM efficiently)')

    # Training duration
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (default: 50K for phase 1, 100K for phase 2)')

    # Model settings
    parser.add_argument('--policy', type=str, default='MlpPolicy',
                        choices=['MlpPolicy', 'CnnPolicy'],
                        help='Policy network type')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Minibatch size (larger = better GPU usage)')
    parser.add_argument('--n-steps', type=int, default=512,
                        help='Steps per environment before update')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs for PPO update')

    # GPU settings
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                        help='Device to use (auto detects GPU)')

    # Checkpointing
    parser.add_argument('--save-freq', type=int, default=10000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval-freq', type=int, default=5000,
                        help='Evaluate every N steps')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of episodes for evaluation')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Output
    parser.add_argument('--output-dir', type=str, default='./rl_models',
                        help='Directory to save models')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')

    args = parser.parse_args()

    # Set defaults based on phase
    if args.timesteps is None:
        args.timesteps = 50000 if args.phase == 1 else 100000

    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'phase{args.phase}_{timestamp}'

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    tensorboard_log = os.path.join(run_dir, 'tensorboard')
    os.makedirs(tensorboard_log, exist_ok=True)

    print("=" * 80)
    print(f"🚀 PiDog Fast Running Training - Phase {args.phase}")
    print("=" * 80)
    print(f"Reward mode: {'conservative' if args.phase == 1 else 'fast_running'}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output directory: {run_dir}")
    print("=" * 80)
    print()

    # Determine reward mode
    reward_mode = 'conservative' if args.phase == 1 else 'fast_running'

    # Create vectorized environments for parallel training
    print(f"Creating {args.num_envs} parallel environments...")
    env = SubprocVecEnv([
        make_env(rank=i, reward_mode=reward_mode, seed=42)
        for i in range(args.num_envs)
    ])

    # Wrap with monitor for logging
    env = VecMonitor(env)

    print("✅ Environments created!")
    print()

    # Create evaluation environment (single, for consistent metrics)
    print("Creating evaluation environment...")
    eval_env = PiDogHardwareEnv(
        node_name='pidog_eval_env',
        reward_mode=reward_mode
    )
    print("✅ Evaluation environment created!")
    print()

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,  # Adjust for parallel envs
        save_path=os.path.join(run_dir, 'checkpoints'),
        name_prefix='pidog_rl',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, 'best_model'),
        log_path=os.path.join(run_dir, 'eval_logs'),
        eval_freq=args.eval_freq // args.num_envs,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # Create or load model
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
        print("✅ Model loaded!")
    else:
        print("🆕 Creating new model...")

        # PPO hyperparameters optimized for robotics
        model = PPO(
            args.policy,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,              # Discount factor
            gae_lambda=0.95,         # GAE parameter
            clip_range=0.2,          # PPO clip range
            clip_range_vf=None,      # No value function clipping
            normalize_advantage=True,
            ent_coef=0.01,           # Entropy coefficient (exploration)
            vf_coef=0.5,             # Value function coefficient
            max_grad_norm=0.5,       # Gradient clipping
            use_sde=False,           # State-dependent exploration
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=tensorboard_log,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Larger network for 48D obs
                activation_fn=torch.nn.ReLU,
            ),
            verbose=1,
            device=args.device,
        )

        print("✅ Model created!")

    print()
    print("=" * 80)
    print("🎮 Starting Training!")
    print("=" * 80)

    if args.phase == 1:
        print("Phase 1 Goals:")
        print("  - Learn stable trotting gait")
        print("  - Avoid stalling (> 0.05 m/s)")
        print("  - Stay upright (roll < 45°)")
        print("  - Target speed: 0.8-1.0 m/s")
        print()
    else:
        print("Phase 2 Goals:")
        print("  - Maximize forward speed")
        print("  - Achieve 1.5-2.0 m/s")
        print("  - Use flight phase (galloping)")
        print("  - Maintain stability")
        print()

    print("📊 Monitoring:")
    print(f"  - TensorBoard: tensorboard --logdir {tensorboard_log}")
    print(f"  - Checkpoints: {os.path.join(run_dir, 'checkpoints')}")
    print(f"  - Best model: {os.path.join(run_dir, 'best_model')}")
    print()
    print("⏱️  Estimated time:")
    print(f"  - Phase 1 (50K steps): ~20-30 minutes")
    print(f"  - Phase 2 (100K steps): ~40-60 minutes")
    print()
    print("=" * 80)
    print()

    # Train!
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=args.run_name,
            reset_num_timesteps=(args.resume is None),
            progress_bar=True,
        )

        print()
        print("=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)

        # Save final model
        final_model_path = os.path.join(run_dir, 'final_model')
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")

        # Save as .pth for deployment
        final_pth_path = os.path.join(run_dir, 'final_model.pth')
        torch.save(model.policy.state_dict(), final_pth_path)
        print(f"💾 PyTorch weights saved to: {final_pth_path}")

        print()
        print("📊 Next steps:")
        print("  1. Check TensorBoard for training curves")
        print("  2. Evaluate best model in Gazebo")
        print("  3. If Phase 1: Train Phase 2 for higher speed")
        print("  4. If Phase 2: Deploy to real hardware!")
        print()

    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user")
        print("💾 Saving current model...")

        interrupted_path = os.path.join(run_dir, 'interrupted_model')
        model.save(interrupted_path)
        print(f"✅ Model saved to: {interrupted_path}")
        print()

    finally:
        # Clean up
        env.close()
        eval_env.close()
        print("🧹 Cleaned up environments")


if __name__ == '__main__':
    main()
