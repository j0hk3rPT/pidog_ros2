#!/usr/bin/env python3
"""
Train PiDog quadruped locomotion using MuJoCo and PPO.

This script trains a policy for standing/walking using:
- MuJoCo physics simulation (fast, stable)
- Native IMU sensors (no crashes!)
- PPO algorithm from Stable-Baselines3
- GPU acceleration for neural network
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
import sys
import os

# Add pidog_gaits to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pidog_gaits'))

from pidog_gaits.pidog_mujoco_env import PiDogMuJoCoEnv


def make_env(rank: int = 0):
    """Create environment factory for parallel training."""
    def _init():
        env = PiDogMuJoCoEnv(render_mode=None)
        return env
    return _init


def main():
    print("=" * 70)
    print("PiDog MuJoCo RL Training with PPO")
    print("=" * 70)

    # Configuration
    num_envs = 16  # Parallel environments (fast with MuJoCo!)
    total_timesteps = 1_000_000  # 1M timesteps
    save_freq = 50_000  # Save every 50k steps
    eval_freq = 100_000  # Evaluate every 100k steps

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📊 Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Parallel environments: {num_envs}")
    print(f"   - Total timesteps: {total_timesteps:,}")
    print(f"   - Save frequency: {save_freq:,}")

    # Create vectorized environment
    print(f"\n🔧 Creating {num_envs} parallel environments...")
    if num_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_env()])

    print(f"   ✅ Environments created")

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env()])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // num_envs,  # Adjust for vectorized env
        save_path='./models/mujoco_checkpoints/',
        name_prefix='pidog_ppo',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/mujoco_best/',
        log_path='./logs/mujoco_eval/',
        eval_freq=eval_freq // num_envs,
        deterministic=True,
        render=False,
    )

    # Create PPO model
    print(f"\n🧠 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log="./logs/mujoco_tensorboard/",
    )

    print(f"   ✅ PPO model created")
    print(f"\n📝 Model Architecture:")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space.shape}")
    print(f"   - Policy: MLP (Multi-Layer Perceptron)")
    print(f"   - Learning rate: 3e-4")
    print(f"   - Batch size: 64")

    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Training will run for {total_timesteps:,} timesteps")
    print(f"   Progress will be saved to ./models/mujoco_checkpoints/")
    print(f"   TensorBoard logs: ./logs/mujoco_tensorboard/")
    print(f"\n   Monitor training with:")
    print(f"   tensorboard --logdir ./logs/mujoco_tensorboard/")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        # Save final model
        print(f"\n💾 Saving final model...")
        model.save("./models/pidog_mujoco_final")
        print(f"   ✅ Model saved to ./models/pidog_mujoco_final.zip")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        model.save("./models/pidog_mujoco_interrupted")
        print(f"   ✅ Model saved to ./models/pidog_mujoco_interrupted.zip")

    finally:
        env.close()
        eval_env.close()

    print(f"\n" + "=" * 70)
    print(f"✅ Training complete!")
    print(f"=" * 70)

    # Test trained model
    print(f"\n🎮 Testing trained model...")
    test_env = PiDogMuJoCoEnv(render_mode=None)

    obs, info = test_env.reset()
    total_reward = 0
    steps = 0

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"   Episode finished after {steps} steps")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Final height: {info['body_height']:.3f}m")
            print(f"   Upright: {info['upright']:.3f}")
            break

    test_env.close()

    print(f"\n🎯 Next Steps:")
    print(f"1. Monitor training: tensorboard --logdir ./logs/mujoco_tensorboard/")
    print(f"2. Test model: python3 test_trained_model.py")
    print(f"3. Deploy to real robot (same URDF, seamless transfer!)")


if __name__ == "__main__":
    main()
