"""
Vision-Based Reinforcement Learning Training for PiDog

Uses CNN policy with multi-modal observations (camera + proprioception)
to maximize GPU utilization and learn robust, dog-like running behaviors.
"""

import argparse
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn as nn

from .pidog_rl_env_vision import PiDogVisionEnv


class MultiModalFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-modal observations.

    Combines:
    - CNN for image processing
    - MLP for vector observations (proprioception)

    This architecture maximizes GPU utilization!
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # CNN for image (84x84x3)
        n_input_channels = observation_space['image'].shape[2]  # 3 (RGB)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_image = torch.as_tensor(observation_space['image'].sample()[None]).float()
            sample_image = sample_image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            n_flatten = self.cnn(sample_image).shape[1]

        # MLP for vector observations (ALL sensors: proprioception + ultrasonic + touch)
        vector_dim = observation_space['vector'].shape[0]  # 44D (was 42D)
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Combine both streams
        combined_dim = n_flatten + 128
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        """
        Forward pass through multi-modal architecture.
        """
        # Process image through CNN
        # observations['image'] is (B, H, W, C), need (B, C, H, W)
        image = observations['image'].float() / 255.0  # Normalize to [0, 1]
        image = image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        image_features = self.cnn(image)

        # Process vector through MLP
        vector_features = self.vector_mlp(observations['vector'])

        # Combine both feature streams
        combined = torch.cat([image_features, vector_features], dim=1)
        features = self.combiner(combined)

        return features


def make_env(rank=0):
    """
    Create a wrapped vision environment for training.
    """
    def _init():
        env = PiDogVisionEnv(node_name=f'pidog_vision_env_{rank}')
        env = Monitor(env)
        return env

    return _init


def train_vision_rl(
    pretrained_model=None,
    output_dir='./models/rl_vision',
    total_timesteps=100000,
    n_envs=1,
    learning_rate=3e-4,
    device='cuda'
):
    """
    Train PiDog with vision-based RL using CNN policy.

    This fully utilizes GPU for both CNN processing and policy updates!
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PiDog Vision-Based RL Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("=" * 60)
    print("Multi-Modal Observations:")
    print("  - Vision: 84x84 RGB camera (CNN processing)")
    print("  - Proprioception: 42D vector (IMU + joints + gait)")
    print("=" * 60)

    # Create environment(s)
    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Custom policy kwargs with our multi-modal feature extractor
    policy_kwargs = dict(
        features_extractor_class=MultiModalFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Larger networks for vision
    )

    # Create PPO model with MultiInputPolicy (for Dict observations)
    model = PPO(
        'MultiInputPolicy',  # Use MultiInputPolicy for Dict observations
        env,
        learning_rate=learning_rate,
        n_steps=2048,        # Steps per environment per update
        batch_size=64,       # Minibatch size (GPU batching!)
        n_epochs=10,         # Epochs per update
        gamma=0.99,          # Discount factor
        gae_lambda=0.95,     # GAE parameter
        clip_range=0.2,      # PPO clip range
        ent_coef=0.01,       # Entropy coefficient
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Gradient clipping
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(output_dir, 'tensorboard')
    )

    print("\n" + "=" * 60)
    print("GPU Utilization Maximized!")
    print("=" * 60)
    print("Architecture:")
    print("  Image (84x84x3) → CNN (3 conv layers) → 3136D")
    print("  Vector (42D) → MLP (2 layers) → 128D")
    print("  Combined → 256D features → Policy + Value heads")
    print("=" * 60)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(output_dir, 'checkpoints'),
        name_prefix='rl_vision_model'
    )

    # Training
    print("\nStarting vision-based RL training...")
    print("TIP: Monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {os.path.join(output_dir, 'tensorboard')}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    final_path = os.path.join(output_dir, 'final_model.zip')
    model.save(final_path)
    print(f"\n✓ Training complete! Model saved to: {final_path}")

    # Save as PyTorch checkpoint
    torch_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'policy_state_dict': model.policy.state_dict(),
    }, torch_path)
    print(f"✓ PyTorch checkpoint saved to: {torch_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train PiDog with Vision-Based RL')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--output', type=str, default='./models/rl_vision',
                        help='Output directory for RL models')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--envs', type=int, default=1,
                        help='Number of parallel environments')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training (default: cuda)')

    args = parser.parse_args()

    # Check GPU availability
    device = args.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available! Check your GPU setup.")
        print("\n[INFO] Using GPU for vision-based RL training")
        print("[INFO] GPU will be fully utilized for CNN processing!")
    else:
        print("\n[INFO] Using CPU for training")

    # Train
    model = train_vision_rl(
        pretrained_model=args.pretrained,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        learning_rate=args.lr,
        device=device
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the vision model in Gazebo")
    print("2. Deploy to hardware with camera")
    print("3. Watch your PiDog run like a real dog!")


if __name__ == '__main__':
    main()
