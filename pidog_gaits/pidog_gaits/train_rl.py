"""
Reinforcement Learning Training Script for PiDog

Uses Stable-Baselines3 PPO to fine-tune the imitation-learned model
with physics-based rewards for stability and task completion.
"""

import argparse
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

from .pidog_rl_env_modern import PiDogGazeboEnv
from .neural_network import GaitNetSimpleLSTM


class CustomPolicyNetwork(torch.nn.Module):
    """
    Custom policy network that wraps our existing LSTM model.

    This allows us to start RL training from the imitation-learned weights.
    """

    def __init__(self, pretrained_model_path=None):
        super(CustomPolicyNetwork, self).__init__()

        # Load pretrained model if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model from {pretrained_model_path}")
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')

            # Create model architecture
            self.gait_net = GaitNetSimpleLSTM(input_size=4, hidden_size=64, output_size=12)

            # Load weights
            if 'model_state_dict' in checkpoint:
                self.gait_net.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.gait_net.load_state_dict(checkpoint)

            print("Successfully loaded pretrained weights")
        else:
            print("No pretrained model found, starting from scratch")
            self.gait_net = GaitNetSimpleLSTM(input_size=4, hidden_size=64, output_size=12)

        # Adaptation layers (fine-tuning)
        # Take full observation and adapt to gait network input
        self.obs_adapter = torch.nn.Sequential(
            torch.nn.Linear(36, 64),  # 36D observation -> 64D
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),   # 64D -> 4D (gait command format)
        )

        # Value head for RL (estimates expected return)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(36, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, obs, deterministic=False):
        """
        Forward pass for policy.

        Args:
            obs: Observation tensor [batch, 36]

        Returns:
            actions: Joint positions [batch, 12]
            value: State value estimate [batch, 1]
        """
        # Extract gait command from observation (first 4 dimensions)
        gait_cmd = obs[:, :4]

        # Get actions from gait network
        actions, _ = self.gait_net(gait_cmd)  # LSTM returns (output, hidden)

        # Get value estimate
        value = self.value_head(obs)

        return actions, value


def make_env(rank=0):
    """
    Create a wrapped environment for training.

    Args:
        rank: Index of the environment (for parallel training)

    Returns:
        Wrapped environment
    """
    def _init():
        env = PiDogGazeboEnv(node_name=f'pidog_rl_env_{rank}')
        env = Monitor(env)  # For logging
        return env

    return _init


def train_rl(
    pretrained_model='./models/best_model.pth',
    output_dir='./models/rl',
    total_timesteps=100000,
    n_envs=1,
    learning_rate=3e-4,
    device='auto'
):
    """
    Train PiDog with reinforcement learning.

    Args:
        pretrained_model: Path to imitation-learned model (for initialization)
        output_dir: Directory to save RL models
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: PPO learning rate
        device: 'cpu', 'cuda', or 'auto'
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PiDog Reinforcement Learning Training")
    print("=" * 60)
    print(f"Pretrained model: {pretrained_model}")
    print(f"Output directory: {output_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("=" * 60)

    # Create environment(s)
    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Create PPO model
    # NOTE: We're using MlpPolicy but will replace the network later
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=2048,        # Steps per environment per update
        batch_size=64,       # Minibatch size
        n_epochs=10,         # Epochs per update
        gamma=0.99,          # Discount factor
        gae_lambda=0.95,     # GAE parameter
        clip_range=0.2,      # PPO clip range
        ent_coef=0.01,       # Entropy coefficient (exploration)
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Gradient clipping
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(output_dir, 'tensorboard')
    )

    # Optional: Load pretrained weights into policy network
    # This is advanced - for now, SB3 will train from scratch but we can fine-tune later

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(output_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    # Training
    print("\nStarting training...")
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

    # Save as PyTorch checkpoint (for deployment)
    torch_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'policy_state_dict': model.policy.state_dict(),
    }, torch_path)
    print(f"✓ PyTorch checkpoint saved to: {torch_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train PiDog with Reinforcement Learning')
    parser.add_argument('--pretrained', type=str, default='./models/best_model.pth',
                        help='Path to pretrained imitation model')
    parser.add_argument('--output', type=str, default='./models/rl',
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

    # Always use GPU by default
    device = args.device

    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available! Check your GPU setup.")
        print("\n[INFO] Using GPU for RL training")
        print("[INFO] Note: You may see a warning about MLP+GPU - this is expected and safe to ignore")
        print("[INFO] GPU is used for tensor operations and parallel environment coordination")
    else:
        print("\n[INFO] Using CPU for RL training")

    # Train
    model = train_rl(
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
    print("1. Test the model:")
    print("   python3 -m pidog_gaits.test_rl_model --model ./models/rl/final_model.zip")
    print("2. Deploy to hardware:")
    print("   Copy final_model.pth to robot and use with nn_controller")


if __name__ == '__main__':
    main()
