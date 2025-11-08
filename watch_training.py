#!/usr/bin/env python3
"""
Watch PiDog learn in real-time with MuJoCo viewer.
This runs a single environment with visualization so you can see the robot learning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pidog_gaits'))

from pidog_gaits.pidog_mujoco_env import PiDogMuJoCoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
import time

class RenderCallback(BaseCallback):
    """Callback to render environment during training."""

    def __init__(self, render_env, render_freq=100, verbose=0):
        super().__init__(verbose)
        self.render_env = render_env
        self.render_freq = render_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Render periodically
        if self.n_calls % self.render_freq == 0:
            self.render_env.render()
            time.sleep(0.001)  # Small delay for viewer

        return True

    def _on_rollout_end(self) -> None:
        # Print episode stats
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

                if len(self.episode_rewards) > 0:
                    mean_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"reward={info['episode']['r']:.2f}, "
                          f"length={info['episode']['l']}, "
                          f"mean_reward_10={mean_reward:.2f}")


def main():
    print("=" * 70)
    print("Watch PiDog Learn with MuJoCo Viewer")
    print("=" * 70)

    # Check DISPLAY
    display = os.environ.get('DISPLAY')
    if not display:
        print("\n⚠️  WARNING: DISPLAY not set. Viewer might not work.")
        print("   To fix: export DISPLAY=:0")
        input("\nPress Enter to continue anyway, or Ctrl+C to exit...")

    # Configuration
    total_timesteps = 100_000  # Quick training for visualization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n📊 Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Timesteps: {total_timesteps:,}")
    print(f"   - Render mode: Interactive viewer")

    # Create environment with rendering
    print(f"\n🔧 Creating environment with viewer...")
    try:
        env = PiDogMuJoCoEnv(render_mode="human")
        print(f"   ✅ Environment created with viewer")
    except Exception as e:
        print(f"   ❌ Failed to create viewer: {e}")
        print(f"\n   Falling back to headless mode...")
        env = PiDogMuJoCoEnv(render_mode=None)
        render_available = False
    else:
        render_available = True

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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log="./logs/mujoco_watch/",
    )
    print(f"   ✅ Model created")

    # Create render callback
    if render_available:
        callback = RenderCallback(env, render_freq=10)
        print(f"\n🎮 Viewer Controls:")
        print(f"   - Left mouse drag: Rotate view")
        print(f"   - Right mouse drag: Pan view")
        print(f"   - Scroll wheel: Zoom")
        print(f"   - Viewer will update during training")
    else:
        callback = None

    # Train with visualization
    print(f"\n🚀 Starting training with visualization...")
    print(f"   Watch the robot learn in the viewer window!")
    print(f"   Press Ctrl+C to stop early\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

        print(f"\n✅ Training complete!")

        # Test learned policy
        print(f"\n🎮 Testing learned policy...")
        obs, info = env.reset()
        total_reward = 0

        for i in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if render_available:
                env.render()
                time.sleep(0.02)  # Slow down for viewing

            if terminated or truncated:
                print(f"   Episode finished after {i+1} steps")
                print(f"   Total reward: {total_reward:.2f}")
                print(f"   Final height: {info['body_height']:.3f}m")
                break

        # Save model
        model.save("./models/pidog_watched")
        print(f"\n💾 Model saved to ./models/pidog_watched.zip")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        model.save("./models/pidog_watched_interrupted")
        print(f"   💾 Model saved to ./models/pidog_watched_interrupted.zip")

    finally:
        env.close()

    print(f"\n" + "=" * 70)
    print(f"✅ Done!")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
