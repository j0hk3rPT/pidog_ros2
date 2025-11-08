#!/bin/bash

# Reinforcement Learning Training Script for PiDog
# Trains using PPO with stability rewards

set -e

echo "=============================================="
echo "PiDog Reinforcement Learning Training"
echo "=============================================="
echo ""

# Check if running in ROCm training container
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found!"
    echo ""
    echo "Please run this in the ROCm training container:"
    echo "  docker-compose --profile rocm run pidog-rocm"
    exit 1
fi

# Check if stable-baselines3 is installed
if ! python3 -c "import stable_baselines3" 2>/dev/null; then
    echo "📦 Installing Stable-Baselines3..."
    pip install -r requirements_rl.txt
fi

# Parse arguments
TIMESTEPS="${1:-100000}"
ENVS="${2:-4}"  # Default to 4 parallel envs (good for 20GB VRAM)
DEVICE="${3:-auto}"

echo "Configuration:"
echo "  Timesteps: $TIMESTEPS"
echo "  Parallel Envs: $ENVS"
echo "  Device: $DEVICE"
echo "  Pretrained Model: ./models/best_model.pth"
echo ""

# Check if pretrained model exists
if [ ! -f "./models/best_model.pth" ]; then
    echo "⚠️  Warning: No pretrained model found at ./models/best_model.pth"
    echo "   Training will start from scratch (slower)"
    echo ""
fi

echo "Starting RL training..."
echo "TIP: This will take a while. Monitor with TensorBoard:"
echo "  tensorboard --logdir ./models/rl/tensorboard"
echo ""

# Run training
python3 -m pidog_gaits.pidog_gaits.train_rl \
    --pretrained ./models/best_model.pth \
    --output ./models/rl \
    --timesteps $TIMESTEPS \
    --envs $ENVS \
    --device $DEVICE

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Model saved to: ./models/rl/final_model.zip"
echo ""
echo "Next steps:"
echo "  1. Test in Gazebo:"
echo "     ros2 launch pidog_gaits rl_demo.launch.py"
echo "  2. Monitor training:"
echo "     tensorboard --logdir ./models/rl/tensorboard"
