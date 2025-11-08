#!/bin/bash
# Quick start script for Phase 1 training (Conservative baseline)

echo "🚀 Starting Phase 1 Training (Conservative)"
echo "Target: Stable trot at 0.8-1.0 m/s"
echo ""

# Check if in unified container
if [ ! -d "/workspace" ]; then
    echo "⚠️  Warning: You should run this inside the unified Docker container!"
    echo "   Run: docker-compose --profile unified run --rm pidog-unified"
    echo ""
fi

# Activate ROS2 workspace
cd /home/user/pidog_ros2
source install/setup.bash

# Run training with GPU acceleration
python3 train_fast_running.py \
    --phase 1 \
    --timesteps 50000 \
    --num-envs 16 \
    --batch-size 512 \
    --device auto \
    --save-freq 10000 \
    --eval-freq 5000

echo ""
echo "✅ Phase 1 training complete!"
echo ""
echo "Next steps:"
echo "  1. Check results: tensorboard --logdir ./rl_models"
echo "  2. Test best model in Gazebo"
echo "  3. Start Phase 2: ./train_phase2.sh"
echo ""
