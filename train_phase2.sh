#!/bin/bash
# Quick start script for Phase 2 training (Fast running)

echo "🚀 Starting Phase 2 Training (Fast Running)"
echo "Target: High speed at 1.5-2.0 m/s with galloping"
echo ""

# Check for Phase 1 checkpoint
PHASE1_BEST="./rl_models/phase1_*/best_model/best_model.zip"
if [ ! -f $PHASE1_BEST ]; then
    echo "⚠️  No Phase 1 model found!"
    echo "   You should train Phase 1 first: ./train_phase1.sh"
    echo "   Or train Phase 2 from scratch (not recommended)"
    echo ""
    read -p "Train Phase 2 from scratch? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    RESUME_ARG=""
else
    echo "✅ Found Phase 1 checkpoint: $PHASE1_BEST"
    echo "   Will continue training from Phase 1"
    echo ""
    RESUME_ARG="--resume $PHASE1_BEST"
fi

# Activate ROS2 workspace
cd /home/user/pidog_ros2
source install/setup.bash

# Run training with higher speed priority
python3 train_fast_running.py \
    --phase 2 \
    --timesteps 100000 \
    --num-envs 16 \
    --batch-size 512 \
    --device auto \
    --save-freq 10000 \
    --eval-freq 5000 \
    $RESUME_ARG

echo ""
echo "✅ Phase 2 training complete!"
echo ""
echo "Next steps:"
echo "  1. Check results: tensorboard --logdir ./rl_models"
echo "  2. Test best model in Gazebo"
echo "  3. Deploy to real PiDog hardware!"
echo ""
