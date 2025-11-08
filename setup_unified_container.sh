#!/bin/bash

# Setup Unified ROCm + ROS2 + Gazebo Container
# This combines GPU acceleration with full ROS2 simulation in one container

set -e

echo "=============================================="
echo "Building Unified ROCm + ROS2 Container"
echo "=============================================="
echo ""
echo "This will:"
echo "  1. Build Docker image with ROCm PyTorch + ROS2 Jazzy + Gazebo"
echo "  2. Install all dependencies in one container"
echo "  3. Enable GPU-accelerated RL training with Gazebo"
echo ""
echo "Time: ~10-15 minutes (one-time build)"
echo ""
read -p "Press Enter to start build..."

# Allow X11 access for Gazebo GUI
echo ""
echo "Configuring X11 access for Gazebo..."
xhost +local:docker

# Build the Docker image
echo ""
echo "Building Docker image (this takes ~10-15 min)..."
docker-compose --profile unified build pidog-unified

echo ""
echo "=============================================="
echo "✓ Build Complete!"
echo "=============================================="
echo ""
echo "To start the unified container:"
echo "  docker-compose --profile unified run --rm pidog-unified"
echo ""
echo "Inside the container you can:"
echo "  1. Build workspace:"
echo "     colcon build"
echo "     source install/setup.bash"
echo ""
echo "  2. Launch Gazebo:"
echo "     ros2 launch pidog_description gazebo.launch.py"
echo ""
echo "  3. Collect training data:"
echo "     ros2 launch pidog_gaits collect_data_enhanced.launch.py"
echo ""
echo "  4. Train imitation model (GPU):"
echo "     python3 -m pidog_gaits.pidog_gaits.train \\"
echo "       --data ./training_data/gait_data_*.npz \\"
echo "       --model simple_lstm \\"
echo "       --epochs 200 \\"
echo "       --batch_size 1024 \\"
echo "       --device cuda"
echo ""
echo "  5. Train RL model (GPU):"
echo "     python3 -m pidog_gaits.pidog_gaits.train_rl \\"
echo "       --pretrained ./models/best_model.pth \\"
echo "       --output ./models/rl \\"
echo "       --timesteps 100000 \\"
echo "       --envs 4 \\"
echo "       --device cuda"
echo ""
echo "Everything in ONE container! 🚀"
echo ""
