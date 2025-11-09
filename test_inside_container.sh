#!/bin/bash
#
# Run this INSIDE the Docker container
# Tests what's blocking the RL training
#

set -e

echo "========================================="
echo "Container Diagnostic Test"
echo "========================================="
echo ""

# Make sure we're sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "Sourcing ROS2..."
    source /opt/ros/rolling/setup.bash
    source /workspace/install/setup.bash || source install/setup.bash
fi

echo "[Test 1] Check ROS2 topics while training is running..."
echo ""

# List all topics
echo "Available topics:"
ros2 topic list | head -20

echo ""
echo "[Test 2] Check message rates..."

# Check clock (should be FAST if Gazebo running at unlimited speed)
echo "Checking /clock rate (should be >1000 Hz if running fast)..."
timeout 3 ros2 topic hz /clock 2>&1 | head -5 || echo "  No /clock messages!"

echo ""
echo "Checking /joint_states rate..."
timeout 3 ros2 topic hz /joint_states 2>&1 | head -5 || echo "  No /joint_states messages!"

echo ""
echo "Checking /camera rate..."
timeout 3 ros2 topic hz /camera 2>&1 | head -5 || echo "  No /camera messages!"

echo ""
echo "[Test 3] Check if robot is spawned..."
ros2 topic echo /joint_states --once 2>&1 | head -10 || echo "  No robot spawned!"

echo ""
echo "[Test 4] Gazebo process status..."
ps aux | grep "gz sim" | grep -v grep || echo "  No Gazebo process!"

echo ""
echo "[Test 5] Python training status..."
ps aux | grep "train_rl" | grep -v grep || echo "  No training process!"

echo ""
echo "========================================="
echo "Diagnostic complete!"
echo ""
echo "If /clock shows <100 Hz, Gazebo is rate-limited"
echo "If /camera shows no messages, vision pipeline is broken"
echo "If /joint_states shows no messages, robot not spawned"
echo "========================================="
