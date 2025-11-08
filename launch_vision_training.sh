#!/bin/bash
# Launch Gazebo + Vision-Based RL Training
# This script launches Gazebo GUI in the background, then starts RL training

set -e

echo "======================================================"
echo "PiDog Vision-Based RL Training with Gazebo GUI"
echo "======================================================"

# Parse arguments
TIMESTEPS=10000
ENVS=1
OUTPUT="./models/rl_vision"
PRETRAINED=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --timesteps)
      TIMESTEPS="$2"
      shift 2
      ;;
    --envs)
      ENVS="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --pretrained)
      PRETRAINED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--timesteps N] [--envs N] [--output DIR] [--pretrained MODEL]"
      exit 1
      ;;
  esac
done

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "[1/3] Sourcing ROS2 environment..."
    source install/setup.bash
else
    echo "[1/3] ROS2 already sourced ($ROS_DISTRO)"
fi

# Launch Gazebo in background (GUI mode)
echo "[2/3] Launching Gazebo GUI..."
ros2 launch pidog_description gazebo.launch.py &
GAZEBO_PID=$!

# Wait for Gazebo to start
echo "Waiting for Gazebo to initialize (10 seconds)..."
sleep 10

# Check if Gazebo is running
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "ERROR: Gazebo failed to start!"
    exit 1
fi

echo "✓ Gazebo is running (PID: $GAZEBO_PID)"

# Build training command
TRAIN_CMD="python3 -m pidog_gaits.train_rl_vision --output $OUTPUT --timesteps $TIMESTEPS --envs $ENVS --device cuda --no-headless"
if [ -n "$PRETRAINED" ]; then
    TRAIN_CMD="$TRAIN_CMD --pretrained $PRETRAINED"
fi

# Start training
echo "[3/3] Starting vision-based RL training..."
echo "Command: $TRAIN_CMD"
echo ""

# Trap to kill Gazebo on exit
cleanup() {
    echo ""
    echo "Shutting down Gazebo..."
    kill $GAZEBO_PID 2>/dev/null || true
    wait $GAZEBO_PID 2>/dev/null || true
    echo "Done!"
}
trap cleanup EXIT INT TERM

# Run training (this will block until training finishes)
$TRAIN_CMD

# Training complete - cleanup will run automatically
