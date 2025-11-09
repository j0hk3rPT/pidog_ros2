#!/bin/bash
#
# Fast RL Training Script
#
# This script launches Gazebo in headless mode with optimized physics settings
# for maximum training speed, then runs RL training.
#
# Expected speedup: 10-30x faster than real-time
#

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Fast RL Training for PiDog${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Parse arguments
TIMESTEPS=${1:-10000}
ENVS=${2:-1}
OUTPUT=${3:-./models/rl_fast}
PRETRAINED=${4:-./models/best_model.pth}

echo -e "${GREEN}Configuration:${NC}"
echo "  Timesteps: $TIMESTEPS"
echo "  Parallel envs: $ENVS"
echo "  Output: $OUTPUT"
echo "  Pretrained: $PRETRAINED"
echo ""

# Check if workspace is sourced
if [ -z "$AMENT_PREFIX_PATH" ]; then
    echo -e "${YELLOW}WARNING: ROS2 workspace not sourced!${NC}"
    echo "Running: source install/setup.bash"
    source install/setup.bash
fi

echo -e "${BLUE}[1/3] Launching Gazebo (headless, fast mode)...${NC}"
echo "  - No GUI rendering"
echo "  - Unlimited real-time factor"
echo "  - Reduced physics solver iterations (50 vs 300)"
echo "  - 5ms physics steps (200 Hz vs 1000 Hz)"
echo ""

# Launch Gazebo in background
ros2 launch pidog_description gazebo_rl_fast.launch.py > /tmp/gazebo_rl.log 2>&1 &
GAZEBO_PID=$!

# Wait for Gazebo to start
echo "Waiting for Gazebo to initialize..."
sleep 10

# Check if Gazebo is running
if ! kill -0 $GAZEBO_PID 2>/dev/null; then
    echo -e "${YELLOW}ERROR: Gazebo failed to start!${NC}"
    echo "Check log: /tmp/gazebo_rl.log"
    cat /tmp/gazebo_rl.log
    exit 1
fi

echo -e "${GREEN}✓ Gazebo running (PID: $GAZEBO_PID)${NC}"
echo ""

echo -e "${BLUE}[2/3] Starting RL training...${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${BLUE}[3/3] Cleaning up...${NC}"
    echo "Stopping Gazebo (PID: $GAZEBO_PID)"
    kill $GAZEBO_PID 2>/dev/null || true
    sleep 2
    # Force kill if still running
    kill -9 $GAZEBO_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Register cleanup on exit
trap cleanup EXIT INT TERM

# Run RL training
python3 -m pidog_gaits.train_rl \
    --pretrained "$PRETRAINED" \
    --output "$OUTPUT" \
    --timesteps "$TIMESTEPS" \
    --envs "$ENVS" \
    --device cuda

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Model saved to: $OUTPUT"
echo ""