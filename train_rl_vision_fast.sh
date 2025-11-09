#!/bin/bash
#
# Fast Vision-Based RL Training Script
#
# Trains with FULL sensor suite: Camera + Ultrasonic + IMU + Joint Encoders
# Optimized for maximum speed with headless Gazebo and fast physics.
#
# Expected GPU usage: High (CNN processing + Gazebo rendering)
# Expected CPU usage: High (physics simulation + multiple environments)
#

set -e

# CRITICAL: OpenBLAS performance fix for Gazebo
export OPENBLAS_NUM_THREADS=4

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fast Vision-Based RL Training for PiDog${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
TIMESTEPS=${1:-10000}
ENVS=${2:-1}
OUTPUT=${3:-./models/rl_vision_fast}
PRETRAINED=${4:-./models/best_model.pth}

echo -e "${GREEN}Configuration:${NC}"
echo "  Timesteps: $TIMESTEPS"
echo "  Parallel envs: $ENVS"
echo "  Output: $OUTPUT"
echo "  Pretrained: $PRETRAINED"
echo ""
echo -e "${GREEN}Sensors Active:${NC}"
echo "  - Camera: 84x84x3 RGB vision"
echo "  - Ultrasonic: Distance measurement (HC-SR04)"
echo "  - IMU: Orientation + angular velocity"
echo "  - Joint Encoders: 12 joint positions + velocities"
echo ""

# Check if workspace is sourced
if [ -z "$AMENT_PREFIX_PATH" ]; then
    echo -e "${YELLOW}WARNING: ROS2 workspace not sourced!${NC}"
    echo "Running: source install/setup.bash"
    source install/setup.bash
fi

echo -e "${BLUE}[1/3] Launching Gazebo (headless, fast mode + all sensors)...${NC}"
echo "  - No GUI rendering (headless)"
echo "  - Unlimited real-time factor (run as fast as CPU allows)"
echo "  - Reduced physics solver (50 iters vs 300)"
echo "  - 5ms physics steps (200 Hz vs 1000 Hz)"
echo "  - Camera rendering ON (for vision)"
echo "  - Ultrasonic sensor ON"
echo ""

# Launch Gazebo in background
ros2 launch pidog_description gazebo_rl_fast.launch.py > /tmp/gazebo_rl_vision.log 2>&1 &
GAZEBO_PID=$!

# Wait for Gazebo to start
echo "Waiting for Gazebo + sensors to initialize..."
sleep 15  # Vision sensors need more time

# Check if Gazebo is running
if ! kill -0 $GAZEBO_PID 2>/dev/null; then
    echo -e "${YELLOW}ERROR: Gazebo failed to start!${NC}"
    echo "Check log: /tmp/gazebo_rl_vision.log"
    cat /tmp/gazebo_rl_vision.log
    exit 1
fi

echo -e "${GREEN}✓ Gazebo running with all sensors (PID: $GAZEBO_PID)${NC}"
echo ""

echo -e "${BLUE}[2/3] Starting Vision-Based RL training...${NC}"
echo "  - Multi-modal CNN policy (image + vector)"
echo "  - GPU-accelerated CNN feature extraction"
echo "  - Enhanced rewards: speed + stability + obstacle avoidance"
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

# Run Vision-Based RL training
python3 -m pidog_gaits.train_rl_vision \
    --pretrained "$PRETRAINED" \
    --output "$OUTPUT" \
    --timesteps "$TIMESTEPS" \
    --envs "$ENVS" \
    --device cuda

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Vision Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Model saved to: $OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Monitor training: tensorboard --logdir $OUTPUT/tensorboard"
echo "  2. Evaluate model: python3 -m pidog_gaits.test_rl_vision --model $OUTPUT/final_model.zip"
echo ""
