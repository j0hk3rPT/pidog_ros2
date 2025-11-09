#!/bin/bash
#
# Quick Test: Optimized Reward Function
#
# This runs a SHORT training session with optimized rewards
# to verify the behavior before committing to full training.
#

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Optimized Reward Function${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${GREEN}Quick test configuration:${NC}"
echo "  Timesteps: 10,000 (quick test)"
echo "  Parallel envs: 2"
echo "  Imitation model: models/best_model.pth"
echo "  Output: models/reward_test_optimized"
echo ""

echo -e "${YELLOW}Optimizations:${NC}"
echo "  ✓ Speed multiplier: 8.0 → 10.0"
echo "  ✓ Progressive speed milestones (up to +10.0)"
echo "  ✓ CRITICAL obstacle avoidance (-15.0 penalty)"
echo "  ✓ Sim-to-real penalties (jerk, energy, limits)"
echo ""

read -p "Start test? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Source ROS2
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/rolling/setup.bash 2>/dev/null || true
    source install/setup.bash 2>/dev/null || true
fi

echo ""
echo -e "${BLUE}[1/2] Launching Gazebo (headless)...${NC}"

ros2 launch pidog_description gazebo_rl_fast.launch.py \
    > /tmp/reward_test_gazebo.log 2>&1 &
GAZEBO_PID=$!

sleep 15

if ! kill -0 $GAZEBO_PID 2>/dev/null; then
    echo -e "${YELLOW}ERROR: Gazebo failed!${NC}"
    cat /tmp/reward_test_gazebo.log
    exit 1
fi

echo -e "${GREEN}✓ Gazebo running${NC}"

cleanup() {
    echo ""
    echo "Stopping Gazebo..."
    kill $GAZEBO_PID 2>/dev/null || true
    sleep 2
    kill -9 $GAZEBO_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo -e "${BLUE}[2/2] Running optimized training test...${NC}"
echo ""

# Create a temporary training script that uses optimized environment
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, 'install/pidog_gaits/lib/python3.12/site-packages')

# Monkey-patch to use optimized environment
import pidog_gaits.train_rl_vision as train_module
from pidog_gaits.pidog_rl_env_vision_optimized import PiDogVisionEnvOptimized

# Replace environment
original_env = train_module.PiDogVisionEnv
train_module.PiDogVisionEnv = PiDogVisionEnvOptimized

# Run training with test parameters
sys.argv = [
    'train_rl_vision',
    '--pretrained', './models/best_model.pth',
    '--output', './models/reward_test_optimized',
    '--timesteps', '10000',
    '--envs', '2',
    '--device', 'cuda'
]

train_module.main()
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Review results:"
echo "  tensorboard --logdir models/reward_test_optimized/tensorboard"
echo ""
echo "Compare with original rewards:"
echo "  tensorboard --logdir models/ --port 6006"
echo ""
echo "If you like the behavior, use optimized rewards in production:"
echo "  Edit train_production_pipeline.sh (see REWARD_OPTIMIZATION.md)"
echo ""
