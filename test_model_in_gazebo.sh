#!/bin/bash
#
# Test a trained RL model in Gazebo with visualization
#
# This script:
# 1. Launches Gazebo with GUI (for visualization)
# 2. Runs the trained model
# 3. Shows the robot performing the learned behavior
#

set -e

# CRITICAL: OpenBLAS performance fix
export OPENBLAS_NUM_THREADS=4

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Trained RL Model in Gazebo${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
MODEL=${1:-./models/rl_vision_fast/final_model.zip}
EPISODES=${2:-5}
GUI=${3:-yes}  # Show Gazebo GUI by default for testing

if [ ! -f "$MODEL" ]; then
    echo -e "${YELLOW}Error: Model not found: $MODEL${NC}"
    echo ""
    echo "Available models:"
    find models -name "*.zip" 2>/dev/null || echo "  No models found"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Episodes: $EPISODES"
echo "  Gazebo GUI: $GUI"
echo ""

# Check if workspace is sourced
if [ -z "$AMENT_PREFIX_PATH" ]; then
    echo -e "${YELLOW}Sourcing ROS2 workspace...${NC}"
    source install/setup.bash
fi

echo -e "${BLUE}[1/3] Launching Gazebo...${NC}"

if [ "$GUI" == "yes" ]; then
    echo "  Using NORMAL world (with GUI for visualization)"
    echo "  You can watch the robot perform the learned behavior!"
    # Use normal launch with GUI
    ros2 launch pidog_description gazebo.launch.py use_rviz:=false > /tmp/gazebo_test.log 2>&1 &
else
    echo "  Using HEADLESS mode (no GUI)"
    # Use fast headless launch
    ros2 launch pidog_description gazebo_rl_fast.launch.py > /tmp/gazebo_test.log 2>&1 &
fi

GAZEBO_PID=$!
echo "  Gazebo PID: $GAZEBO_PID"

# Wait for Gazebo to start
echo ""
echo "Waiting for Gazebo to initialize..."
sleep 15

# Check if Gazebo is still running
if ! kill -0 $GAZEBO_PID 2>/dev/null; then
    echo -e "${YELLOW}ERROR: Gazebo failed to start!${NC}"
    echo "Check log: /tmp/gazebo_test.log"
    cat /tmp/gazebo_test.log | tail -20
    exit 1
fi

echo -e "${GREEN}✓ Gazebo running${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${BLUE}[3/3] Cleaning up...${NC}"
    echo "Stopping Gazebo (PID: $GAZEBO_PID)"
    kill $GAZEBO_PID 2>/dev/null || true
    sleep 2
    kill -9 $GAZEBO_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Register cleanup
trap cleanup EXIT INT TERM

echo -e "${BLUE}[2/3] Running trained model...${NC}"
echo ""
echo "Watch the Gazebo window to see the robot in action!"
echo "(Press Ctrl+C to stop)"
echo ""

# Run the test
python3 test_rl_model.py \
    --model "$MODEL" \
    --episodes "$EPISODES"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Testing Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
