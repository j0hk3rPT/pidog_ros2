#!/bin/bash

# Quick Test Data Collection
# Collects minimal data (10 seconds) to test the pipeline

set -e

echo "=============================================="
echo "Quick Test Data Collection"
echo "=============================================="
echo ""
echo "This will collect 10 seconds per gait"
echo "Just enough to test the training pipeline"
echo ""
echo "For production, use: ./collect_training_data.sh 90"
echo ""

# Check if we're in the right environment
if [ ! -d "./src/pidog_gaits" ]; then
    echo "❌ Error: Not in workspace root"
    echo "Please run from /home/user/pidog_ros2"
    exit 1
fi

# Check if workspace is sourced
if [ -z "$AMENT_PREFIX_PATH" ]; then
    echo "❌ Error: Workspace not sourced"
    echo "Please run: source install/setup.bash"
    exit 1
fi

# Run data collection (10 seconds per gait)
./collect_training_data.sh 10

echo ""
echo "=============================================="
echo "✓ Test data collection complete!"
echo "=============================================="
echo ""
echo "Next steps (in ROCm training container):"
echo "  1. docker-compose --profile rocm run pidog-rocm"
echo "  2. ./test_rl_quick.sh"
echo ""
