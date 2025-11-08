#!/bin/bash

# Automated Training Data Collection Script
# Cycles through all gaits and records data for neural network training

set -e

echo "=============================================="
echo "PiDog Training Data Collection"
echo "=============================================="
echo ""

# Duration per gait (in seconds)
DURATION=${1:-20}

echo "⏱️  Recording $DURATION seconds per gait"
echo ""

# List of gaits to record
# NOTE: If certain gaits cause the robot to fall, you can:
#   1. Comment them out here temporarily
#   2. Collect them separately with manual monitoring
#   3. Adjust duration to reduce fall risk
GAITS=(
    "walk_forward"   # Usually stable
#    "walk_backward"  # Usually stable
#    "walk_left"      # May cause instability
#    "walk_right"     # May cause instability
    "trot_forward"   # Usually stable
#    "trot_backward"  # May cause instability
#    "trot_left"      # May cause instability
#    "trot_right"     # May cause instability
#    "sit"            # Static pose - stable
#    "lie"            # Static pose - stable
#    "stretch"        # Static pose - stable
)

TOTAL_GAITS=${#GAITS[@]}
TOTAL_TIME=$((TOTAL_GAITS * DURATION))

echo "📊 Will record $TOTAL_GAITS gaits"
echo "⏰ Total time: $TOTAL_TIME seconds (~$((TOTAL_TIME / 60)) minutes)"
echo ""
echo "✋ Press Ctrl+C in the data collection terminal when done to save data"
echo ""

# Wait for simulation to stabilize
echo "⏳ Waiting 5 seconds for robot to stabilize..."
sleep 5
echo "✅ Starting data collection!"
echo ""

# Cycle through each gait
for i in "${!GAITS[@]}"; do
    GAIT="${GAITS[$i]}"
    NUM=$((i + 1))

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$NUM/$TOTAL_GAITS] Recording: $GAIT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Send stand command FIRST to set joint angles
    echo "  📍 Setting stand pose..."
    ros2 topic pub /gait_command std_msgs/msg/String "data: 'stand'" --once
    sleep 1

    # Then reset robot position in Gazebo (moves body upright)
    echo "  🔄 Resetting robot position in Gazebo..."
    gz service -s /world/pidog_world/set_pose \
        --reqtype gz.msgs.Pose \
        --reptype gz.msgs.Boolean \
        --timeout 2000 \
        --req "name: 'Robot.urdf', position: {x: 0.0, y: 0.0, z: 0.12}, orientation: {x: 0, y: 0, z: 0, w: 1}" &>/dev/null || echo "  ⚠️  Gazebo reset failed, continuing..."
    sleep 2

    # Send gait command
    echo "  ▶️  Starting gait: $GAIT"
    ros2 topic pub /gait_command std_msgs/msg/String "data: '$GAIT'" --once
    sleep 1

    # Show progress bar
    for ((sec=1; sec<=DURATION; sec++)); do
        PROGRESS=$((sec * 50 / DURATION))
        REMAINING=$((50 - PROGRESS))

        # Ensure valid ranges for cut command
        if [ $PROGRESS -gt 0 ]; then
            BAR=$(printf "%-50s" "#" | sed "s/ /#/g" | cut -c1-$PROGRESS)
        else
            BAR=""
        fi

        if [ $REMAINING -gt 0 ]; then
            SPACES=$(printf "%-50s" " " | cut -c1-$REMAINING)
        else
            SPACES=""
        fi

        printf "\r[%s%s] %2d/%2ds" "$BAR" "$SPACES" "$sec" "$DURATION"
        sleep 1
    done
    echo ""
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Data collection complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANT: Press Ctrl+C in the data collection terminal to save data!"
echo ""
echo "Data will be saved to: ./training_data/"
echo ""
echo "Next steps:"
echo "  1. Stop data collection with Ctrl+C"
echo "  2. Check files: ls -lh ./training_data/"
echo "  3. Train model: ./train_gpu.sh"
echo ""
