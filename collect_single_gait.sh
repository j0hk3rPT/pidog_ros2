#!/bin/bash

# Collect data for a single gait
# Usage: ./collect_single_gait.sh <gait_name> [duration]
#
# Example: ./collect_single_gait.sh walk_left 15

set -e

GAIT_NAME=${1:-walk_forward}
DURATION=${2:-20}

echo "=============================================="
echo "PiDog Single Gait Data Collection"
echo "=============================================="
echo ""
echo "🎯 Gait: $GAIT_NAME"
echo "⏱️  Duration: $DURATION seconds"
echo ""

# Wait for stabilization
echo "⏳ Waiting 5 seconds for robot to stabilize..."
sleep 5
echo "✅ Starting!"
echo ""

# Reset robot pose in Gazebo
echo "🔄 Resetting robot pose in Gazebo..."
gz service -s /world/pidog/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req 'reset: {all: true}' &>/dev/null || echo "⚠️  Gazebo reset failed, continuing..."
sleep 2

# Reset to stand first
echo "📍 Setting stand pose..."
ros2 topic pub /gait_command std_msgs/msg/String "data: 'stand'" --once
sleep 2

# Start target gait
echo "▶️  Starting gait: $GAIT_NAME"
ros2 topic pub /gait_command std_msgs/msg/String "data: '$GAIT_NAME'" --once
sleep 1

# Show progress
echo ""
for ((sec=1; sec<=DURATION; sec++)); do
    PROGRESS=$((sec * 50 / DURATION))
    REMAINING=$((50 - PROGRESS))

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

# Return to stand
echo "🔄 Returning to stand pose..."
ros2 topic pub /gait_command std_msgs/msg/String "data: 'stand'" --once

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Collection complete for: $GAIT_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANT: Press Ctrl+C in the data collection terminal to save data!"
echo ""
