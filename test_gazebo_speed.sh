#!/bin/bash
#
# Test if Gazebo is actually running at max speed
#

set -e

echo "========================================="
echo "Gazebo Speed Test"
echo "========================================="
echo ""

# Source ROS2
if [ -z "$ROS_DISTRO" ]; then
    echo "Sourcing ROS2..."
    source /opt/ros/rolling/setup.bash 2>/dev/null || true
    source install/setup.bash 2>/dev/null || true
fi

echo "[1/3] Launching Gazebo in background..."
ros2 launch pidog_description gazebo_rl_fast.launch.py > /tmp/gz_test.log 2>&1 &
GZ_PID=$!
echo "  PID: $GZ_PID"

# Wait for startup
echo ""
echo "[2/3] Waiting for Gazebo to initialize (15 seconds)..."
sleep 15

# Check if running
if ! kill -0 $GZ_PID 2>/dev/null; then
    echo "  ❌ Gazebo died! Check log:"
    cat /tmp/gz_test.log
    exit 1
fi

echo "  ✓ Gazebo still running"
echo ""

echo "[3/3] Monitoring simulation rate for 10 seconds..."
echo ""

# Monitor /clock topic rate
timeout 10 ros2 topic hz /clock 2>&1 || echo "  ⚠ No /clock messages"

echo ""
echo "Checking Gazebo CPU usage..."
GZ_CPU=$(ps aux | grep "gz sim" | grep -v grep | awk '{print $3}' | head -1)
echo "  Gazebo CPU: ${GZ_CPU}%"

if (( $(echo "$GZ_CPU < 10" | bc -l 2>/dev/null || echo "1") )); then
    echo "  ❌ Very low CPU usage - Gazebo might be stuck!"
    echo ""
    echo "Checking for errors in log:"
    tail -20 /tmp/gz_test.log
else
    echo "  ✓ Gazebo is active"
fi

echo ""
echo "========================================="
echo "Stopping Gazebo..."
kill $GZ_PID 2>/dev/null || true
sleep 2
kill -9 $GZ_PID 2>/dev/null || true
echo "Done"
