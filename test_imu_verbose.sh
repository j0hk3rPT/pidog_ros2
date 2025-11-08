#!/bin/bash
# Test IMU with verbose Gazebo logging to diagnose issues

source install/setup.bash

echo "======================================================================"
echo "Gazebo IMU Test with VERBOSE LOGGING"
echo "======================================================================"
echo ""
echo "This test launches Gazebo with:"
echo "  - Only gz-sim-imu-system plugin (NOT sensors-system)"
echo "  - Verbose logging enabled"
echo "  - IMU sensor in URDF"
echo ""
echo "Based on: https://gazebosim.org/docs/latest/sensors/"
echo "Key insight: IMU only needs gz-sim-imu-system, NOT sensors-system"
echo ""
echo "======================================================================"
echo ""

# Set Gazebo verbosity to maximum
export GZ_VERBOSE=1
export IGN_VERBOSE=1

# Launch Gazebo with test world (IMU plugin only)
echo "Launching Gazebo with verbose logging..."
echo "Watch for plugin loading messages and any errors..."
echo ""

ros2 run ros_gz_sim create -world pidog_description/worlds/pidog_with_imu_only.sdf -v 4 &
GZ_PID=$!

echo ""
echo "Gazebo PID: $GZ_PID"
echo "Waiting 20 seconds for initialization..."
sleep 20

echo ""
echo "Checking Gazebo services..."
gz service -l | head -20

echo ""
echo "Checking if world service exists..."
if gz service -l | grep -q "/world/pidog_world/create"; then
    echo "✅ World service available!"
else
    echo "❌ World service NOT available"
fi

echo ""
echo "Checking Gazebo topics..."
gz topic -l | head -20

echo ""
echo "Stopping Gazebo..."
kill $GZ_PID
wait $GZ_PID 2>/dev/null

echo ""
echo "======================================================================"
echo "Test complete. Check output above for:"
echo "  1. Plugin loading messages"
echo "  2. Any error or warning messages"
echo "  3. Whether world service became available"
echo "======================================================================"
