#!/bin/bash
# Test script to verify Gazebo IMU sensor is working

echo "======================================================================"
echo "Gazebo IMU Sensor Test"
echo "======================================================================"
echo ""
echo "This script will:"
echo "1. Launch Gazebo with the robot"
echo "2. Check if IMU data is being published to /imu/data"
echo "3. Display sample IMU messages"
echo ""
echo "Press Ctrl+C to stop the test"
echo "======================================================================"
echo ""

# Source the workspace
source install/setup.bash

# Launch Gazebo in background
echo "Starting Gazebo..."
ros2 launch pidog_description gazebo.launch.py use_rviz:=false &
GAZEBO_PID=$!

# Wait for Gazebo to start
echo "Waiting for Gazebo to initialize (30 seconds)..."
sleep 30

# Check if IMU topic exists
echo ""
echo "Checking for /imu/data topic..."
ros2 topic list | grep "/imu/data"

if [ $? -eq 0 ]; then
    echo "✅ /imu/data topic found!"
    echo ""
    echo "Topic info:"
    ros2 topic info /imu/data
    echo ""
    echo "Sample IMU message (first message):"
    timeout 10 ros2 topic echo /imu/data --once

    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✅ SUCCESS! IMU is publishing data!"
        echo "======================================================================"
    else
        echo ""
        echo "======================================================================"
        echo "❌ FAILED: IMU topic exists but no data received"
        echo "======================================================================"
    fi
else
    echo "❌ /imu/data topic not found"
    echo ""
    echo "Available topics:"
    ros2 topic list
    echo ""
    echo "======================================================================"
    echo "❌ FAILED: IMU topic not being published"
    echo "======================================================================"
fi

# Cleanup
echo ""
echo "Stopping Gazebo..."
kill $GAZEBO_PID
wait $GAZEBO_PID 2>/dev/null

echo "Test complete."
