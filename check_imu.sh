#!/bin/bash
# Quick IMU test - checks if IMU data is being published

source install/setup.bash

echo "======================================================================"
echo "Quick IMU Data Check"
echo "======================================================================"
echo ""
echo "Checking if /imu/data topic exists and has data..."
echo ""

# Check if topic exists
if ros2 topic list 2>/dev/null | grep -q "/imu/data"; then
    echo "✅ /imu/data topic found!"
    echo ""

    echo "Attempting to read one IMU message (30 second timeout)..."
    if timeout 30 ros2 topic echo /imu/data --once 2>/dev/null; then
        echo ""
        echo "======================================================================"
        echo "✅ SUCCESS! IMU is publishing data!"
        echo "======================================================================"
        exit 0
    else
        echo ""
        echo "======================================================================"
        echo "❌ FAILED: Topic exists but no data received within 30 seconds"
        echo "======================================================================"
        echo ""
        echo "Possible issues:"
        echo "1. Robot not spawned in Gazebo yet (check Gazebo GUI)"
        echo "2. IMU sensor not initialized"
        echo "3. Sensor plugin configuration issue"
        echo ""
        echo "Debug commands:"
        echo "  gz topic -l | grep imu    # Check Gazebo IMU topic"
        echo "  gz topic -e -t /imu       # Echo Gazebo IMU data"
        exit 1
    fi
else
    echo "❌ /imu/data topic not found"
    echo ""
    echo "Available topics:"
    ros2 topic list 2>/dev/null
    echo ""
    echo "======================================================================"
    echo "❌ FAILED: IMU bridge not running"
    echo "======================================================================"
    exit 1
fi
