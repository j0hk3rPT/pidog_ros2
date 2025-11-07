#!/bin/bash
# Debug script to monitor topic data flow

echo "=== Checking ROS2 topics ==="
echo ""

echo "1. Available topics:"
ros2 topic list
echo ""

echo "2. /motor_pos (gait generator output) - 2 messages:"
timeout 2 ros2 topic echo /motor_pos --once 2>/dev/null || echo "No data on /motor_pos"
echo ""

echo "3. /position_controller/commands (controller input) - 2 messages:"
timeout 2 ros2 topic echo /position_controller/commands --once 2>/dev/null || echo "No data on /position_controller/commands"
echo ""

echo "4. /joint_states (actual joint positions from Gazebo) - 1 message:"
timeout 2 ros2 topic echo /joint_states --once 2>/dev/null || echo "No data on /joint_states"
echo ""

echo "5. Topic frequencies:"
echo "/motor_pos:"
timeout 3 ros2 topic hz /motor_pos 2>/dev/null || echo "No data"
echo "/position_controller/commands:"
timeout 3 ros2 topic hz /position_controller/commands 2>/dev/null || echo "No data"
echo ""

echo "6. Active nodes:"
ros2 node list
