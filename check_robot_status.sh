#!/bin/bash

# Robot Status Diagnostic Script
# Run this while collect_data_enhanced.launch.py is running

echo "=============================================="
echo "PiDog Robot Status Diagnostic"
echo "=============================================="
echo ""

echo "1. Checking if Gazebo is running..."
if pgrep -x "ruby" > /dev/null; then
    echo "   ✅ Gazebo is running"
else
    echo "   ❌ Gazebo is NOT running"
    exit 1
fi
echo ""

echo "2. Checking ros2_control controllers..."
ros2 control list_controllers 2>/dev/null || echo "   ⚠️  Cannot check controllers (is ROS2 sourced?)"
echo ""

echo "3. Checking /motor_pos topic (gait generator output)..."
timeout 2 ros2 topic echo /motor_pos --once 2>/dev/null && echo "   ✅ Gait generator is publishing" || echo "   ❌ No data on /motor_pos"
echo ""

echo "4. Checking /position_controller/commands topic..."
timeout 2 ros2 topic echo /position_controller/commands --once 2>/dev/null && echo "   ✅ Controller is receiving commands" || echo "   ❌ No data on /position_controller/commands"
echo ""

echo "5. Checking current gait..."
echo "   Listening for gait generator status..."
timeout 2 ros2 topic echo /motor_pos --once 2>/dev/null | head -5
echo ""

echo "=============================================="
echo "How to make the robot move:"
echo "=============================================="
echo ""
echo "The robot starts in 'stand' pose (static)."
echo "To make it walk, run ONE of these commands:"
echo ""
echo "  # Option 1: Automated data collection (cycles through all gaits)"
echo "  ./collect_training_data.sh 20"
echo ""
echo "  # Option 2: Manual gait commands"
echo "  ros2 topic pub /gait_command std_msgs/msg/String \"data: 'walk_forward'\" --once"
echo "  ros2 topic pub /gait_command std_msgs/msg/String \"data: 'trot_forward'\" --once"
echo "  ros2 topic pub /gait_command std_msgs/msg/String \"data: 'sit'\" --once"
echo ""
