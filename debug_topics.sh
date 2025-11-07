#!/bin/bash
# Debug script to monitor topic data flow and diagnose left/right asymmetry

echo "=============================================="
echo "PiDog ROS2 Debug Information"
echo "=============================================="
echo ""

echo "1. Available topics:"
ros2 topic list
echo ""

echo "2. Active nodes:"
ros2 node list
echo ""

echo "3. Controller status:"
ros2 control list_controllers 2>/dev/null || echo "Controller manager not available"
echo ""

echo "4. /motor_pos (gait generator output) - 3 samples:"
echo "   Order: [BR, FR, BL, FL] (2 angles each)"
echo "   Analyzing left/right symmetry..."
for i in 1 2 3; do
    echo "   Sample $i:"
    timeout 1 ros2 topic echo /motor_pos --once 2>/dev/null | grep -A 8 "position:" || echo "   No data"
done
echo ""

echo "5. /position_controller/commands (controller input) - 3 samples:"
echo "   Order: [BR, FR, BL, FL] (2 angles each)"
for i in 1 2 3; do
    echo "   Sample $i:"
    timeout 1 ros2 topic echo /position_controller/commands --once 2>/dev/null | grep -A 8 "data:" || echo "   No data"
done
echo ""

echo "6. /joint_states (actual positions from Gazebo) - 3 samples:"
echo "   Checking for oscillations/vibrations..."
for i in 1 2 3; do
    echo "   Sample $i:"
    timeout 1 ros2 topic echo /joint_states --once 2>/dev/null | grep -A 20 "position:" | head -12 || echo "   No data"
    sleep 0.2
done
echo ""

echo "7. Topic frequencies (Hz):"
echo "   /motor_pos:"
timeout 3 ros2 topic hz /motor_pos 2>/dev/null | grep "average rate" || echo "   No data"
echo "   /position_controller/commands:"
timeout 3 ros2 topic hz /position_controller/commands 2>/dev/null | grep "average rate" || echo "   No data"
echo "   /joint_states:"
timeout 3 ros2 topic hz /joint_states 2>/dev/null | grep "average rate" || echo "   No data"
echo ""

echo "=============================================="
echo "Analysis Tips:"
echo "- Left legs (BL, FL) and Right legs (BR, FR) should have similar magnitudes"
echo "- If positions vary wildly between samples, there's oscillation"
echo "- Check if left/right pairs are symmetric (BL vs BR, FL vs FR)"
echo "=============================================="
