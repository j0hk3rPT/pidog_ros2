#!/bin/bash
# Test if Gazebo world loads without sensor plugins

source install/setup.bash

echo "======================================================================"
echo "Testing Gazebo World Loading (NO SENSOR PLUGINS)"
echo "======================================================================"
echo ""
echo "This test launches Gazebo with sensor plugins DISABLED"
echo "to determine if they are causing the world service hang."
echo ""
echo "Launching Gazebo with test world file..."
echo ""

# Launch Gazebo with test world (no sensors)
ros2 run ros_gz_sim gz_sim -r pidog_description/worlds/pidog_test_no_sensors.sdf --verbose &
GZ_PID=$!

echo "Waiting 15 seconds for Gazebo to load..."
sleep 15

echo ""
echo "Checking if world service is available..."
if gz service -l 2>/dev/null | grep -q "/world/pidog_world/create"; then
    echo "✅ SUCCESS! World service is available!"
    echo ""
    echo "This means sensor plugins were causing the hang."
    echo "We need an alternative IMU approach."
else
    echo "❌ FAILED! World service still not available."
    echo ""
    echo "This means the issue is NOT the sensor plugins."
    echo "Something else is preventing Gazebo from loading."
fi

echo ""
echo "All Gazebo services:"
gz service -l 2>/dev/null | head -20

echo ""
echo "Stopping Gazebo..."
kill $GZ_PID
wait $GZ_PID 2>/dev/null

echo "Test complete."
