#!/bin/bash

# Reset robot pose in Gazebo to standing position
# This physically moves the robot back to upright if it falls

set -e

echo "📍 Setting stand pose..."
ros2 topic pub /gait_command std_msgs/msg/String "data: 'stand'" --once
sleep 1

echo "🔄 Resetting robot position in Gazebo..."

# Use set_pose service to reset robot position and orientation
# Position: x=0, y=0, z=0.12 (same as spawn height)
# Orientation: upright (quaternion w=1, x=y=z=0)
gz service -s /world/pidog_world/set_pose \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req "name: 'Robot.urdf', position: {x: 0.0, y: 0.0, z: 0.12}, orientation: {x: 0, y: 0, z: 0, w: 1}"

if [ $? -eq 0 ]; then
    echo "✅ Robot reset complete!"
else
    echo "❌ Reset failed! Is Gazebo running?"
    echo "   Try: gz service -l | grep world"
    exit 1
fi

sleep 2  # Allow physics to stabilize
