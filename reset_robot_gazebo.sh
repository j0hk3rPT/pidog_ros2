#!/bin/bash

# Reset robot pose in Gazebo to standing position
# This physically moves the robot back to upright if it falls

set -e

echo "🔄 Resetting robot pose in Gazebo..."

# Reset the entire world (simplest approach)
gz service -s /world/pidog/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req 'reset: {all: true}'

# Alternative: Reset just the robot model pose
# gz service -s /world/pidog/set_pose \
#     --reqtype gz.msgs.Pose \
#     --reptype gz.msgs.Boolean \
#     --timeout 2000 \
#     --req 'name: "pidog", position: {x: 0, y: 0, z: 0.15}, orientation: {x: 0, y: 0, z: 0, w: 1}'

echo "✅ Robot reset complete!"
sleep 2  # Allow physics to stabilize
