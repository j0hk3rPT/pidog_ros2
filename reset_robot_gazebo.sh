#!/bin/bash

# Reset robot pose in Gazebo to standing position
# This physically moves the robot back to upright if it falls

set -e

echo "🔄 Resetting robot pose in Gazebo..."

# Reset only model poses (NOT 'all: true' which deletes entities!)
gz service -s /world/pidog_world/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req 'reset: {model_only: true}'

if [ $? -eq 0 ]; then
    echo "✅ Robot reset complete!"
else
    echo "❌ Reset failed! Is Gazebo running?"
    echo "   Try: gz service -l | grep world"
    exit 1
fi

sleep 2  # Allow physics to stabilize
