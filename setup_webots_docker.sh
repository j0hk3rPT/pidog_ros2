#!/bin/bash
#
# Webots Docker Setup Script for ROS 2
#
# This script configures the Docker environment to run Webots simulator
# with the PiDog ROS 2 project.
#

set -e

echo "========================================="
echo "Webots Docker Setup for ROS 2"
echo "========================================="
echo ""

# Check and install missing libraries
echo "Checking for required libraries..."
MISSING_LIBS=()

if ! ldconfig -p | grep -q "libsndio.so.7"; then
    MISSING_LIBS+=("libsndio7.0")
fi

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    echo "Installing missing libraries: ${MISSING_LIBS[*]}"
    apt-get update -qq
    apt-get install -y -qq "${MISSING_LIBS[@]}"
    echo "✓ Libraries installed successfully"
else
    echo "✓ All required libraries present"
fi

echo ""

# Set environment variables
echo "Setting environment variables..."
export DISPLAY=:99
export USER=${USER:-root}
export USERNAME=${USERNAME:-root}
export WEBOTS_HOME=/root/.ros/webotsR2025a/webots

# Check if Xvfb is running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting Xvfb virtual display..."
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &>/dev/null &
    sleep 2
    echo "Xvfb started on display :99"
else
    echo "Xvfb already running"
fi

# Test OpenGL
echo ""
echo "Testing OpenGL rendering..."
if DISPLAY=:99 glxinfo >/dev/null 2>&1; then
    echo "✓ OpenGL is working correctly"
    DISPLAY=:99 glxinfo | grep "direct rendering"
    DISPLAY=:99 glxinfo | grep "OpenGL renderer"
else
    echo "✗ OpenGL test failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Environment variables set:"
echo "  DISPLAY=$DISPLAY"
echo "  USER=$USER"
echo "  USERNAME=$USERNAME"
echo "  WEBOTS_HOME=$WEBOTS_HOME"
echo ""
echo "You can now run:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  source install/setup.bash"
echo "  ros2 launch pidog_gaits gait_demo.launch.py"
echo ""
