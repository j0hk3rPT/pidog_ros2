#!/bin/bash
# Setup rendering for Gazebo in Docker
# Use this if Gazebo window is blank/white

echo "Setting up rendering environment for Gazebo..."

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Detected Docker environment"

    # Check for AMD GPU
    if lspci | grep -i amd > /dev/null 2>&1; then
        echo "AMD GPU detected"
        export LIBGL_ALWAYS_SOFTWARE=0
        export GALLIUM_DRIVER=radeonsi
    # Check for NVIDIA GPU
    elif lspci | grep -i nvidia > /dev/null 2>&1; then
        echo "NVIDIA GPU detected"
        export LIBGL_ALWAYS_SOFTWARE=0
    else
        echo "No GPU detected, using software rendering"
        export LIBGL_ALWAYS_SOFTWARE=1
        export GALLIUM_DRIVER=llvmpipe
    fi
else
    echo "Not in Docker, using default rendering"
fi

# Set Gazebo rendering engine
export OGRE_RTShader_WRITE=1
export OGRE_RENDERDOC_CAPTURE=0

# Print current settings
echo ""
echo "Rendering settings:"
echo "  LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-not set}"
echo "  GALLIUM_DRIVER=${GALLIUM_DRIVER:-not set}"
echo ""
echo "Run: ros2 launch pidog_description gazebo.launch.py"
