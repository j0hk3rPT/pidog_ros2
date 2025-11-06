#!/bin/bash
# Install Gazebo Harmonic for ROS 2 Humble on Ubuntu 24.04

set -e

echo "============================================"
echo "Installing Gazebo Harmonic for Ubuntu 24.04"
echo "============================================"

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu version: $UBUNTU_VERSION"

# Update package list first
echo "Updating package list..."
sudo apt-get update

# Ensure curl is available for downloading GPG key
echo "Installing curl if needed..."
sudo apt-get install -y curl

# Add Gazebo repository
echo "Adding Gazebo repository..."
sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update package list again after adding repository
echo "Updating package list with new repository..."
sudo apt-get update

# Install Gazebo Harmonic (package names for Ubuntu 24.04)
echo "Installing Gazebo Harmonic..."
sudo apt-get install -y \
    gz-harmonic \
    libgz-sim8-dev \
    libgz-common5-dev \
    libgz-plugin2-dev

# Install ROS 2 - Gazebo bridge
echo "Installing ROS 2 Gazebo bridge..."
sudo apt-get install -y \
    ros-humble-ros-gzharmonic \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-interfaces

# Install ros2_control packages
echo "Installing ros2_control..."
sudo apt-get install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-position-controllers \
    ros-humble-effort-controllers \
    ros-humble-velocity-controllers

# Install gz-ros2-control (if available)
echo "Installing gz_ros2_control..."
sudo apt-get install -y ros-humble-gz-ros2-control || echo "Note: gz-ros2-control not available, will use ign_ros2_control"

# Install additional dependencies
echo "Installing additional dependencies..."
sudo apt-get install -y \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-rviz2

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "Installed versions:"
gz sim --version || echo "gz sim command not found"
echo ""
echo "To use Gazebo with PiDog:"
echo "  1. Source your ROS 2 workspace: source ~/pidog_ros2/install/setup.bash"
echo "  2. Launch simulation: ros2 launch pidog_description gazebo.launch.py"
echo ""
echo "Verify installation with:"
echo "  ros2 pkg list | grep ros_gz"
echo "  ros2 pkg list | grep ros2_control"
echo ""
echo "If gz-harmonic failed to install, you may need to:"
echo "  1. Check available packages: apt-cache search gz-"
echo "  2. Install manually: sudo apt install <correct-package-name>"
echo ""
