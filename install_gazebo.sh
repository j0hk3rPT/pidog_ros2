#!/bin/bash
# Install Gazebo Harmonic and ros2_control for ROS 2 Humble

set -e

echo "============================================"
echo "Installing Gazebo Harmonic for ROS 2 Humble"
echo "============================================"

# Add Gazebo package repository
echo "Adding Gazebo repository..."
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Gazebo Harmonic
echo "Installing Gazebo Harmonic..."
sudo apt-get install -y gz-harmonic

# Install ROS 2 - Gazebo bridge
echo "Installing ROS 2 Gazebo bridge..."
sudo apt-get install -y ros-humble-ros-gz

# Install ros2_control packages
echo "Installing ros2_control..."
sudo apt-get install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-gz-ros2-control \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-position-controllers

# Install additional dependencies
echo "Installing additional dependencies..."
sudo apt-get install -y \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "To use Gazebo with PiDog:"
echo "  1. Source your ROS 2 workspace: source ~/pidog_ros2/install/setup.bash"
echo "  2. Launch simulation: ros2 launch pidog_description gazebo.launch.py"
echo ""
echo "Verify installation with:"
echo "  gz sim --version"
echo "  ros2 pkg list | grep gz"
echo ""
