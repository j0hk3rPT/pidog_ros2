#!/bin/bash
# Install Gazebo Fortress and ros2_control for ROS 2 Humble

set -e

echo "============================================"
echo "Installing Gazebo Fortress for ROS 2 Humble"
echo "============================================"

# Add Gazebo package repository
echo "Adding Gazebo repository..."
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Gazebo Fortress (the correct version for ROS 2 Humble)
echo "Installing Gazebo Fortress..."
sudo apt-get install -y gz-fortress

# Install ROS 2 - Gazebo bridge for Humble
echo "Installing ROS 2 Gazebo bridge..."
sudo apt-get install -y \
    ros-humble-ros-gzfortress \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim

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

# Install gazebo_ros2_control (for Gazebo integration)
echo "Installing gazebo_ros2_control..."
sudo apt-get install -y ros-humble-gazebo-ros2-control || echo "Note: gazebo_ros2_control not available, will use alternative method"

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
gz sim --version
echo ""
echo "To use Gazebo with PiDog:"
echo "  1. Source your ROS 2 workspace: source ~/pidog_ros2/install/setup.bash"
echo "  2. Launch simulation: ros2 launch pidog_description gazebo.launch.py"
echo ""
echo "Verify installation with:"
echo "  ros2 pkg list | grep ros_gz"
echo "  ros2 pkg list | grep ros2_control"
echo ""
