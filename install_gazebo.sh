#!/bin/bash
# Install Gazebo Classic 11 and ros2_control for ROS 2 Humble

set -e

echo "============================================"
echo "Installing Gazebo Classic for ROS 2 Humble"
echo "============================================"

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Gazebo Classic 11 (the version bundled with ROS 2 Humble)
echo "Installing Gazebo Classic 11..."
sudo apt-get install -y gazebo

# Install ROS 2 Gazebo packages (Classic, not the new gz-sim)
echo "Installing ROS 2 Gazebo Classic packages..."
sudo apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-msgs \
    ros-humble-gazebo-plugins

# Install gazebo_ros2_control
echo "Installing gazebo_ros2_control..."
sudo apt-get install -y ros-humble-gazebo-ros2-control

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
gazebo --version
echo ""
echo "To use Gazebo with PiDog:"
echo "  1. Source your ROS 2 workspace: source ~/pidog_ros2/install/setup.bash"
echo "  2. Launch simulation: ros2 launch pidog_description gazebo.launch.py"
echo ""
echo "Verify installation with:"
echo "  ros2 pkg list | grep gazebo"
echo "  ros2 pkg list | grep ros2_control"
echo ""
