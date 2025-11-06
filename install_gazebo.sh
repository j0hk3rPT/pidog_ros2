#!/bin/bash
# Install ROS 2 Jazzy + Gazebo Harmonic for Ubuntu 24.04
# This is the officially supported LTS combination for Ubuntu 24.04

set -e

echo "============================================"
echo "Installing ROS 2 Jazzy + Gazebo Harmonic"
echo "for Ubuntu 24.04 (Noble)"
echo "============================================"

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu version: $UBUNTU_VERSION"

if [ "$UBUNTU_VERSION" != "24.04" ]; then
    echo "WARNING: This script is designed for Ubuntu 24.04"
    echo "You are running Ubuntu $UBUNTU_VERSION"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update package list first
echo "Updating package list..."
sudo apt-get update

# Ensure curl is available
echo "Installing curl if needed..."
sudo apt-get install -y curl gnupg lsb-release

# ============================================
# Install ROS 2 Jazzy
# ============================================
echo ""
echo "============================================"
echo "Installing ROS 2 Jazzy Jalisco"
echo "============================================"

# Add ROS 2 repository
echo "Adding ROS 2 repository..."
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update after adding ROS repository
sudo apt-get update

# Install ROS 2 Jazzy Desktop
echo "Installing ROS 2 Jazzy (this may take a while)..."
sudo apt-get install -y ros-jazzy-desktop

# Install development tools
echo "Installing ROS 2 development tools..."
sudo apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-dev-tools

# Initialize rosdep
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    echo "Initializing rosdep..."
    sudo rosdep init
fi
rosdep update

# ============================================
# Install Gazebo Harmonic
# ============================================
echo ""
echo "============================================"
echo "Installing Gazebo Harmonic"
echo "============================================"

# Add Gazebo repository
echo "Adding Gazebo repository..."
sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update package list with Gazebo repository
echo "Updating package list..."
sudo apt-get update

# Install Gazebo Harmonic
echo "Installing Gazebo Harmonic..."
sudo apt-get install -y gz-harmonic

# ============================================
# Install ROS 2 - Gazebo Integration
# ============================================
echo ""
echo "============================================"
echo "Installing ROS 2 - Gazebo Bridge"
echo "============================================"

# Install ros_gz packages (ROS 2 Jazzy with Gazebo Harmonic)
echo "Installing ros_gz packages..."
sudo apt-get install -y ros-jazzy-ros-gz

# ============================================
# Install ros2_control packages
# ============================================
echo ""
echo "============================================"
echo "Installing ros2_control"
echo "============================================"

sudo apt-get install -y \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-position-controllers \
    ros-jazzy-effort-controllers \
    ros-jazzy-velocity-controllers

# Install gz-ros2-control
echo "Installing gz_ros2_control..."
sudo apt-get install -y ros-jazzy-gz-ros2-control || echo "Note: gz-ros2-control not found, may need manual installation"

# ============================================
# Install additional dependencies
# ============================================
echo ""
echo "============================================"
echo "Installing Additional Dependencies"
echo "============================================"

sudo apt-get install -y \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-rviz2

# ============================================
# Setup environment
# ============================================
echo ""
echo "============================================"
echo "Setting up environment"
echo "============================================"

# Add ROS 2 sourcing to bashrc if not already there
if ! grep -q "source /opt/ros/jazzy/setup.bash" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# ROS 2 Jazzy setup" >> ~/.bashrc
    echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
    echo "Added ROS 2 Jazzy sourcing to ~/.bashrc"
fi

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "Installed versions:"
echo "  ROS 2: $(ls /opt/ros/)"
gz sim --version 2>/dev/null || echo "  Gazebo: Command not found (may need to restart terminal)"
echo ""
echo "IMPORTANT: Source your environment:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  OR restart your terminal"
echo ""
echo "To build and run PiDog simulation:"
echo "  cd ~/pidog_ros2"
echo "  colcon build --symlink-install"
echo "  source install/setup.bash"
echo "  ros2 launch pidog_description gazebo.launch.py"
echo ""
echo "Verify installation:"
echo "  ros2 pkg list | grep ros_gz"
echo "  ros2 pkg list | grep ros2_control"
echo ""
