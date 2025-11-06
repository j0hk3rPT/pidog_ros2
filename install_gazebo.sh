#!/bin/bash
# Install ROS 2 Jazzy + Gazebo Harmonic for Ubuntu 24.04
# Following official documentation from:
# - https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
# - https://gazebosim.org/docs/harmonic/install_ubuntu/

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

# ============================================
# Install ROS 2 Jazzy (Official Method)
# ============================================
echo ""
echo "============================================"
echo "Installing ROS 2 Jazzy Jalisco"
echo "============================================"

# Set locale
echo "Setting up locale..."
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Enable required repositories
echo "Enabling required repositories..."
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y

# Setup ROS 2 apt repository (new official method)
echo "Setting up ROS 2 apt repository..."
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
rm /tmp/ros2-apt-source.deb

# Update and upgrade
echo "Updating package lists..."
sudo apt update
sudo apt upgrade -y

# Install ROS 2 Jazzy Desktop
echo "Installing ROS 2 Jazzy Desktop (this may take 5-10 minutes)..."
sudo apt install -y ros-jazzy-desktop

# Install development tools
echo "Installing ROS 2 development tools..."
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-dev-tools

# Initialize rosdep
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    echo "Initializing rosdep..."
    sudo rosdep init
fi
echo "Updating rosdep..."
rosdep update

# ============================================
# Install Gazebo Harmonic (Official Method)
# ============================================
echo ""
echo "============================================"
echo "Installing Gazebo Harmonic"
echo "============================================"

# Install prerequisites
echo "Installing prerequisites..."
sudo apt install -y curl lsb-release gnupg

# Add Gazebo repository
echo "Adding Gazebo repository..."
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update and install Gazebo Harmonic
echo "Installing Gazebo Harmonic..."
sudo apt update
sudo apt install -y gz-harmonic

# ============================================
# Install ROS 2 - Gazebo Integration
# ============================================
echo ""
echo "============================================"
echo "Installing ROS 2 - Gazebo Bridge"
echo "============================================"

# Install ros_gz packages (ROS 2 Jazzy with Gazebo Harmonic)
echo "Installing ros_gz packages..."
sudo apt install -y ros-jazzy-ros-gz

# ============================================
# Install ros2_control packages
# ============================================
echo ""
echo "============================================"
echo "Installing ros2_control"
echo "============================================"

sudo apt install -y \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-position-controllers \
    ros-jazzy-effort-controllers \
    ros-jazzy-velocity-controllers

# Install gz-ros2-control
echo "Installing gz_ros2_control..."
sudo apt install -y ros-jazzy-gz-ros2-control || echo "Note: gz-ros2-control not found, may need manual installation"

# ============================================
# Install additional dependencies
# ============================================
echo ""
echo "============================================"
echo "Installing Additional Dependencies"
echo "============================================"

sudo apt install -y \
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
source /opt/ros/jazzy/setup.bash
echo "  ROS 2: $(ros2 --version 2>/dev/null || echo 'Unable to check version')"
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
