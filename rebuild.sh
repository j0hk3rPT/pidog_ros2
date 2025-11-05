#!/bin/bash

# ROS2 Workspace Rebuild Script
# Cleans and rebuilds the PiDog packages

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================="
echo "PiDog ROS2 Workspace Rebuild"
echo -e "==============================================${NC}"
echo ""

# Parse arguments
CLEAN_TYPE="quick"
if [ "$1" == "full" ] || [ "$1" == "--full" ]; then
    CLEAN_TYPE="full"
fi

# Show what we're going to do
if [ "$CLEAN_TYPE" == "full" ]; then
    echo -e "${YELLOW}Mode: FULL CLEAN${NC}"
    echo "This will:"
    echo "  1. Remove build/, install/, and log/ directories"
    echo "  2. Rebuild ALL packages from scratch"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo -e "${GREEN}Mode: QUICK REBUILD${NC}"
    echo "This will:"
    echo "  1. Clean only pidog_gaits and pidog_sim packages"
    echo "  2. Rebuild these packages"
    echo ""
    echo -e "${YELLOW}Tip: Use './rebuild.sh full' for a complete clean rebuild${NC}"
    echo ""
fi

# Navigate to workspace root
cd /home/user/pidog_ros2

# Full clean
if [ "$CLEAN_TYPE" == "full" ]; then
    echo -e "${BLUE}[1/4] Cleaning workspace...${NC}"
    rm -rf build/ install/ log/
    echo -e "${GREEN}✓ Removed build/, install/, log/${NC}"
    echo ""

    echo -e "${BLUE}[2/4] Rebuilding all packages...${NC}"
    colcon build
    echo -e "${GREEN}✓ All packages rebuilt${NC}"
    echo ""
else
    # Quick clean - only modified packages
    echo -e "${BLUE}[1/4] Cleaning pidog_gaits and pidog_sim...${NC}"
    if [ -d "build/pidog_gaits" ]; then
        rm -rf build/pidog_gaits
        echo -e "${GREEN}✓ Removed build/pidog_gaits${NC}"
    fi
    if [ -d "build/pidog_sim" ]; then
        rm -rf build/pidog_sim
        echo -e "${GREEN}✓ Removed build/pidog_sim${NC}"
    fi
    if [ -d "install/pidog_gaits" ]; then
        rm -rf install/pidog_gaits
        echo -e "${GREEN}✓ Removed install/pidog_gaits${NC}"
    fi
    if [ -d "install/pidog_sim" ]; then
        rm -rf install/pidog_sim
        echo -e "${GREEN}✓ Removed install/pidog_sim${NC}"
    fi
    echo ""

    echo -e "${BLUE}[2/4] Rebuilding modified packages...${NC}"
    colcon build --packages-select pidog_gaits pidog_sim
    echo -e "${GREEN}✓ pidog_gaits and pidog_sim rebuilt${NC}"
    echo ""
fi

echo -e "${BLUE}[3/4] Sourcing installation...${NC}"
source install/setup.bash
echo -e "${GREEN}✓ Installation sourced${NC}"
echo ""

echo -e "${BLUE}[4/4] Verifying rebuild...${NC}"

# Check if packages exist
if [ -f "install/pidog_gaits/share/pidog_gaits/package.xml" ]; then
    echo -e "${GREEN}✓ pidog_gaits installed${NC}"
else
    echo -e "${RED}✗ pidog_gaits NOT found${NC}"
fi

if [ -f "install/pidog_sim/share/pidog_sim/package.xml" ]; then
    echo -e "${GREEN}✓ pidog_sim installed${NC}"
else
    echo -e "${RED}✗ pidog_sim NOT found${NC}"
fi

echo ""
echo -e "${GREEN}=============================================="
echo "✓ Rebuild Complete!"
echo -e "==============================================${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC} Source the workspace in your terminal:"
echo ""
echo "  ${BLUE}source install/setup.bash${NC}"
echo ""
echo "Or add to your ~/.bashrc:"
echo ""
echo "  ${BLUE}source /home/user/pidog_ros2/install/setup.bash${NC}"
echo ""
echo "Then you can run:"
echo "  ${GREEN}ros2 launch pidog_gaits collect_data.launch.py${NC}"
echo "  ${GREEN}./collect_training_data.sh 40${NC}"
echo ""
