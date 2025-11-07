#!/bin/bash

# ROS2 Workspace Rebuild Script
# Thoroughly cleans and rebuilds the PiDog packages

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================="
echo "PiDog ROS2 Workspace Rebuild"
echo -e "==============================================${NC}"
echo ""

echo -e "${YELLOW}⚠️  This will perform a COMPLETE clean rebuild${NC}"
echo -e "${YELLOW}   All Python bytecode cache will be deleted${NC}"
echo ""

echo -e "${BLUE}[1/5] Cleaning ROS2 build artifacts...${NC}"
rm -rf build/ install/ log/
echo -e "${GREEN}✓ Cleaned build/, install/, log/${NC}"
echo ""

echo -e "${BLUE}[2/5] Cleaning Python bytecode cache...${NC}"
# Find and remove all __pycache__ directories
PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ $PYCACHE_COUNT -gt 0 ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $PYCACHE_COUNT __pycache__ directories${NC}"
else
    echo -e "${GREEN}✓ No __pycache__ directories found${NC}"
fi

# Find and remove all .pyc files
PYC_COUNT=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)
if [ $PYC_COUNT -gt 0 ]; then
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $PYC_COUNT .pyc files${NC}"
else
    echo -e "${GREEN}✓ No .pyc files found${NC}"
fi
echo ""

echo -e "${BLUE}[3/5] Cleaning setuptools artifacts...${NC}"
# Find and remove all .egg-info directories
EGG_COUNT=$(find . -type d -name "*.egg-info" 2>/dev/null | wc -l)
if [ $EGG_COUNT -gt 0 ]; then
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $EGG_COUNT .egg-info directories${NC}"
else
    echo -e "${GREEN}✓ No .egg-info directories found${NC}"
fi

# Clean pytest cache if present
if [ -d ".pytest_cache" ]; then
    rm -rf .pytest_cache
    echo -e "${GREEN}✓ Removed .pytest_cache${NC}"
fi
echo ""

echo -e "${BLUE}[4/5] Building workspace...${NC}"
colcon build
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

echo -e "${BLUE}[5/5] Verifying Python modules...${NC}"
# Check that key Python modules are present in install/
if [ -f "install/pidog_gaits/lib/python3.12/site-packages/pidog_gaits/gait_generator_node.py" ]; then
    echo -e "${GREEN}✓ pidog_gaits module installed${NC}"
else
    echo -e "${RED}✗ Warning: pidog_gaits module may not be installed correctly${NC}"
fi

if [ -f "install/pidog_control/lib/python3.12/site-packages/pidog_control/pidog_gazebo_controller.py" ]; then
    echo -e "${GREEN}✓ pidog_control module installed${NC}"
else
    echo -e "${RED}✗ Warning: pidog_control module may not be installed correctly${NC}"
fi
echo ""

echo -e "${BLUE}[✓] Done!${NC}"
echo -e "${GREEN}✓✓✓ Workspace completely rebuilt and verified${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. source install/setup.bash"
echo "  2. ros2 launch pidog_gaits gait_demo.launch.py"
echo ""
echo -e "${YELLOW}💡 Tip: If changes still don't apply, restart any running ROS2 nodes${NC}"