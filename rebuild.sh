#!/bin/bash

# ROS2 Workspace Rebuild Script
# Cleans and rebuilds the PiDog packages

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================="
echo "PiDog ROS2 Workspace Rebuild"
echo -e "==============================================${NC}"
echo ""

echo -e "${BLUE}[1/3] Cleaning workspace...${NC}"
rm -rf build/ install/ log/
echo -e "${GREEN}✓ Cleaned build/, install/, log/${NC}"
echo ""

echo -e "${BLUE}[2/3] Building workspace...${NC}"
colcon build
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

echo -e "${BLUE}[3/3] Done!${NC}"
echo -e "${GREEN}✓ Workspace rebuilt successfully${NC}"
echo ""
echo -e "${BLUE}Next step:${NC} source install/setup.bash"