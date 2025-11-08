#!/bin/bash

# Nuclear Clean Script
# Use this if changes still don't apply after rebuild.sh
# This performs the most aggressive cleaning possible

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}=============================================="
echo "⚠️  NUCLEAR CLEAN - USE WITH CAUTION ⚠️"
echo -e "==============================================${NC}"
echo ""
echo -e "${YELLOW}This will delete:${NC}"
echo "  • All build artifacts (build/, install/, log/)"
echo "  • All Python bytecode (__pycache__/, *.pyc)"
echo "  • All setuptools artifacts (*.egg-info)"
echo "  • CMake cache files"
echo "  • All compiled libraries (*.so)"
echo ""
echo -e "${RED}Use this ONLY if normal rebuild doesn't work!${NC}"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo -e "${YELLOW}[1/6] Removing ROS2 build directories...${NC}"
rm -rf build/ install/ log/
echo -e "${GREEN}✓ Removed build/, install/, log/${NC}"

echo ""
echo -e "${YELLOW}[2/6] Removing Python bytecode cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Removed all Python bytecode${NC}"

echo ""
echo -e "${YELLOW}[3/6] Removing setuptools artifacts...${NC}"
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".eggs" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Removed setuptools artifacts${NC}"

echo ""
echo -e "${YELLOW}[4/6] Removing CMake cache files...${NC}"
find . -type f -name "CMakeCache.txt" -delete 2>/dev/null || true
find . -type d -name "CMakeFiles" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Removed CMake cache${NC}"

echo ""
echo -e "${YELLOW}[5/6] Removing compiled libraries...${NC}"
find . -type f -name "*.so" -delete 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Removed compiled libraries${NC}"

echo ""
echo -e "${YELLOW}[6/6] Removing test caches...${NC}"
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Removed test caches${NC}"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓✓✓ Nuclear clean complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Run: ./rebuild.sh"
echo "  2. Run: source install/setup.bash"
echo "  3. Test your changes"
echo ""
