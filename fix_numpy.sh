#!/bin/bash
#
# Quick fix for NumPy 2.x incompatibility with ROS2 cv_bridge
#
# This downgrades NumPy from 2.x to 1.x to fix the segmentation fault
#

set -e

echo "========================================="
echo "NumPy Compatibility Fix"
echo "========================================="
echo ""

# Check current NumPy version
echo "Current NumPy version:"
python3 -c "import numpy; print(f'  NumPy {numpy.__version__}')"
echo ""

# Check if NumPy 2.x is installed
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")

if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "NumPy 2.x detected - incompatible with ROS2 cv_bridge"
    echo "Downgrading to NumPy 1.x..."
    echo ""

    # Uninstall NumPy 2.x and install NumPy 1.x
    pip uninstall -y numpy
    pip install "numpy<2"

    echo ""
    echo "✓ NumPy downgraded successfully"
    echo ""
    echo "New NumPy version:"
    python3 -c "import numpy; print(f'  NumPy {numpy.__version__}')"
    echo ""
    echo "You can now run RL training without segmentation faults!"
else
    echo "NumPy version is compatible (< 2.0)"
    echo "No action needed."
fi

echo ""
echo "========================================="
echo "Fix Complete"
echo "========================================="
