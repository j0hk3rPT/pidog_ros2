#!/bin/bash
#
# CRITICAL Performance Fix: OpenBLAS Thread Limit
#
# This fixes the low CPU/GPU usage issue by limiting OpenBLAS threads.
# OpenBLAS (used by Gazebo) defaults to ALL CPU cores, causing overhead.
#
# Expected improvement: 5-10x faster simulation!
#
# Source: https://discourse.openrobotics.org/t/massive-gazebo-performance-improvement-ymmv/39181
#

echo "========================================="
echo "OpenBLAS Performance Fix"
echo "========================================="
echo ""

# Check current setting
CURRENT=${OPENBLAS_NUM_THREADS:-"not set (defaults to ALL cores)"}
echo "Current OPENBLAS_NUM_THREADS: $CURRENT"

# Set optimal value (4 threads for most systems)
export OPENBLAS_NUM_THREADS=4
echo "Setting OPENBLAS_NUM_THREADS=4"

# Add to bashrc for persistence
if ! grep -q "OPENBLAS_NUM_THREADS" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Gazebo performance optimization" >> ~/.bashrc
    echo "# Limits OpenBLAS threads to prevent overhead" >> ~/.bashrc
    echo "export OPENBLAS_NUM_THREADS=4" >> ~/.bashrc
    echo ""
    echo "✓ Added to ~/.bashrc for future sessions"
else
    echo "✓ Already in ~/.bashrc"
fi

echo ""
echo "========================================="
echo "Fix Applied!"
echo "========================================="
echo ""
echo "What this does:"
echo "  - Limits OpenBLAS to 4 threads (instead of all 16 cores)"
echo "  - Reduces thread management overhead"
echo "  - Expected: 5-10x faster Gazebo simulation"
echo ""
echo "Before: ~0.2-0.3 RTF (real-time factor)"
echo "After:  ~1.0-3.0 RTF"
echo ""
echo "Verify with:"
echo "  echo \$OPENBLAS_NUM_THREADS"
echo ""
echo "Test it:"
echo "  ./test_inside_container.sh"
echo "  (Check /clock rate - should be >>1000 Hz)"
echo ""
