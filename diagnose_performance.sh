#!/bin/bash
#
# Performance Diagnostic Script
# Checks why CPU/GPU usage is low during RL training
#

echo "========================================"
echo "RL Training Performance Diagnostics"
echo "========================================"
echo ""

# 1. Check if NumPy is correct version
echo "[1/8] Checking NumPy version..."
NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)" 2>&1)
if [[ "$NUMPY_VER" == 2.* ]]; then
    echo "  ❌ NumPy $NUMPY_VER detected (INCOMPATIBLE!)"
    echo "     Run: ./fix_numpy.sh"
else
    echo "  ✓ NumPy $NUMPY_VER (compatible)"
fi
echo ""

# 2. Check if Gazebo is running
echo "[2/8] Checking if Gazebo is running..."
GZ_PIDS=$(pgrep -f "gz sim" | wc -l)
if [ "$GZ_PIDS" -eq 0 ]; then
    echo "  ❌ Gazebo NOT running!"
    echo "     Start with: ros2 launch pidog_description gazebo_rl_fast.launch.py &"
else
    echo "  ✓ Gazebo running ($GZ_PIDS instances)"
    # Get actual CPU usage of Gazebo
    GZ_CPU=$(ps aux | grep -E "gz sim" | grep -v grep | awk '{sum+=$3} END {printf "%.1f", sum}')
    echo "     Current CPU usage: ${GZ_CPU}%"
fi
echo ""

# 3. Check Gazebo real-time factor
echo "[3/8] Checking Gazebo real-time factor..."
if [ -f "pidog_description/worlds/pidog_rl_fast.sdf" ]; then
    RTF=$(grep -A 1 "real_time_factor" pidog_description/worlds/pidog_rl_fast.sdf | grep -oP '>\K[0-9.]+(?=<)')
    if [ "$RTF" == "0" ] || [ "$RTF" == "0.0" ]; then
        echo "  ✓ real_time_factor = $RTF (UNLIMITED)"
    else
        echo "  ❌ real_time_factor = $RTF (RATE LIMITED!)"
        echo "     Should be 0 for max speed"
    fi
else
    echo "  ⚠ Fast world file not found"
fi
echo ""

# 4. Check if training is actually running
echo "[4/8] Checking Python training process..."
PYTHON_PIDS=$(pgrep -f "train_rl_vision" | wc -l)
if [ "$PYTHON_PIDS" -eq 0 ]; then
    echo "  ❌ No training process running"
else
    echo "  ✓ Training running ($PYTHON_PIDS processes)"
    # Get Python CPU usage
    PY_CPU=$(ps aux | grep -E "train_rl_vision" | grep -v grep | awk '{sum+=$3} END {printf "%.1f", sum}')
    echo "     Python CPU usage: ${PY_CPU}%"
fi
echo ""

# 5. Check GPU usage (if nvidia)
echo "[5/8] Checking GPU usage..."
if command -v nvidia-smi &> /dev/null; then
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  GPU Utilization: ${GPU_UTIL}%"
    echo "  GPU Memory: ${GPU_MEM}%"
elif command -v rocm-smi &> /dev/null; then
    echo "  AMD GPU detected (rocm-smi available)"
    rocm-smi --showuse 2>/dev/null | grep -E "GPU use|Memory" || echo "  (Run rocm-smi manually for details)"
else
    echo "  ⚠ No GPU monitoring tool found"
fi
echo ""

# 6. Check number of parallel environments
echo "[6/8] Checking parallel environment count..."
PYTHON_ARGS=$(ps aux | grep -E "train_rl" | grep -v grep | head -1)
if echo "$PYTHON_ARGS" | grep -q "\--envs"; then
    NUM_ENVS=$(echo "$PYTHON_ARGS" | grep -oP '\--envs\s+\K[0-9]+')
    echo "  Parallel environments: $NUM_ENVS"
    if [ "$NUM_ENVS" -eq 1 ]; then
        echo "  ⚠ Only 1 environment - may not max out CPU/GPU"
        echo "     Try: ./train_rl_vision_fast.sh 10000 4"
    else
        echo "  ✓ Using multiple environments (better utilization)"
    fi
else
    echo "  ⚠ Could not detect environment count"
fi
echo ""

# 7. Check if Gazebo is headless
echo "[7/8] Checking Gazebo mode..."
if ps aux | grep -E "gz sim" | grep -q "\-s"; then
    echo "  ✓ Gazebo running in HEADLESS mode (-s flag)"
else
    echo "  ⚠ Gazebo might be running with GUI (slower)"
fi
echo ""

# 8. Overall system load
echo "[8/8] System resource summary..."
echo "  Total CPU cores: $(nproc)"
echo "  Overall CPU usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
FREE_MEM=$(free -h | grep Mem | awk '{print $4}')
echo "  Available memory: $FREE_MEM"
echo ""

echo "========================================"
echo "Diagnosis Complete"
echo "========================================"
echo ""

# Provide recommendations
echo "Recommendations:"
echo ""

if [[ "$NUMPY_VER" == 2.* ]]; then
    echo "1. ❌ CRITICAL: Fix NumPy version first!"
    echo "   Run: ./fix_numpy.sh"
    echo ""
fi

if [ "$GZ_PIDS" -eq 0 ]; then
    echo "2. ❌ Start Gazebo:"
    echo "   ros2 launch pidog_description gazebo_rl_fast.launch.py &"
    echo ""
fi

if [ "$NUM_ENVS" == "1" ] || [ -z "$NUM_ENVS" ]; then
    echo "3. ⚠ Use more parallel environments for higher CPU usage:"
    echo "   ./train_rl_vision_fast.sh 20000 4  # 4 parallel envs"
    echo ""
fi

# Check if CPU usage is actually low
TOTAL_CPU=$(ps aux | grep -E "gz sim|train_rl" | grep -v grep | awk '{sum+=$3} END {print sum}')
if (( $(echo "$TOTAL_CPU < 50" | bc -l 2>/dev/null || echo "1") )); then
    echo "4. ⚠ CPU usage is low ($TOTAL_CPU%). Possible causes:"
    echo "   - Gazebo might be rate-limited (check real_time_factor=0)"
    echo "   - Training waiting on I/O or ROS messages"
    echo "   - Python GIL bottleneck (use more parallel envs)"
    echo ""
fi

echo "For live monitoring, run in another terminal:"
echo "  watch -n 1 'ps aux | grep -E \"gz sim|train_rl\" | grep -v grep'"
