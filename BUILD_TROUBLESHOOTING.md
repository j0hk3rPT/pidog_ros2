# Build Troubleshooting Guide

## Problem: Changes Not Applied After Rebuild

If you modify Python code and rebuild but the changes don't take effect, this is usually caused by **cached Python bytecode**.

### Solution Ladder (Try in Order)

#### 1. Standard Rebuild (Try First)
```bash
./rebuild.sh
source install/setup.bash
```

**What it cleans:**
- `build/`, `install/`, `log/` directories
- All `__pycache__/` directories
- All `*.pyc` bytecode files
- Setuptools `*.egg-info` directories
- Pytest cache

**Verifies:**
- Key Python modules are installed correctly
- Shows counts of cleaned files

#### 2. Restart ROS2 Nodes (If Still Not Working)
```bash
# Kill all running ROS2 nodes
pkill -f "ros2 launch"
pkill -f "python3.*pidog"

# Rebuild and relaunch
./rebuild.sh
source install/setup.bash
ros2 launch pidog_gaits gait_demo.launch.py
```

**Why:** Running ROS2 nodes may have loaded old Python modules into memory.

#### 3. Nuclear Clean (Last Resort)
```bash
./clean_all.sh  # Prompts for confirmation
./rebuild.sh
source install/setup.bash
```

**What it cleans (in addition to standard rebuild):**
- `*.pyo` optimized bytecode
- `.eggs` directories
- `dist/` directories
- CMake cache files (`CMakeCache.txt`, `CMakeFiles/`)
- Compiled libraries (`*.so`)
- `.tox` directories

**Use when:**
- Standard rebuild doesn't work
- Switching between Python versions
- Build corruption suspected
- Very strange behavior after changes

## Common Scenarios

### Scenario 1: Modified `gait_generator_node.py`
```bash
# Edit the file
vim pidog_gaits/pidog_gaits/gait_generator_node.py

# Standard rebuild is usually enough
./rebuild.sh
source install/setup.bash

# Restart the node
ros2 launch pidog_gaits gait_demo.launch.py
```

### Scenario 2: Modified Multiple Python Files
```bash
# After editing multiple files
./rebuild.sh
source install/setup.bash

# Make sure no old nodes are running
pkill -f "ros2 launch"
ros2 launch pidog_gaits gait_demo.launch.py
```

### Scenario 3: Changes Still Not Applying
```bash
# Nuclear option
./clean_all.sh
./rebuild.sh
source install/setup.bash
ros2 launch pidog_gaits gait_demo.launch.py
```

### Scenario 4: Modified URDF or Config Files
```bash
# URDF/YAML changes need rebuild but not aggressive cleaning
./rebuild.sh
source install/setup.bash
ros2 launch pidog_description gazebo.launch.py
```

## Understanding Python Bytecode Caching

### Why Changes Don't Apply

Python compiles `.py` files to `.pyc` bytecode for faster loading:

```
pidog_gaits/
  pidog_gaits/
    gait_generator_node.py      # Your source code
    __pycache__/
      gait_generator_node.cpython-312.pyc  # Cached bytecode
```

**The problem:**
1. You edit `gait_generator_node.py`
2. `colcon build` copies files to `install/`
3. Python loads **old bytecode** from `__pycache__/` instead of new source
4. Your changes don't run

**The solution:**
- `./rebuild.sh` deletes all `__pycache__/` directories
- Forces Python to recompile from source

### When Python Bytecode is NOT the Problem

If the issue persists after nuclear clean, check:

1. **Did you source the workspace?**
   ```bash
   source install/setup.bash
   ```

2. **Are you editing the right file?**
   ```bash
   # Edit source, not installed files
   vim pidog_gaits/pidog_gaits/gait_generator_node.py  # ✅ Correct
   # NOT
   vim install/pidog_gaits/.../gait_generator_node.py  # ❌ Wrong
   ```

3. **Is the node actually running your code?**
   ```bash
   # Check which Python executable is running
   ps aux | grep python3 | grep pidog
   ```

4. **Is there a syntax error?**
   ```bash
   # Check for Python syntax errors
   python3 -m py_compile pidog_gaits/pidog_gaits/gait_generator_node.py
   ```

## Verification Checklist

After rebuilding, verify:

```bash
# 1. Check file was copied to install/
ls -lh install/pidog_gaits/lib/python3.12/site-packages/pidog_gaits/gait_generator_node.py

# 2. Check timestamp matches your edit
stat pidog_gaits/pidog_gaits/gait_generator_node.py
stat install/pidog_gaits/lib/python3.12/site-packages/pidog_gaits/gait_generator_node.py

# 3. No __pycache__ in source directories
find pidog_gaits/ -name "__pycache__"  # Should be empty

# 4. Workspace is sourced
echo $AMENT_PREFIX_PATH  # Should include install path
```

## Quick Reference

| Problem | Solution | Command |
|---------|----------|---------|
| First time issue | Standard rebuild | `./rebuild.sh` |
| Changes not applied | Standard rebuild + restart | `./rebuild.sh && pkill -f "ros2 launch"` |
| Still not working | Nuclear clean | `./clean_all.sh && ./rebuild.sh` |
| URDF/YAML changes | Standard rebuild | `./rebuild.sh` |
| Permission errors | Use root in container | `docker exec -it -u root pidog_ros2 bash` |

## Advanced: Manual Cleaning

If scripts don't work, manual cleaning:

```bash
# Remove build artifacts
rm -rf build/ install/ log/

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Remove setuptools artifacts
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Rebuild
colcon build
source install/setup.bash
```

## Prevention Tips

1. **Always rebuild after editing Python files**
2. **Always source after rebuilding**
3. **Always restart ROS2 nodes after rebuilding**
4. **Edit source files, never installed files**
5. **Run rebuild script from workspace root**

## Still Having Issues?

If none of the above works:

1. Check Docker container is up to date
2. Verify ROS2 Jazzy is correctly installed
3. Check disk space: `df -h`
4. Check for Python version conflicts
5. Try rebuilding individual packages: `colcon build --packages-select pidog_gaits`
