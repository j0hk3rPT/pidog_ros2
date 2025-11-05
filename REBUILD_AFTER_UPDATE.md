# 🔨 Rebuild Instructions After Git Pull

When you pull updates that modify launch files or Python code, you need to rebuild the ROS2 workspace.

## Quick Rebuild

```bash
cd /home/user/pidog_ros2

# Rebuild only the changed packages
colcon build --packages-select pidog_gaits pidog_sim

# Source the updated installation
source install/setup.bash
```

## When Is Rebuild Needed?

### ✅ Rebuild Required:
- Launch files changed (*.launch.py)
- Python node code changed (*.py in package)
- Package configuration changed (package.xml, setup.py)
- After `git pull` if launch files were updated

### ❌ No Rebuild Needed:
- Shell scripts changed (*.sh)
- Documentation changed (*.md)
- World files changed (*.wbt)
- Data files or configs

## Full Rebuild (If Issues Persist)

```bash
cd /home/user/pidog_ros2

# Clean build
rm -rf build/ install/ log/

# Full rebuild
colcon build

# Source
source install/setup.bash
```

## After Latest Update (Robot Stability Fix)

The latest changes fixed the robot jumping issue. To apply:

```bash
cd /home/user/pidog_ros2

# Pull changes
git pull

# Rebuild (launch file changed)
colcon build --packages-select pidog_gaits

# Source
source install/setup.bash

# Test - robot should now start in stable 'stand' pose
ros2 launch pidog_gaits collect_data.launch.py
```

## Verify Rebuild Worked

```bash
# Check launch file parameter
ros2 param get /gait_generator default_gait

# Should show: 'stand' (not 'walk_forward')
```

If it still shows 'walk_forward', you need to rebuild and re-source.
