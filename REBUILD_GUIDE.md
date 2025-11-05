# 🔨 Rebuild Guide

Quick reference for rebuilding the PiDog ROS2 workspace.

---

## 🚀 Quick Rebuild (Recommended)

When you pull updates that affect launch files or Python code:

```bash
cd /home/user/pidog_ros2
./rebuild.sh
```

**What it does:**
- Cleans only `pidog_gaits` and `pidog_sim` packages
- Rebuilds these packages
- Sources the installation
- Fast (~30 seconds)

---

## 🔥 Full Clean Rebuild

When things are really broken or you want a fresh start:

```bash
cd /home/user/pidog_ros2
./rebuild.sh full
```

**What it does:**
- Removes ALL build artifacts (`build/`, `install/`, `log/`)
- Rebuilds EVERYTHING from scratch
- Sources the installation
- Slower (~2-3 minutes)

---

## 🛠️ Manual Rebuild Commands

### Quick Rebuild (Specific Packages)

```bash
cd /home/user/pidog_ros2

# Clean specific packages
rm -rf build/pidog_gaits install/pidog_gaits
rm -rf build/pidog_sim install/pidog_sim

# Rebuild
colcon build --packages-select pidog_gaits pidog_sim

# Source
source install/setup.bash
```

### Full Clean Rebuild

```bash
cd /home/user/pidog_ros2

# Clean everything
rm -rf build/ install/ log/

# Rebuild all
colcon build

# Source
source install/setup.bash
```

---

## ⚠️ When Do You Need To Rebuild?

### ✅ REBUILD REQUIRED

After changes to:
- **Launch files** (`*.launch.py`)
- **Python node code** (`*.py` in package source)
- **Package configuration** (`package.xml`, `setup.py`)
- **After `git pull`** if launch files or Python code changed

### ❌ NO REBUILD NEEDED

After changes to:
- **Shell scripts** (`*.sh`) - Just `chmod +x` if needed
- **Documentation** (`*.md`)
- **World files** (`*.wbt`)
- **Data/config files**

---

## 🔍 Common Issues

### "Module not found" or "Package not found"

**Solution:** Rebuild and source
```bash
./rebuild.sh
source install/setup.bash
```

### "Old parameters still showing"

**Solution:** Full rebuild
```bash
./rebuild.sh full
source install/setup.bash
```

### "Colcon command not found"

**Solution:** Source ROS2
```bash
source /opt/ros/jazzy/setup.bash  # or humble, galactic
./rebuild.sh
```

---

## 📋 Rebuild Checklist

After pulling updates from git:

```bash
# 1. Pull changes
git pull

# 2. Rebuild
./rebuild.sh

# 3. Source (in EACH terminal)
source install/setup.bash

# 4. Verify
ros2 pkg list | grep pidog  # Should show pidog packages

# 5. Test
ros2 launch pidog_gaits collect_data.launch.py
```

---

## 🎯 Quick Commands

```bash
# Quick rebuild after git pull
git pull && ./rebuild.sh && source install/setup.bash

# Full clean rebuild
./rebuild.sh full

# Check what's installed
ls -l install/*/share/*/

# Verify package
ros2 pkg list | grep pidog
ros2 pkg prefix pidog_gaits
```

---

## 💡 Pro Tips

### 1. Auto-source on Terminal Start

Add to `~/.bashrc`:
```bash
source /home/user/pidog_ros2/install/setup.bash
```

Then you don't need to source manually each time!

### 2. Rebuild Alias

Add to `~/.bashrc`:
```bash
alias rebuild='cd /home/user/pidog_ros2 && ./rebuild.sh && source install/setup.bash'
```

Then just type: `rebuild`

### 3. Quick Check

After rebuild, verify with:
```bash
ros2 node list  # While nodes are running
ros2 param get /gait_generator default_gait  # Should show 'stand'
```

---

## 🐛 Troubleshooting

### Rebuild fails with errors

```bash
# Check Python syntax
python3 -m py_compile pidog_gaits/pidog_gaits/*.py

# Check for missing dependencies
rosdep install --from-paths src --ignore-src -r -y
```

### Still using old code after rebuild

```bash
# Nuclear option - clean everything
rm -rf build/ install/ log/
colcon build
source install/setup.bash
```

### Permission denied

```bash
chmod +x rebuild.sh
```

---

## Summary

| Command | Use Case | Time |
|---------|----------|------|
| `./rebuild.sh` | After git pull | ~30s |
| `./rebuild.sh full` | When things break | ~2-3min |
| Manual commands | Fine-grained control | Varies |

**Most common workflow:**
```bash
git pull
./rebuild.sh
source install/setup.bash
```

Done! 🎉
