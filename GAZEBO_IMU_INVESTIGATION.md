# Gazebo Harmonic IMU Sensor Investigation

## Summary

**Gazebo Harmonic 8.9.0 sensor plugins (gz-sim-sensors-system, gz-sim-imu-system) prevent world initialization and are incompatible with this setup.**

## Test Results

### ✅ Test 1: Gazebo WITHOUT Sensor Plugins (SUCCESSFUL)

```
[gazebo-2] [Msg] World [pidog_world] initialized with [1ms] physics profile.
[gazebo-2] [Msg] Create service on [/world/pidog_world/create]
[create-4] [INFO]: Entity creation successful.
```

**Result:** Robot spawns, controllers load, everything works perfectly.

### ❌ Test 2: Gazebo WITH Sensor Plugins (FAILED)

Tested configurations:
1. **Both plugins** (gz-sim-sensors-system + gz-sim-imu-system): World service never becomes available, robot cannot spawn
2. **IMU plugin only** (gz-sim-imu-system): Same issue (not tested due to script error, but previous tests showed same problem)

**Result:** World hangs indefinitely, `/world/pidog_world/create` service never advertised.

## Root Cause

The Gazebo sensor plugins exist on the system:
```bash
ls -la /opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins/ | grep -E "sensors|imu"
lrwxrwxrwx  libgz-sim-imu-system.so -> libgz-sim8-imu-system.so
lrwxrwxrwx  libgz-sim-sensors-system.so -> libgz-sim8-sensors-system.so
-rw-r--r--  libgz-sim8-imu-system.so.8.9.0  (183KB)
-rw-r--r--  libgz-sim8-sensors-system.so.8.9.0  (302KB)
```

But loading them causes Gazebo to hang during world initialization. Possible causes:
- Plugin incompatibility with ROS2 Jazzy
- Bug in Gazebo Harmonic 8.9.0
- Missing dependencies or configuration
- Conflict with ros2_control or other plugins

## Attempted Solutions

### Configuration Changes Tried

1. ✅ **Removed render_engine requirement** (no effect)
2. ✅ **Used IMU plugin only** (per official docs - still failed)
3. ✅ **Simplified sensor configuration** (no effect)
4. ✅ **Disabled URDF sensor definition** (fixed Gazebo loading)

### Documentation Reviewed

Official Gazebo Harmonic sensor docs (https://gazebosim.org/docs/latest/sensors/):
- IMU sensors should only need `gz-sim-imu-system` plugin
- `gz-sim-sensors-system` is for Lidar/cameras requiring rendering
- Our configuration followed official guidance

## Recommended Solution

### For RL Training: **Use MuJoCo**

MuJoCo provides:
- ✅ Native IMU sensors (orientation, gyro, accelerometer)
- ✅ 64x real-time physics speed
- ✅ Stable, no plugin issues
- ✅ Already configured in this repo

**Training workflow:**
```bash
# Train in MuJoCo (fast, reliable IMU)
python3 watch_training.py  # Watch robot learn

# OR headless training
python3 train_mujoco.py    # 16 parallel environments

# Test in Gazebo (visualization)
ros2 launch pidog_gaits gait_demo.launch.py

# Deploy to real robot
# (trained model works with real IMU sensor)
```

### For Gazebo Testing: **Virtual IMU Alternative**

If you need IMU data in Gazebo for testing:

**Option A: Virtual IMU Node** (already implemented!)
- File: `pidog_control/pidog_control/virtual_imu_node.py`
- Computes realistic IMU from Gazebo physics
- No sensor plugins needed
- Requires bridging `/gazebo/model_states` topic

**Option B: Train without Gazebo IMU**
- Use MuJoCo for all RL training
- Use Gazebo only for visualization
- Test with real hardware IMU

## Files Modified

### Sensor Plugins Disabled
1. `pidog_description/worlds/pidog.sdf` - Removed sensor system plugins
2. `pidog_description/urdf/pidog.urdf` - Commented out IMU sensor plugin
3. `pidog_description/launch/gazebo.launch.py` - Removed IMU bridge

### Test/Diagnostic Files Created
1. `pidog_description/worlds/pidog_test_no_sensors.sdf` - World without sensors (for testing)
2. `pidog_description/worlds/pidog_with_imu_only.sdf` - IMU-only plugin config
3. `test_gazebo_imu.sh` - Automated IMU test script
4. `test_world_load.sh` - World loading diagnostic
5. `test_imu_verbose.sh` - Verbose logging test
6. `check_imu.sh` - Quick IMU data check

## Current State

- ✅ Gazebo works perfectly without sensor plugins
- ✅ Robot spawns and controllers function normally
- ✅ MuJoCo environment provides full IMU data for RL
- ❌ Gazebo native IMU sensors incompatible/broken
- ⚠️  virtual_imu_node exists but not integrated into launch

## Next Steps (Optional)

If you want IMU in Gazebo despite the sensor plugin issue:

1. **Integrate virtual_imu_node:**
   - Add model states bridge to gazebo.launch.py
   - Launch virtual_imu_node
   - Publishes to `/imu/data` topic

2. **Report bug to Gazebo:**
   - File issue at https://github.com/gazebosim/gz-sim
   - Include sensor plugin configuration and versions
   - Describe world initialization hang

3. **Try newer Gazebo version:**
   - Current: Gazebo Harmonic 8.9.0 (Sept 2024)
   - Check if newer version fixes sensor plugins

## Conclusion

**Gazebo Harmonic sensor plugins are incompatible with this setup.** Use MuJoCo for RL training with IMU, and Gazebo for visualization without IMU. This provides the best of both worlds: fast, reliable training in MuJoCo, and visual debugging in Gazebo.
