# Debugging: Robot Not Moving in Gazebo

## Problem
PiDog spawns in Gazebo but shows **ZERO movement** when walk commands are sent.

## Investigation Summary

### What We Know Works
✅ Gazebo launches successfully
✅ Robot spawns correctly
✅ Controllers load (position_controller, joint_state_broadcaster)
✅ simple_walk_gazebo node starts and receives commands
✅ Logs show "🐕 Starting walk!" when commands are sent

### What's NOT Working
❌ Robot doesn't move at all (not even bouncing)
❌ No visible joint movement in Gazebo or RViz

## Potential Root Causes

### 1. Command Publisher Conflict
**Issue:** Multiple nodes publishing to `/position_controller/commands` simultaneously could conflict.

**Check:** Make sure only ONE node publishes to the position controller topic.
- `gazebo.launch.py` uses `pidog_gazebo_controller` (publishes standing pose continuously)
- `gazebo_manual.launch.py` uses `simple_walk_gazebo` (publishes walk/stand poses)

**Solution:** Use `gazebo_manual.launch.py` to avoid conflicts.

### 2. Controller Not Receiving Commands
**Possible causes:**
- Commands published before controller is ready
- Topic name mismatch
- Message format incorrect

**Test:** Run `debug_topics.py` to monitor:
```bash
python3 debug_topics.py
```

This shows:
- Commands being published to `/position_controller/commands`
- Actual joint states from `/joint_states`

### 3. Physics/Controller Gains Too Low
**Issue:** `position_proportional_gain: 2` might be too weak to overcome friction/gravity.

**Location:** `pidog_description/config/pidog_controllers.yaml`

Current PID gains:
```yaml
gains:
  body_to_back_right_leg_b: {p: 1000.0, d: 50.0, i: 10.0, i_clamp: 10.0}
```

These seem reasonable, but the `position_proportional_gain: 2` in the hardware interface might be limiting actual torque.

### 4. Joint Movement Values Incorrect
**Issue:** The hardcoded radian values in `simple_walk_gazebo.py` might not create meaningful movement.

**Comparison with existing code:**
- `gait_generator_node.py` uses inverse kinematics to convert Y,Z coordinates (mm) to joint angles
- `simple_walk_gazebo.py` uses hardcoded radian values without IK calculations

**Standing pose:** `[0.0, -0.8, 0.0, -0.8, 0.0, -0.8, 0.0, -0.8]`
- Shoulders: 0.0 rad (straight)
- Knees: -0.8 rad (≈ -46°, bent)

**Walk values should vary** but might be too similar to standing to see movement.

### 5. URDF/Physics Configuration
**Check:**
- Joint limits in URDF
- Friction coefficients
- Mass/inertia values
- Ground contact physics

## Testing Approaches

### Test 1: Ultra-Simple Movement (simple_walk_v2)
Uses very slow (2Hz), very obvious movements:
- Alternates lifting left vs right legs
- Knee goes from -0.8 (bent) to -0.3 (extended)
- Logs each step clearly

```bash
# Update launch file to use simple_walk_v2, then:
colcon build --packages-select pidog_control
source install/setup.bash
ros2 launch pidog_description gazebo_manual.launch.py

# In another terminal:
python3 send_gait_command.py walk
```

Watch for log messages: "Step X: LEFT LEGS UP" / "Step X: RIGHT LEGS UP"
**If still no movement:** Problem is in controller/physics, not gait algorithm.

### Test 2: Single Joint Movement
Tests if ANY joint can move:

```bash
python3 test_single_joint.py
```

Alternates all shoulders between +0.5 and -0.5 every 2 seconds.
**If no movement:** Controller or physics issue, not gait-specific.

### Test 3: Monitor Topics
See what's actually being published:

```bash
# Terminal 1: Monitor commands
ros2 topic echo /position_controller/commands

# Terminal 2: Monitor joint states
ros2 topic echo /joint_states

# Terminal 3: Send walk command
python3 send_gait_command.py walk
```

**Expected:** Commands topic should show changing values.
**If commands don't change:** Gait generator problem.
**If commands change but joints don't:** Controller/physics problem.

### Test 4: Manual Command
Bypass gait generator entirely:

```bash
ros2 topic pub --once /position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]}"
```

**If movement occurs:** Gait generator has a bug.
**If no movement:** Controller or physics issue.

## Next Steps Based on Tests

### If Test 4 (manual command) works:
→ Problem is in gait generation logic
→ Fix: Use proper inverse kinematics from existing Walk/Trot classes

### If Test 4 doesn't work:
→ Problem is in controller or Gazebo physics
→ Check:
  1. Controller manager status: `ros2 control list_controllers`
  2. Hardware interface: Check gz_ros2_control plugin in URDF
  3. Physics timestep: Current is 1ms, might need adjustment
  4. PID gains: Try increasing position_proportional_gain

### If joints move in RViz but not Gazebo:
→ Physics simulation issue
→ Check ground contact, friction, mass properties

### If joints don't move anywhere:
→ Controller not receiving/processing commands
→ Check ros2_control configuration in URDF and controller YAML

## Reference: Working Gait Implementation

The existing `gait_generator_node.py` uses:
1. **Coordinate generation:** `Walk.get_coords()` generates [y, z] positions in mm
2. **Inverse kinematics:** `LegIK.legs_coords_to_angles()` converts to joint angles
3. **Topic:** Publishes to `/motor_pos` (different topic!)
4. **Bridge:** `pidog_gazebo_controller` subscribes to `/motor_pos` and republishes to `/position_controller/commands`

This architecture might be the intended design - the walk gait should publish to `/motor_pos`, not directly to `/position_controller/commands`.

## Quick Fix to Try

Change `simple_walk_gazebo.py` to publish to `/motor_pos` instead and ensure `pidog_gazebo_controller` is running to bridge the commands.

OR

Use the existing gait system:
```bash
ros2 launch pidog_description gazebo.launch.py

# In another terminal:
ros2 run pidog_gaits gait_generator_node

# Send walk command:
ros2 topic pub --once /gait_command std_msgs/msg/String "data: 'walk_forward'"
```
