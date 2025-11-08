# PiDog Data Collection Guide

## Quick Start

### Step 1: Launch Simulation
```bash
# In terminal 1 (inside Docker container):
cd /home/user/pidog_ros2
source install/setup.bash
ros2 launch pidog_gaits collect_data_enhanced.launch.py
```

**Expected behavior:**
- Gazebo window opens with the PiDog robot
- Robot is **standing still** in a neutral pose (this is correct!)
- Controller logs show "Publishing 12 joint values..."
- **The robot will NOT walk automatically - you need to send commands**

### Step 2: Send Gait Commands

You have two options:

#### Option A: Automated Data Collection (Recommended)
```bash
# In terminal 2 (inside Docker container):
cd /home/user/pidog_ros2
source install/setup.bash
./collect_training_data.sh 20
```

This will:
- Cycle through all gaits (walk_forward, walk_backward, trot_forward, sit, etc.)
- Record 20 seconds of data per gait
- Automatically reset the robot position between gaits
- Display progress bars

**Press Ctrl+C** in terminal 1 when done to save the data.

#### Option B: Manual Gait Commands
```bash
# In terminal 2 (inside Docker container):
source install/setup.bash

# Try different gaits:
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'sit'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'stand'" --once
```

## Understanding the System

### Launch Components

`collect_data_enhanced.launch.py` launches three things:

1. **Gazebo** - Physics simulation
2. **gait_generator** - Computes joint angles for each gait
   - Starts with `default_gait: 'stand'` (static pose)
   - Listens to `/gait_command` topic to switch gaits
   - Publishes to `/motor_pos` (12 joint angles)
3. **data_collector_enhanced** - Records training data
   - Subscribes to `/motor_pos` (passively records)
   - Adds observation noise for robust training
   - Saves data on Ctrl+C

### Data Flow

```
/gait_command (String)
    ↓
gait_generator
    ↓
/motor_pos (JointState with 12 angles)
    ↓
pidog_gazebo_controller
    ↓
/position_controller/commands (Float64MultiArray)
    ↓
ros2_control → Gazebo joints
```

**data_collector_enhanced** just listens to `/motor_pos` and records - it doesn't interfere with movement.

## Troubleshooting

### "Robot is not moving at all"

**This is expected!** The robot starts in 'stand' pose (static). To make it move:

1. **Check if the robot is standing properly:**
   - Body should be upright
   - All 4 legs should be on the ground
   - No joints should be collapsed

2. **Send a gait command:**
   ```bash
   ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
   ```

3. **If robot is collapsed/fallen:**
   - Check for controller errors in terminal 1
   - Run: `ros2 control list_controllers`
   - Verify position_controller is `[active]`

### "Controller deactivation error during startup"

You may see this during the first few seconds:
```
[ERROR] [controller_manager]: Deactivating controllers : [ position_controller ] as their update resulted in an error!
```

**This is usually harmless** if it only appears briefly during startup. The controller should recover within 3 seconds.

If the error persists:
1. Stop the launch (Ctrl+C)
2. Rebuild: `cd /home/user/pidog_ros2 && colcon build`
3. Clean Python cache: `find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null`
4. Source and relaunch: `source install/setup.bash && ros2 launch pidog_gaits collect_data_enhanced.launch.py`

### Verify System Status

Run this diagnostic script:
```bash
./check_robot_status.sh
```

This will check:
- Is Gazebo running?
- Are controllers loaded?
- Is gait_generator publishing?
- Is controller receiving commands?

### Check Joint Commands

```bash
# Monitor gait generator output (should show 12 angles):
ros2 topic echo /motor_pos

# Monitor controller commands (should show 12 values):
ros2 topic echo /position_controller/commands

# Check joint states from Gazebo:
ros2 topic echo /joint_states
```

## Available Gaits

### Dynamic Gaits
- `walk_forward` - Sequential leg walking
- `walk_backward` - Reverse walk
- `walk_left` - Turn while walking
- `walk_right` - Turn while walking
- `trot_forward` - Diagonal leg trotting (faster)
- `trot_backward` - Reverse trot
- `trot_left` - Turn while trotting
- `trot_right` - Turn while trotting

### Static Poses
- `stand` - Neutral standing pose
- `sit` - Sitting pose (rear up, front down)
- `lie` - Lying flat
- `stretch` - Stretching pose

## Output Data

Data is saved to: `./training_data/gait_data_enhanced_YYYYMMDD_HHMMSS.{json,npz}`

**Format:**
- **Input features:** [gait_type, direction, turn, phase]
  - gait_type: 0=walk, 1=trot, 2=static_pose
  - direction: -1=backward, 0=none, 1=forward
  - turn: -1=left, 0=straight, 1=right
  - phase: 0.0 to 1.0 (gait cycle position)
- **Output:** 12 joint angles (8 legs + 1 tail + 3 head/neck) with observation noise
- **Velocities:** 12 joint velocities (optional, for LSTM training)

**Observation noise:**
- Position: σ=0.01 rad (~0.57° sensor error)
- Velocity: σ=0.1 rad/s

This noise improves sim-to-real transfer for hardware deployment.

## Next Steps

After collecting data:

1. **Stop data collection:** Press Ctrl+C in terminal 1
2. **Verify data:** `ls -lh ./training_data/`
3. **Train neural network:** `./train_gpu.sh`
4. **Test trained model:** `ros2 launch pidog_gaits nn_demo.launch.py`

## Common Misconceptions

❌ **"collect_data_enhanced.launch.py should make the robot walk automatically"**
✅ **Reality:** The launch file starts the system with robot in 'stand' pose. You must send gait commands to make it walk.

❌ **"The data collector controls the robot"**
✅ **Reality:** data_collector_enhanced is a passive observer - it only records joint states, it doesn't send commands.

❌ **"The robot should be doing something complex immediately"**
✅ **Reality:** The robot starts in a simple standing pose for safety and stability. This is the correct behavior.

## Architecture Summary

```
┌─────────────────────────────────────────┐
│  collect_data_enhanced.launch.py        │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┬──────────────────┐
    │                 │                  │
┌───▼────┐   ┌────────▼──────┐   ┌──────▼────────────┐
│ Gazebo │   │ gait_generator│   │ data_collector    │
│  (sim) │   │  (commands)   │   │  (recorder only)  │
└────────┘   └───────────────┘   └───────────────────┘
```

The robot is **fully functional with all 12 joints** (8 legs + 1 tail + 3 head/neck) for realistic balancing during locomotion.
