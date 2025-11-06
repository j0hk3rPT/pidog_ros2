# PiDog Walking Instructions

## What Was Fixed

### Issue 1: Missing Walk Controller Node
The `gazebo_manual.launch.py` file was missing the `simple_walk_gazebo` node, which is responsible for generating walking gaits. This node has now been added to the launch file.

### Issue 2: Incorrect Shoulder Movement (Bug Fix)
The walking gait had a critical bug where all shoulder joints were moving in sync, causing the robot to just bounce up and down without any forward motion. The shoulder movements have been corrected to create proper push/pull motion:
- When a leg is on the ground: shoulder swings backward (power stroke, pushes robot forward)
- When a leg is lifting: shoulder swings forward (recovery stroke, prepares for next step)

## How to Use

### 1. Rebuild the workspace
```bash
rm -rf build/ install/ log/
colcon build
source install/setup.bash
```

### 2. Launch Gazebo
```bash
ros2 launch pidog_description gazebo_manual.launch.py
```

The robot should now spawn in Gazebo with the `simple_walk_gazebo` node running. You should see a log message:
```
Simple Walk Node started
Send "walk" to /gait_command to start walking
Send "stand" to /gait_command to stop
```

### 3. Make the robot walk

In a **new terminal**, source the workspace and send commands:

```bash
cd /home/user/pidog_ros2
source install/setup.bash

# Start walking
ros2 topic pub --once /gait_command std_msgs/msg/String "data: 'walk'"

# Stop walking (return to standing pose)
ros2 topic pub --once /gait_command std_msgs/msg/String "data: 'stand'"
```

**OR** use the helper script:
```bash
# Start walking
python3 send_gait_command.py walk

# Stop walking
python3 send_gait_command.py stand
```

## Available Commands

- `walk` or `walk_forward` - Start walking
- `stand` or `stop` - Stop walking and return to standing pose

## How It Works

1. **simple_walk_gazebo** node listens on `/gait_command` topic
2. When it receives "walk", it generates a trotting gait pattern
3. Joint positions are published to `/position_controller/commands`
4. The position_controller moves the robot's joints in Gazebo

## Gait Parameters

The walking gait uses:
- **Frequency**: 0.5 Hz (slow, stable walk)
- **Step height**: 0.6 radians (knee lift)
- **Shoulder swing**: 0.5 radians (forward/back motion)

These can be adjusted in `pidog_control/pidog_control/simple_walk_gazebo.py` if needed.
