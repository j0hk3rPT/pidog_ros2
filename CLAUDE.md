# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

PiDog ROS2 is a quadruped robot simulation and control system with neural network-based gait learning.

**Key Components**:
- Traditional gait generators (walk, trot) for quadruped locomotion
- Gazebo simulation environment with ros2_control
- Neural network training pipeline for learning gaits from demonstrations
- Data collection system for imitation learning

**Hardware**: 8 leg joints (2 per leg: shoulder/hip and knee) plus optional head(3)/tail(1) servos

## Docker Setup

Two container workflows for different tasks:

### ROS2 Container (Simulation & Data Collection)
```bash
# On host: Allow X11 access for GUI
xhost +local:docker

# Start and attach to container
docker start pidog_ros2
docker exec -it pidog_ros2 /bin/bash

# Inside container: Setup GPU and build workspace
./setup_rendering.sh  # AMD GPU setup
cd /home/user/pidog_ros2
colcon build
source install/setup.bash
```

### Training Container (GPU-Accelerated Training)
```bash
# AMD ROCm GPU
docker-compose --profile rocm run pidog-rocm

# Inside training container:
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model simple --epochs 100 --device auto
```

**Note**: Source code is mounted at `/workspace` (training container) or `/home/user/pidog_ros2` (ROS2 container).

## Build and Development Commands

### Workspace Build
```bash
./rebuild.sh
# Source workspace after build
source install/setup.bash
```

## Launch Commands

### Launch Commands (Choose ONE)

**IMPORTANT**: Each launch file below includes Gazebo + controllers. Only run ONE at a time.

#### 1. Traditional Gait Demo (Recommended for testing)
```bash
# Launch Gazebo + gait generator (all-in-one)
ros2 launch pidog_gaits gait_demo.launch.py

# Send gait commands (in another terminal)
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'sit'" --once
```

**Available Gaits**:
- Walk: walk_forward, walk_backward, walk_left, walk_right
- Trot: trot_forward, trot_backward, trot_left, trot_right
- Static: stand, sit, lie, stretch

#### 2. Manual/Testing (Gazebo only, no gait generator)
```bash
# Launch Gazebo with basic standing controller
ros2 launch pidog_description gazebo.launch.py
# Robot will stand in place, controlled by pidog_gazebo_controller
```

#### 3. Neural Network Demo
```bash
# Launch Gazebo + trained neural network controller
ros2 launch pidog_gaits nn_demo.launch.py

# Send gait commands same as traditional gaits
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
```

#### 4. Data Collection
```bash
# Launch Gazebo + gait generator + data collector
ros2 launch pidog_gaits collect_data.launch.py
# Automatically cycles through gaits and saves training data
```

### Neural Network Workflow

#### 1. Collect Training Data
```bash
# Enhanced data collection (RECOMMENDED for hardware deployment)
ros2 launch pidog_gaits collect_data_enhanced.launch.py
./collect_training_data.sh 20  # 20 seconds per gait

# Basic data collection (for testing only)
ros2 launch pidog_gaits collect_data.launch.py

# Output: ./training_data/gait_data_[enhanced_]YYYYMMDD_HHMMSS.{json,npz}
# Enhanced includes noise: position σ=0.01 rad, velocity σ=0.1 rad/s
```

#### 2. Train Neural Network
```bash
# Quick training (auto-detect GPU)
./train_gpu.sh

# Manual training with options
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model simple \
    --epochs 200 \
    --batch_size 256 \
    --device auto \
    --save_dir ./models

# Output: ./models/best_model.pth, ./models/training_history.png
```

#### 3. Deploy Trained Model
```bash
# Launch neural network controller
ros2 launch pidog_gaits nn_demo.launch.py

# Test learned behaviors
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
```

## Architecture

### Package Structure

**pidog_description**: Robot URDF models, Gazebo worlds, and ros2_control configuration
- `urdf/pidog.urdf` - Robot description with physics parameters
- `config/pidog_controllers.yaml` - ros2_control PID controller configuration
- `launch/gazebo.launch.py` - Main Gazebo launch file
- `worlds/pidog.sdf` - Custom world with optimized physics

**pidog_control**: Joint control nodes for Gazebo
- `pidog_gazebo_controller.py` - Publishes to `/position_controller/commands` (ros2_control)
- `simple_walk_gazebo.py` - Simple walking controller for testing

**pidog_gaits**: Gait generation and neural network learning
- `gait_generator_node.py` - Traditional gait controller (publishes to `/motor_pos`)
- `inverse_kinematics.py` - Converts leg coordinates to joint angles
- `walk_gait.py` - Sequential leg movement gait
- `trot_gait.py` - Diagonal leg pairing gait
- `data_collector.py` - Records gait data for training
- `neural_network.py` - PyTorch model architectures
- `train.py` - Training script with visualization
- `nn_controller.py` - Neural network inference node

### Key Topics

- `/motor_pos` (sensor_msgs/JointState) - Motor positions from gait generator
- `/position_controller/commands` (std_msgs/Float64MultiArray) - ros2_control commands
- `/gait_command` (std_msgs/String) - Gait selection commands
- `/joint_states` - Published by joint_state_broadcaster (ros2_control)

### Control Flow

**Gazebo Pipeline**:
1. `gazebo.launch.py` spawns robot with ros2_control plugins
2. Controller manager loads `position_controller` and `joint_state_broadcaster`
3. Gait generator publishes to `/motor_pos`
4. Controller node reads `/motor_pos` and publishes to `/position_controller/commands`

**Neural Network Pipeline**:
1. Collect data: Gait generator runs, data_collector records (input features + joint angles)
2. Train: PyTorch model learns mapping [gait_type, direction, turn, phase] → [8 joint angles]
3. Deploy: nn_controller replaces gait generator, publishes learned angles to `/motor_pos`

### Joint Naming Convention

**Gazebo URDF joints** (ros2_control):
- body_to_back_right_leg_b, back_right_leg_b_to_a
- body_to_front_right_leg_b, front_right_leg_b_to_a
- body_to_back_left_leg_b, back_left_leg_b_to_a
- body_to_front_left_leg_b, front_left_leg_b_to_a

**Gait generator motor indices**:
- motor_0 to motor_7: leg joints (2 per leg)
- motor_8 to motor_11: head/tail (currently disabled in Gazebo)

**Important**: Left leg joints have flipped axes (rpy="0 1.57 3.1415" in URDF):
- Right legs: negative angle bends knee DOWN
- Left legs: positive angle bends knee DOWN

## Neural Network Details

**Input features** (4D for basic models):
- gait_type: 0=walk, 1=trot, 2=static_pose
- direction: -1=backward, 0=none, 1=forward
- turn: -1=left, 0=straight, 1=right
- phase: 0.0 to 1.0 (gait cycle position)

**Output**: 12 joint angles (radians) for all motors (8 legs + 4 head/tail)

**Available Models**:

1. **GaitNet (Simple)** - Basic MLP (~200K params)
   - Input(4) → Dense(128) → Dense(256) → Dense(128) → Dense(12)
   - Good for: Initial testing, fast training

2. **GaitNetLarge** - Larger MLP (~1M params)
   - Input(4) → Dense(256) → Dense(512) → Dense(512) → Dense(256) → Dense(12)
   - Good for: When simple model underfits

3. **GaitNetSimpleLSTM** - LSTM with temporal memory (~13K params) ⭐ **RECOMMENDED**
   - Input(4) → LSTM(64) → Dense(32) → Dense(12)
   - Output: All 12 motors (8 legs + 4 head/tail for balance)
   - Good for: Sim-to-real transfer, handles servo lag
   - **Best choice for hardware deployment**

4. **GaitNetLSTM** - Full state feedback LSTM (~35K params)
   - Input(28) → LSTM(128) → Dense(64) → Dense(12)
   - Input = [gait_cmd(4), joint_pos(12), joint_vel(12)]
   - Output: All 12 motors (8 legs + 4 head/tail for balance)
   - Good for: Advanced control, closed-loop feedback

**Training expectations**:
- 10,000+ samples recommended
- Validation loss < 0.01 indicates good learning
- Converges in 50-100 epochs typically
- LSTM models may need 100-200 epochs

## Physics Configuration

**Current Tuned Parameters** (realistic values matching hardware for sim-to-real transfer):

**Servo Specifications** (Real Hardware - SunFounder SF006FM 9g Digital Servo):
- Torque: 0.127-0.137 Nm (1.3-1.4 kgf·cm at 4.8-6V)
- Speed: 333-400°/s (5.8-7.0 rad/s)
- Operating voltage: 4.8-6.0V
- Range: 0-180°

**Joint Parameters** (Simulation):
- Leg joints (8): effort=0.15 Nm, velocity=7.0 rad/s (400°/s), damping=0.5, friction=0.5
  - **Matches real servo torque for accurate sim-to-real transfer**
- Neck joints (3): effort=0.14 Nm, velocity=7.5 rad/s, damping=0.3, friction=0.5, stiffness=50.0
- Tail joint (1): effort=0.15 Nm, velocity=7.0 rad/s, damping=0.5, friction=0.5

**Contact Physics** (Gazebo ODE):
- Foot contact: kp=1e6, kd=100, mu1=0.8 (soft contact prevents bouncing)
- Ground: kp=1e7, kd=100, mu=1.0
- ODE Solver: type="world" (Dantzig direct solver)
- ODE Constraints: CFM=0.0 (stiff contacts), ERP=0.2
- Solver iterations: 150

**PID Controller Gains** (50 Hz update rate):
- P=8.0 (strong tracking without saturation)
- D=2.5 (good damping prevents oscillation)
- I=0.1 (small integral for steady-state correction)
- position_proportional_gain=1.0 (no amplification)

**Design Philosophy**:
- **Realistic servo limits** (0.15 Nm) → matches real hardware for sim-to-real transfer
- **Soft contacts** (kp=1e6 vs 1e15) → prevents bouncing/vibration
- **Balanced PID** (P=8, D=2.5) → good tracking with damping

**Troubleshooting**:
- **Bouncing/vibration**: Reduce contact kp in URDF (currently 1e6)
- **Weak/frozen joints**: Increase effort limits or PID P gain
- **Asymmetry/wrong joint control**: Check for duplicate ros2_control blocks in URDF
- **Motor index issues**: Verify inverse_kinematics.py outputs [BR, FR, BL, FL] order
- **Oscillation**: Increase PID D gain or reduce P gain

**Common Issues Fixed** (based on SunFounder reference implementation):
- ⚠️ **Duplicate ros2_control blocks**: URDF must have only ONE ros2_control block (using gz_ros2_control-system for Gazebo Harmonic)
- ⚠️ **Motor mapping**: inverse_kinematics.py outputs [BR, FR, BL, FL] order to match controller config
- ⚠️ **IK transformations** (from SunFounder):
  - `alpha = angle2 + angle1` (NO shoulder offset subtraction!)
  - `foot_angle = beta - π/2` (NOT `π/2 - beta`!)
  - Negate RIGHT legs (odd indices), not left legs
- ⚠️ **Pose consistency**: ALL poses (stand, sit, lie, stretch) generated from IK using [y, z] coordinates
- ⚠️ **Walking from stand**: Stand pose uses z=80mm (same as Z_ORIGIN in walk_gait.py)
- ⚠️ **Gait parameters**: Walk/Trot match SunFounder exactly (LEG_STEP_HEIGHT=20mm, LEG_STEP_WIDTH=80mm, Z_ORIGIN=80mm)

## Common Workflows

### Adding a New Gait
1. Create gait generator in `pidog_gaits/pidog_gaits/` (e.g., `run_gait.py`)
2. Add to `gait_generator_node.py` in `_generate_all_gaits()`
3. Rebuild: `./rebuild.sh`
4. Test: `ros2 topic pub /gait_command std_msgs/msg/String "data: 'run_forward'" --once`
5. Collect data with new gait: `./collect_training_data.sh`
6. Retrain neural network: `./train_gpu.sh`

### Debugging Simulation Issues
```bash
# Quick diagnostics (run from workspace root)
./debug_topics.sh

# Manual checks:
# Check if Gazebo is running
gz sim -l

# Monitor joint states
ros2 topic echo /joint_states

# Check controller status
ros2 control list_controllers

# View motor positions from gait generator
ros2 topic echo /motor_pos

# Check ros2_control commands
ros2 topic echo /position_controller/commands
```

**Control Flow**:
```
gait_generator → /motor_pos → pidog_gazebo_controller → /position_controller/commands → ros2_control → Gazebo joints
```

### Tuning Gait Parameters
Edit parameters in `pidog_gaits/pidog_gaits/walk_gait.py` and `trot_gait.py`:
- `LEG_STEP_HEIGHT` - Height of leg lift during step (mm)
- `LEG_STEP_WIDTH` - Forward/backward step distance (mm)
- `STEP_COUNT` - Number of steps per section (affects gait speed/smoothness)
  - Walk: 6 steps × 8 sections = 48 frames (~1.6s per cycle at 30Hz)
  - Trot: 10 steps × 2 sections = 20 frames (~0.67s per cycle at 30Hz)

After editing, rebuild and test:
```bash
./rebuild.sh
ros2 launch pidog_gaits gait_demo.launch.py
```

## Dependencies

**ROS2**: Jazzy (in Docker)
- rclpy
- ros_gz_sim, ros_gz_bridge
- controller_manager, joint_state_broadcaster, position_controllers

**Python**:
- torch, torchvision
- numpy
- matplotlib

**GPU Support**: AMD ROCm (optional CUDA/CPU)

## Sim-to-Real Transfer

**Enhanced Data Collection** (recommended for hardware deployment):
- Use `collect_data_enhanced.launch.py` for noise-augmented training data
- Position observation noise: σ=0.01 rad (~0.57° sensor error)
- Velocity observation noise: σ=0.1 rad/s
- Train with **GaitNetSimpleLSTM** model for best sim-to-real transfer

**Physics parameters** are detailed in the **Physics Configuration** section above.

## Path Conventions

**Workspace Location**: `/home/user/pidog_ros2` (ROS2 container) or `/workspace` (training container)

All relative paths (`./training_data`, `./models`, etc.) are relative to the workspace root.
