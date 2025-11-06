# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PiDog ROS2 is a quadruped robot simulation and control system with neural network-based gait learning. It consists of:
- Traditional gait generators (walk, trot) for quadruped locomotion
- Gazebo simulation environment
- ros2_control integration for joint control
- Neural network training pipeline to learn gaits from demonstrations
- Data collection system for imitation learning

The robot has 8 leg joints (2 per leg: shoulder/hip and knee) plus optional head/tail servos.

## Docker Setup

Two separate container workflows:

**ROS2 Container** (for simulation and data collection):
```bash
# On host: Allow X11 access
xhost +local:docker

# Start/attach to ROS2 container
docker start pidog_ros2  # or use the docker run command if first time
docker exec -it pidog_ros2 /bin/bash

# Inside container: Setup GPU and rebuild
./setup_rendering.sh  # AMD GPU setup
cd /home/user/pidog_ros2
colcon build
source install/setup.bash
```

**Training Container** (for GPU-accelerated training):
```bash
# Using docker-compose with ROCm profile for AMD GPU
docker-compose --profile rocm run pidog-rocm

# Inside training container:
python3 -m pidog_gaits.pidog_gaits.train --data ./training_data/gait_data_*.npz --model simple --epochs 100 --device auto
```

Note: Source code is mounted at `/workspace` in training container, `/home/user/pidog_ros2` in ROS2 container.

## Build and Development Commands

### Workspace Build
```bash
# Full rebuild (removes all build artifacts)
./rebuild.sh full

# Quick rebuild (only modified packages: pidog_gaits, pidog_sim)
./rebuild.sh

# Build specific package
colcon build --packages-select pidog_gaits
colcon build --packages-select pidog_description

# Source workspace after build
source install/setup.bash
```

### Running Tests
```bash
# Run Python linters
ament_flake8 pidog_gaits/
ament_pep257 pidog_gaits/

# Run pytest (if tests exist)
pytest pidog_gaits/test/
```

## Launch Commands

### Gazebo Simulation
```bash
# Launch Gazebo with PiDog robot and ros2_control
ros2 launch pidog_description gazebo.launch.py

# Launch with manual control
ros2 launch pidog_description gazebo_manual.launch.py

# Send gait commands (in another terminal)
python3 send_gait_command.py walk
python3 send_gait_command.py stand
```

### Traditional Gait Demo
```bash
# Launch with traditional gait generator (stand pose by default)
ros2 launch pidog_gaits gait_demo.launch.py

# Send gait commands (in another terminal)
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'sit'" --once
```

**Available gaits**: walk_forward, walk_backward, walk_left, walk_right, trot_forward, trot_backward, trot_left, trot_right, stand, sit, lie, stretch

### Neural Network Workflow

#### 1. Collect Training Data

**Basic (Original)**:
```bash
# Standard data collection
ros2 launch pidog_gaits collect_data.launch.py

# Use automated collection script
./collect_training_data.sh 20  # 20 seconds per gait

# Data saved to: ./training_data/gait_data_YYYYMMDD_HHMMSS.{json,npz}
```

**Enhanced (with observation noise - RECOMMENDED)**:
```bash
# Enhanced data collection with noise for robustness
ros2 launch pidog_gaits collect_data_enhanced.launch.py

# Use automated collection script
./collect_training_data.sh 20

# Data saved to: ./training_data/gait_data_enhanced_YYYYMMDD_HHMMSS.{json,npz}
# Includes: position noise (σ=0.01 rad), velocity noise (σ=0.1 rad/s)
```

#### 2. Train Neural Network
```bash
# Auto-detect GPU and train (recommended)
./train_gpu.sh

# Manual training with PyTorch
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model large \
    --epochs 200 \
    --batch_size 256 \
    --device auto \
    --save_dir ./models

# Model types: simple (~200K params) or large (~1M params)
# Training output: ./models/best_model.pth, ./models/training_history.png
```

#### 3. Deploy Trained Model
```bash
# Launch with neural network controller (replaces traditional gaits)
ros2 launch pidog_gaits nn_demo.launch.py

# Send commands to test learned behaviors
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
```

### Docker GPU Training
```bash
# AMD ROCm GPU
docker-compose --profile rocm up -d pidog-rocm
docker exec -it pidog_training_rocm bash

# NVIDIA CUDA GPU
docker-compose --profile cuda up -d pidog-cuda
docker exec -it pidog_training_cuda bash

# CPU only
docker-compose --profile cpu up -d pidog-cpu
```

## Architecture

### Package Structure

**pidog_description**: Robot URDF models, Gazebo worlds, and ros2_control configuration
- `urdf/pidog.urdf` - Robot description with physics parameters
- `config/pidog_controllers.yaml` - ros2_control PID gains (P=8.0, D=0.3, I=0.1)
- `launch/gazebo.launch.py` - Main Gazebo launch file
- `worlds/pidog.sdf` - Custom world with optimized physics (CFM=0.001 for soft contacts)

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

**Output**: 8 joint angles (radians) for leg motors

**Available Models**:

1. **GaitNet (Simple)** - Basic MLP (~200K params)
   - Input(4) → Dense(128) → Dense(256) → Dense(128) → Dense(8)
   - Good for: Initial testing, fast training

2. **GaitNetLarge** - Larger MLP (~1M params)
   - Input(4) → Dense(256) → Dense(512) → Dense(512) → Dense(256) → Dense(8)
   - Good for: When simple model underfits

3. **GaitNetSimpleLSTM** - LSTM with temporal memory (~13K params) ⭐ **RECOMMENDED**
   - Input(4) → LSTM(64) → Dense(32) → Dense(8)
   - Good for: Sim-to-real transfer, handles servo lag
   - **Best choice for hardware deployment**

4. **GaitNetLSTM** - Full state feedback LSTM (~35K params)
   - Input(20) → LSTM(128) → Dense(64) → Dense(8)
   - Input = [gait_cmd(4), joint_pos(8), joint_vel(8)]
   - Good for: Advanced control, closed-loop feedback

**Training expectations**:
- 10,000+ samples recommended
- Validation loss < 0.01 indicates good learning
- Converges in 50-100 epochs typically
- LSTM models may need 100-200 epochs

## Physics Tuning

Current Gazebo configuration (optimized for stability):
- Joint effort: 0.35 Nm (3.6 kg·cm, above real servo 1.6 kg·cm)
- Joint damping: 0.5 (balanced for stability)
- World CFM: 0.001 (soft contacts to prevent jitter)
- PID gains: P=8.0, D=0.3, I=0.1 (moderate responsiveness)

If the robot exhibits instability:
1. Check PID gains in `pidog_description/config/pidog_controllers.yaml`
2. Verify joint effort limits in `pidog_description/urdf/pidog.urdf`
3. Review world physics in `pidog_description/worlds/pidog.sdf`

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

### Tuning Gait Parameters
Gait parameters are in `pidog_gaits/pidog_gaits/walk_gait.py` and `trot_gait.py`:
- `stride_height`: Height of leg lift during step
- `stride_length`: Forward/backward step distance
- `swing_period`: Duration of swing phase
- Edit parameters, rebuild package, and test

## Dependencies

**ROS2 packages**:
- rclpy
- ros_gz_sim, ros_gz_bridge (Gazebo integration)
- controller_manager, joint_state_broadcaster, position_controllers (ros2_control)

**Python packages**:
- torch, torchvision (PyTorch for neural networks)
- numpy (data processing)
- matplotlib (training visualization)

## Installation Notes

This workspace uses **ROS2 Jazzy** in Docker with AMD GPU support (ROCm).


## Sim-to-Real Transfer Notes

### Physics Parameters (Tuned Configuration)

**Joint Torque Limits** - Empirically tuned:
- Leg joints (8 total): 0.35 Nm effort limit
- Tail joint: 0.35 Nm
- Neck joints (3 total): 0.14 Nm

**Joint Dynamics** - Anti-jitter settings based on Gazebo best practices:
- Leg joints: damping=0.7, friction=1.0
- Neck joints: damping=0.3, friction=0.5, stiffness=50.0

**Contact Physics** - Stable configuration for Gazebo ODE:
- Foot contact: kp=1e15, kd=1e13, mu1=0.8 (very stiff for stable ground)
- Ground: kp=1e7, kd=100, mu=1.0
- ODE Solver: type="world" (Dantzig direct solver for stability)
- ODE Constraints: CFM=0.0 (stiff contacts), ERP=0.2 (recommended)
- Solver iterations: 150 (high for convergence)

**PID Controller Gains** - Tuned for stable position control:
- P=5.0 (moderate responsiveness)
- D=1.0 (damping for stiff contacts)
- I=0.05 (minimal steady-state correction)
- Update rate: 50 Hz (matches servo PWM rate)

**Spawn Height**: 0.12m (prevents ground penetration on startup)

**Enhanced Data Collection** - For robust sim-to-real transfer:
- Position observation noise: σ=0.01 rad (~0.57° sensor error)
- Velocity observation noise: σ=0.1 rad/s
- Use `collect_data_enhanced.launch.py`

## Path Assumptions

**Important**: The workspace is located at `/home/user/pidog_ros2` inside the Docker container. This path is used in:
- All scripts and launch files
- Volume mount: `$(pwd):/home/user/pidog_ros2`
- Working directory in container

All relative paths (e.g., `./training_data`, `./models`) work from the workspace root.
