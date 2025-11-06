# PiDog Gazebo Simulation Setup

Complete guide for running PiDog in Gazebo Harmonic with ROS 2 Humble.

## Prerequisites

- **Ubuntu 22.04 (Jammy)**
- **ROS 2 Humble** installed
- **Python 3.10+**

## Installation

### 1. Install Gazebo Harmonic and Dependencies

Run the installation script:

```bash
cd ~/pidog_ros2
chmod +x install_gazebo.sh
./install_gazebo.sh
```

This script installs:
- Gazebo Harmonic
- ros_gz bridge packages
- ros2_control and controllers
- All required dependencies

### 2. Build the Workspace

```bash
cd ~/pidog_ros2
colcon build --symlink-install
source install/setup.bash
```

## Running the Simulation

### Launch Gazebo with PiDog

```bash
ros2 launch pidog_description gazebo.launch.py
```

This will:
1. Start Gazebo Harmonic
2. Load the PiDog robot model (URDF)
3. Spawn the robot at position (0, 0, 0.12)
4. Start ros2_control with position controllers
5. Launch the standing pose controller
6. The robot will stand in a stable pose

### Launch Options

**Without RViz (Gazebo only):**
```bash
ros2 launch pidog_description gazebo.launch.py use_rviz:=false
```

**Change simulation speed:**
Edit `pidog_description/worlds/pidog.sdf` and modify `real_time_factor`.

## Robot Configuration

### Standing Pose

The robot stands with:
- **Shoulders**: 0.0 rad (straight out from body)
- **Knees**: -0.8 rad (~45° down to lift body off ground)

### Motor Specifications

Based on real PiDog servo specs:
- **Type**: Digital servo with position feedback
- **Torque**: 1.4 kg·cm (0.137 N·m) at 6V
- **Speed**: 0.15 sec/60° at 6V
- **Operating Angle**: 180° ± 10°
- **Weight**: 13.5g per servo

### Simulation Parameters

**Physics (pidog_description/worlds/pidog.sdf):**
- Timestep: 1ms (1000 Hz)
- Solver iterations: 50
- CFM: 0.0 (rigid constraints)
- ERP: 0.2 (error reduction parameter)

**PID Gains (pidog_description/config/pidog_controllers.yaml):**
- P = 1000 (strong position holding)
- I = 10 (eliminate steady-state error)
- D = 50 (damping to prevent oscillation)

**Contact Parameters:**
- Friction: 1.0 (paw to ground)
- Contact stiffness: 1,000,000 N/m
- Contact damping: 100 N·s/m

## Architecture

### Package Structure

```
pidog_ros2/
├── pidog_description/          # Robot model and simulation
│   ├── urdf/pidog.urdf         # Robot URDF with ros2_control
│   ├── worlds/pidog.sdf        # Gazebo world file
│   ├── config/                 # Controller configurations
│   ├── launch/                 # Launch files
│   ├── meshes/                 # 3D models (DAE files)
│   └── scripts/                # Helper scripts
├── pidog_control/              # Robot controllers
│   └── pidog_control/
│       ├── pidog_gazebo_controller.py   # Gazebo controller
│       └── pidog_gait_control.py        # Gait generation
└── pidog_gaits/                # Gait patterns
```

### Control Flow

```
┌─────────────────────────────────────────────┐
│  pidog_gazebo_controller                    │
│  Publishes standing pose at 50Hz            │
│  Topic: /position_controller/commands       │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────┐
│  ros2_control Position Controller           │
│  PID control: P=1000, I=10, D=50            │
│  Generates joint torques                    │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────┐
│  Gazebo Physics Engine (1ms timestep)       │
│  Applies torques to robot joints            │
│  Simulates contacts, dynamics               │
└─────────────────────────────────────────────┘
```

### Topics

**Published by controllers:**
- `/joint_states` - Current joint positions/velocities
- `/position_controller/commands` - Desired joint positions

**Published by Gazebo:**
- `/clock` - Simulation time
- `/tf` - Transform tree

## Troubleshooting

### Robot Falls/Collapses

**Symptom**: Legs don't support body weight

**Solutions**:
1. Check PID gains in `pidog_description/config/pidog_controllers.yaml`
2. Increase P gain for stronger holding force
3. Verify controllers are running:
   ```bash
   ros2 control list_controllers
   ```
4. Check joint commands:
   ```bash
   ros2 topic echo /position_controller/commands
   ```

### Robot Bounces/Jitters

**Symptom**: Robot shakes or bounces when standing

**Solutions**:
1. Reduce physics timestep in `worlds/pidog.sdf`
2. Increase contact stiffness in URDF (kp parameter)
3. Add more joint damping in URDF
4. Check spawn height - should be ~0.12m

### Controllers Not Loading

**Symptom**: Error "Controller manager not available"

**Solutions**:
1. Verify gz_ros2_control is installed:
   ```bash
   ros2 pkg list | grep gz_ros2_control
   ```
2. Check Gazebo plugin is loaded in URDF
3. Rebuild workspace:
   ```bash
   colcon build --packages-select pidog_description pidog_control
   ```

### Gazebo Won't Start

**Symptom**: "gz sim" command not found

**Solutions**:
1. Verify Gazebo installation:
   ```bash
   gz sim --version
   ```
2. Re-run install script:
   ```bash
   ./install_gazebo.sh
   ```
3. Check if gz-harmonic is in PATH

## Advanced Configuration

### Tuning PID Gains

Edit `pidog_description/config/pidog_controllers.yaml`:

```yaml
gains:
  body_to_back_right_leg_b:
    p: 1000.0    # Increase for stronger holding
    d: 50.0      # Increase to reduce oscillation
    i: 10.0      # Increase to eliminate steady-state error
    i_clamp: 10.0
```

**Guidelines:**
- **P gain**: Controls position holding strength (higher = stiffer)
- **D gain**: Damping to prevent oscillation (higher = more damping)
- **I gain**: Eliminates steady-state error (use sparingly)

### Custom Gaits

To add custom gait patterns:

1. Publish to `/motor_pos` topic:
   ```python
   from sensor_msgs.msg import JointState

   msg = JointState()
   msg.name = ['body_to_back_right_leg_b', 'back_right_leg_b_to_a', ...]
   msg.position = [0.0, -0.8, ...]  # Your gait angles
   publisher.publish(msg)
   ```

2. The `pidog_gazebo_controller` will forward commands to ros2_control

## Differences from Webots

**Webots (OLD):**
- Custom PROTO files
- Built-in motor controllers
- Single controller process
- `.wbt` world files

**Gazebo (NEW):**
- Standard URDF format
- ros2_control framework
- Modular controller architecture
- `.sdf` world files

**Migration Benefits:**
- Better ROS 2 integration
- Standard robotics tools
- More community support
- Easier debugging

## Performance Optimization

### For Real-time Performance

1. **Reduce visualization quality** in Gazebo GUI
2. **Disable shadows** in world file
3. **Run headless**:
   ```bash
   gz sim -s -r pidog_world.sdf
   ```

### For Accuracy

1. **Reduce timestep** (but slower):
   ```xml
   <max_step_size>0.0005</max_step_size>
   ```
2. **Increase solver iterations**:
   ```xml
   <iters>100</iters>
   ```

## References

- [Gazebo Documentation](https://gazebosim.org/docs)
- [ros2_control Documentation](https://control.ros.org/)
- [PiDog Hardware Specs](https://github.com/sunfounder/pidog)
- [ROS 2 Jazzy Documentation](https://docs.ros.org/en/jazzy/)

## License

MIT License - See LICENSE file for details
