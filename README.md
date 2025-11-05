# PiDog ROS 2

A ROS 2 simulation and control package for the PiDog quadruped robot, featuring gait generation, neural network controllers, and Webots integration.

## Features

- 🐕 **Webots 3D Simulation** - Full physics simulation with realistic robot model
- 🚶 **Multiple Gaits** - Walk, trot, and various poses (stand, sit, lie, stretch)
- 🧠 **Neural Network Control** - Train and deploy learned gaits
- 🎮 **ROS 2 Integration** - Standard ROS 2 interfaces for control and sensing
- 🐋 **Docker Support** - Easy setup with Docker containers

## Quick Start

### Option 1: Docker (Recommended)

The easiest way to get started is using Docker with X11 forwarding:

```bash
# On host: Allow Docker X11 access
xhost +local:docker

# Run container
docker run -it \
  --name pidog_ros2 \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/user/pidog_ros2 \
  -w /home/user/pidog_ros2 \
  ros:jazzy \
  /bin/bash

# Inside container: Install dependencies
apt-get update && apt-get install -y \
    libgl1 libgl1-mesa-dri libglu1-mesa libsndio7.0 \
    libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
    x11-utils mesa-utils libxrandr2 libxss1 libxcursor1 libxcomposite1 \
    libasound2t64 libxi6 libxtst6 ros-jazzy-webots-ros2

# Build and run
colcon build --packages-select pidog_gaits
source install/setup.bash
ros2 launch pidog_gaits gait_demo.launch.py
```

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed Docker instructions including macOS and Windows.

### Option 2: Native Installation

```bash
# Install ROS 2 Jazzy
# Follow: https://docs.ros.org/en/jazzy/Installation.html

# Install Webots R2025a
# Download from: https://github.com/cyberbotics/webots/releases/tag/R2025a

# Install dependencies
sudo apt install ros-jazzy-webots-ros2

# Build workspace
cd pidog_ros2
colcon build
source install/setup.bash

# Launch simulation
ros2 launch pidog_gaits gait_demo.launch.py
```

## Package Structure

- **pidog_description** - URDF robot description and mesh files
- **pidog_sim** - Webots simulation integration and driver
- **pidog_control** - Basic joint control nodes
- **pidog_gaits** - Gait generation and neural network training

## Usage

### Launch Gait Demo

```bash
source install/setup.bash
ros2 launch pidog_gaits gait_demo.launch.py
```

This launches:
- Webots simulator with PiDog robot
- Gait generator node (publishes motor commands)
- Robot starts in "stand" pose

### Available Gaits and Poses

**Dynamic Gaits:**
- `walk_forward`, `walk_backward`, `walk_left`, `walk_right`
- `trot_forward`, `trot_backward`, `trot_left`, `trot_right`

**Static Poses:**
- `stand`, `sit`, `lie`, `stretch`

### Control via ROS Topics

Change gait/pose by publishing to `/gait_command`:

```bash
# Switch to walk forward
ros2 topic pub /gait_command std_msgs/String "data: 'walk_forward'" --once

# Switch to sit pose
ros2 topic pub /gait_command std_msgs/String "data: 'sit'" --once
```

### Neural Network Training

See [NEURAL_NETWORK_QUICKSTART.md](NEURAL_NETWORK_QUICKSTART.md) for training custom gaits.

## Development

### Building Specific Packages

```bash
# Build single package
colcon build --packages-select pidog_gaits

# Build with dependencies
colcon build --packages-up-to pidog_sim

# Clean build
rm -rf build/ install/ log/
colcon build
```

### Running Tests

```bash
colcon test --packages-select pidog_gaits
colcon test-result --verbose
```

## Architecture

### Robot Model Creation Process

1. **CAD Design** - Parts designed in FreeCAD and exported as .dae meshes
2. **Mesh Processing** - Blender used to correct CoM and mesh deviations
3. **URDF Creation** - Robot description with joints, links, and visuals
4. **Webots Conversion** - URDF converted to Webots PROTO using urdf2webots:

```bash
python -m urdf2webots.importer \
  --input=urdf/pidog.urdf \
  --output=proto/ \
  --normal \
  --init-pos="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
```

### ROS 2 Node Architecture

```
┌─────────────────┐
│  gait_generator │ ──┐
└─────────────────┘   │
                      │ /motor_pos (JointState)
┌─────────────────┐   │
│ nn_controller   │ ──┤
└─────────────────┘   │
                      ↓
              ┌──────────────────┐
              │ pidog_sim_driver │
              └──────────────────┘
                      ↓
              ┌──────────────────┐
              │  Webots Simulator│
              └──────────────────┘
```

## Troubleshooting

### Webots window doesn't appear in Docker

Make sure you're using X11 forwarding:
- Linux: `xhost +local:docker` before running container
- macOS: Install and configure XQuartz
- Windows: Install VcXsrv or Xming

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for platform-specific instructions.

### Robot is invisible in simulation

Fixed in recent updates. Pull latest changes:
```bash
git pull
colcon build --packages-select pidog_sim
```

### "libsndio.so.7: cannot open shared object file"

Install the audio library:
```bash
sudo apt install libsndio7.0
```

### Module import errors

Make sure to source the workspace after building:
```bash
source install/setup.bash
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add license information]

## Resources

- [ROS 2 Jazzy Documentation](https://docs.ros.org/en/jazzy/)
- [Webots Documentation](https://cyberbotics.com/doc/guide/index)
- [Docker Setup Guide](DOCKER_SETUP.md)
- [Neural Network Training](NEURAL_NETWORK_QUICKSTART.md)

## Acknowledgments

Based on the SunFounder PiDog hardware platform.
