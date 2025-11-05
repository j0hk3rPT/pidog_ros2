# PiDog ROS 2

A ROS 2 package for simulating and controlling the PiDog quadruped robot using Webots.

## Prerequisites

- Docker installed on your system
- X11 display server (for GUI visualization)

## Docker Setup

### 1. Allow Docker to access X11 display

```bash
xhost +local:docker
```

### 2. Run the Docker container

```bash
docker run -it \
  --name pidog_ros2 \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/user/pidog_ros2 \
  -w /home/user/pidog_ros2 \
  ros:jazzy \
  /bin/bash
```

### 3. Inside the container, install dependencies

```bash
apt-get update && apt-get install -y \
    xvfb \
    libgl1 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libsndio7.0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    x11-utils \
    mesa-utils \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2t64 \
    libxi6 \
    libxtst6 \
    ros-jazzy-webots-ros2
```

### 4. Build the workspace

```bash
colcon build
```

### 5. Source the environment

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### 6. Launch the gait demo

```bash
ros2 launch pidog_gaits gait_demo.launch.py
```

## Development Process

The PiDog robot was created through the following process:

1. Draw parts in FreeCAD and export the .dae files. The parts were all assembled inside FreeCAD to verify relative dimensions.
2. Utilized Blender to correct some deviations in the meshes (e.g. CoM deviation from the (0,0,0))
3. Creating URDF of robot
4. Using [urdf2webots](https://github.com/cyberbotics/urdf2webots) to convert URDF to Webots PROTO format:

```bash
python -m urdf2webots.importer --input=urdf/pidog.urdf --output=proto/ --normal --init-pos="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
```
