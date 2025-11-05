# Running Webots with PiDog ROS 2 in Docker

This guide explains how to run the PiDog ROS 2 project with Webots simulator inside a Docker container with full 3D visualization.

## Prerequisites

- Docker installed on your host system
- X Server running on host (Linux: built-in, macOS: XQuartz, Windows: VcXsrv/Xming)
- Official ROS Docker image: `ros:jazzy`

## Quick Start (Recommended)

### On Linux with X Server

```bash
# On host: Allow Docker to connect to X server
xhost +local:docker

# Run Docker container with X11 forwarding
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

# Optional: Run setup script for additional configuration
./setup_webots_docker.sh

# Build the workspace
colcon build --packages-select pidog_gaits

# Source the workspace
source install/setup.bash

# Launch the simulation - Webots window will appear on your host!
ros2 launch pidog_gaits gait_demo.launch.py
```

## Platform-Specific Setup

### Linux (Recommended - Easiest Setup)

X Server is built-in on most Linux distributions.

```bash
# Allow Docker to access X server
xhost +local:docker

# Run container with X11 forwarding
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

### macOS with XQuartz

1. Install XQuartz:
   ```bash
   brew install --cask xquartz
   ```

2. Start XQuartz and enable "Allow connections from network clients" in Preferences → Security

3. Get your IP address:
   ```bash
   IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
   ```

4. Allow connections:
   ```bash
   xhost +$IP
   ```

5. Run container:
   ```bash
   docker run -it \
     --name pidog_ros2 \
     --privileged \
     -e DISPLAY=$IP:0 \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v $(pwd):/home/user/pidog_ros2 \
     -w /home/user/pidog_ros2 \
     ros:jazzy \
     /bin/bash
   ```

### Windows with VcXsrv or Xming

1. Install VcXsrv (https://sourceforge.net/projects/vcxsrv/) or Xming

2. Start X server with "Disable access control" option checked

3. Run container:
   ```powershell
   docker run -it `
     --name pidog_ros2 `
     --privileged `
     -e DISPLAY=host.docker.internal:0 `
     -v ${PWD}:/home/user/pidog_ros2 `
     -w /home/user/pidog_ros2 `
     ros:jazzy `
     bash
   ```

## Required Dependencies

Install these packages inside the Docker container:

```bash
apt-get update && apt-get install -y \
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

## Building and Running

```bash
# Build the workspace (first time or after code changes)
colcon build --packages-select pidog_gaits

# Source the workspace
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Launch the simulation
ros2 launch pidog_gaits gait_demo.launch.py
```

## Expected Result

When you run the launch command:
1. ✅ Webots 3D window opens on your host display
2. ✅ PiDog robot is visible in the simulation
3. ✅ Robot starts in "stand" pose
4. ✅ All ROS 2 nodes start successfully:
   - `webots-1` - Webots simulator
   - `webots_controller_PiDog-2` - Webots ROS 2 controller
   - `pidog_gait_control-4` - Gait control node
   - `gait_generator-5` - Gait generator (controls robot pose)

## Troubleshooting

### Webots window doesn't appear

**Check X server connection:**
```bash
# Inside container
echo $DISPLAY
xdpyinfo | head
```

**On Linux host:**
```bash
xhost +local:docker
```

### "cannot open display" error

Make sure:
1. X server is running on host
2. DISPLAY environment variable is set correctly
3. X11 socket is mounted: `-v /tmp/.X11-unix:/tmp/.X11-unix`

### "libGL error" or graphics issues

Install mesa drivers:
```bash
apt-get install -y libgl1-mesa-dri mesa-utils
```

### "libsndio.so.7: cannot open shared object file"

Install audio library:
```bash
apt-get install -y libsndio7.0
```

### Robot is invisible in Webots

This was fixed in the mesh path updates. Make sure you're on the latest version:
```bash
git pull
colcon build --packages-select pidog_sim pidog_gaits
```

## Persistent Docker Container Setup

For repeated use, create a Dockerfile:

```dockerfile
FROM ros:jazzy

# Install Webots dependencies
RUN apt-get update && apt-get install -y \
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
    ros-jazzy-webots-ros2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
```

Build and run:
```bash
docker build -t pidog_ros2:webots .

docker run -it \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  pidog_ros2:webots
```

## Notes

- **X11 forwarding** (recommended): Shows Webots window on your host display
- **Headless mode** (Xvfb): Runs without GUI - not recommended for interactive use
- The `--privileged` flag may not be necessary for all setups but ensures full device access
- Performance may be slower than native installation due to Docker overhead

## Additional Resources

- [ROS Docker Tutorial](https://docs.ros.org/en/jazzy/How-To-Guides/Run-2-nodes-in-single-or-separate-docker-containers.html)
- [Webots Docker Documentation](https://cyberbotics.com/doc/guide/running-extern-robot-controllers#running-webots-in-docker)
- [X11 Forwarding Guide](https://wiki.ros.org/docker/Tutorials/GUI)
