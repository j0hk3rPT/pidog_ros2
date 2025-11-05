# Running Webots in Docker

This guide explains how to run the PiDog ROS 2 project with Webots simulator inside a Docker container.

## Prerequisites

- Docker installed on your host system
- Official ROS Docker image: `ros:jazzy`

## Quick Start

If you're already inside the Docker container, run:

```bash
# Run the setup script
./setup_webots_docker.sh

# Source ROS and workspace
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Launch Webots simulation
ros2 launch pidog_gaits gait_demo.launch.py
```

## Detailed Setup

### 1. Required Packages (Already Installed)

The following packages have been installed in your container:

- `xvfb` - Virtual framebuffer for headless X server
- `libgl1`, `libgl1-mesa-dri` - OpenGL libraries
- `libglu1-mesa` - OpenGL utility library
- Various X11 libraries for Webots GUI components
- `mesa-utils` - OpenGL testing utilities

### 2. Environment Configuration

The setup script configures these environment variables:

```bash
export DISPLAY=:99                                    # Virtual display
export USER=root                                      # Username for Webots
export USERNAME=root                                  # Alternative username variable
export WEBOTS_HOME=/root/.ros/webotsR2025a/webots   # Webots installation path
```

These variables are also added to `~/.bashrc` for persistence.

### 3. Virtual Display (Xvfb)

Xvfb provides a virtual display for running graphical applications without a physical screen:

```bash
# Start Xvfb (done automatically by setup script)
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &

# Verify it's running
ps aux | grep Xvfb

# Test OpenGL rendering
DISPLAY=:99 glxinfo | grep "direct rendering"
```

## Docker Run Options (For Host System)

If you need to recreate the container with proper settings, use these options:

### Option 1: Headless Mode (Current Setup)

```bash
docker run -it \
  --name pidog_ros2 \
  --privileged \
  -v $(pwd):/home/user/pidog_ros2 \
  -w /home/user/pidog_ros2 \
  ros:jazzy \
  /bin/bash
```

Then run the setup script inside the container.

### Option 2: X11 Forwarding (If you have X server on host)

**Linux hosts with X server:**

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

# On host, allow Docker to connect to X server
xhost +local:docker
```

**macOS with XQuartz:**

```bash
# Install XQuartz first
# Start XQuartz and enable "Allow connections from network clients"

# Get your IP
IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')

docker run -it \
  --name pidog_ros2 \
  --privileged \
  -e DISPLAY=$IP:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/user/pidog_ros2 \
  -w /home/user/pidog_ros2 \
  ros:jazzy \
  /bin/bash

# On host
xhost +$IP
```

**Windows with VcXsrv or Xming:**

```powershell
# Install VcXsrv or Xming
# Start X server with "Disable access control" option

docker run -it \
  --name pidog_ros2 \
  --privileged \
  -e DISPLAY=host.docker.internal:0 \
  -v ${PWD}:/home/user/pidog_ros2 \
  -w /home/user/pidog_ros2 \
  ros:jazzy \
  bash
```

### Option 3: Dockerfile

Create a `Dockerfile` for a permanent setup:

```dockerfile
FROM ros:jazzy

# Install Webots dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    libgl1 \
    libgl1-mesa-dri \
    libglu1-mesa \
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
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV DISPLAY=:99
ENV USER=root
ENV USERNAME=root

# Start Xvfb automatically
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
```

Build and run:

```bash
docker build -t pidog_ros2:webots .
docker run -it -v $(pwd):/home/user/pidog_ros2 -w /home/user/pidog_ros2 pidog_ros2:webots
```

## Troubleshooting

### Webots fails with "libGL.so.1: cannot open shared object file"

**Solution:** Run the setup script to install all required libraries:
```bash
./setup_webots_docker.sh
```

### "USER or USERNAME environment variable not set"

**Solution:** Export the variables:
```bash
export USER=root
export USERNAME=root
```

Or run the setup script.

### Xvfb not starting

**Check if port is available:**
```bash
# Kill existing Xvfb
pkill Xvfb

# Start fresh
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
```

### OpenGL not working

**Test OpenGL:**
```bash
DISPLAY=:99 glxinfo | head -20
```

If it fails, check:
1. Xvfb is running: `ps aux | grep Xvfb`
2. DISPLAY is set: `echo $DISPLAY`
3. Libraries are installed: `ldconfig -p | grep libGL`

### Webots window not showing

This is expected in headless mode. Webots runs in the background.

To see GUI (requires X11 forwarding):
1. Set up X11 forwarding on host
2. Use `--display` option or set DISPLAY to host

## Testing the Setup

After running the setup script, verify everything works:

```bash
# 1. Check Xvfb is running
ps aux | grep Xvfb

# 2. Test OpenGL
DISPLAY=:99 glxinfo | grep "OpenGL"

# 3. Check environment
echo $DISPLAY
echo $USER
echo $WEBOTS_HOME

# 4. Build the project
colcon build

# 5. Launch simulation
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch pidog_gaits gait_demo.launch.py
```

## Performance Notes

- Headless mode (Xvfb) uses CPU rendering, which is slower than GPU
- For better performance, use X11 forwarding with GPU support
- Consider reducing simulation complexity for Docker environments

## Additional Resources

- [ROS Docker Tutorial](https://docs.ros.org/en/jazzy/How-To-Guides/Run-2-nodes-in-single-or-separate-docker-containers.html)
- [Webots Docker Documentation](https://cyberbotics.com/doc/guide/running-extern-robot-controllers#running-webots-in-docker)
- [X11 Forwarding Guide](https://wiki.ros.org/docker/Tutorials/GUI)
