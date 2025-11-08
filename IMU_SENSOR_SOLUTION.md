# IMU Sensor Solution - Native Gazebo IMU

## Problem

For sim-to-real transfer in RL training, the neural network needs to train on IMU sensor data that matches what it will receive on real hardware.

## Solution: Native Gazebo IMU with Sensor System Plugins

### Architecture

**Native Gazebo IMU approach** (CURRENT SOLUTION):
1. IMU sensor defined in URDF (`pidog.urdf` line 901)
2. Gazebo Sensors system plugin processes the sensor (`pidog.sdf` line 99)
3. Gazebo IMU system plugin handles IMU-specific processing (line 102)
4. ros_gz_bridge bridges Gazebo `/imu` topic to ROS2
5. RL environment subscribes to `/imu` topic (same as real robot)

**Key advantage**: Uses Gazebo's native, well-tested IMU implementation with realistic physics simulation.

### Configuration Files

#### 1. URDF Sensor Definition (`pidog_description/urdf/pidog.urdf`)

```xml
<link name="imu_link">
  <inertial>
    <mass value="0.001"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="body"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.02" rpy="0 0 0"/>
</joint>

<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <topic>imu</topic>
    <enable_metrics>false</enable_metrics>
    <imu>
      <angular_velocity>
        <x><noise type="gaussian"><mean>0.0</mean><stddev>0.01</stddev></noise></x>
        <y><noise type="gaussian"><mean>0.0</mean><stddev>0.01</stddev></noise></y>
        <z><noise type="gaussian"><mean>0.0</mean><stddev>0.01</stddev></noise></z>
      </angular_velocity>
      <linear_acceleration>
        <x><noise type="gaussian"><mean>0.0</mean><stddev>0.1</stddev></noise></x>
        <y><noise type="gaussian"><mean>0.0</mean><stddev>0.1</stddev></noise></y>
        <z><noise type="gaussian"><mean>0.0</mean><stddev>0.1</stddev></noise></z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

#### 2. World File Sensor Plugins (`pidog_description/worlds/pidog.sdf`)

```xml
<!-- System plugins for sensor processing -->
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
  <render_engine>ogre2</render_engine>
</plugin>
<plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
```

**IMPORTANT**: These must be at world level, not model level.

#### 3. Launch File Bridge (`pidog_description/launch/gazebo.launch.py`)

```python
# Bridge IMU sensor from Gazebo to ROS 2
imu_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=['/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'],
    output='screen',
)
```

### Technical Details

**Sensor Output Format** (`sensor_msgs/Imu`):
```python
orientation:        # Quaternion (x, y, z, w)
angular_velocity:   # rad/s (x, y, z) - body frame
linear_acceleration: # m/s² (x, y, z) - body frame, includes gravity
```

**Noise Model**:
- Angular velocity: σ = 0.01 rad/s (Gaussian noise)
- Linear acceleration: σ = 0.1 m/s² (Gaussian noise)
- Matches realistic IMU sensor characteristics

**Update Rate**: 100 Hz (configurable in URDF)

**Frame**: `imu_link` (fixed to robot body, 2cm above body center)

### Sim-to-Real Transfer

**In Simulation**:
- Native Gazebo IMU publishes to `/imu`
- RL network trains on `sensor_msgs/Imu` data
- Realistic noise and physics simulation

**On Real Robot**:
- Real IMU hardware publishes to `/imu`
- Same `sensor_msgs/Imu` message format
- **Zero code changes needed** - network uses identical input!

### Usage

**Launch Gazebo with IMU**:
```bash
ros2 launch pidog_description gazebo.launch.py
```

**Verify IMU is publishing**:
```bash
# Check if topic exists
ros2 topic list | grep imu

# Show one IMU message
ros2 topic echo /imu --once

# Check publishing rate (should be ~100 Hz)
ros2 topic hz /imu
```

**Expected output when robot is standing**:
```yaml
orientation:
  x: 0.0  # Small values (nearly upright)
  y: 0.0
  z: 0.0
  w: 1.0  # Close to 1.0 (upright)
angular_velocity:
  x: 0.0  # Small noise around zero (stationary)
  y: 0.0
  z: 0.0
linear_acceleration:
  x: 0.0
  y: 0.0
  z: 9.81  # Gravity! (robot measures upward acceleration due to ground support)
```

### Troubleshooting

**If `/imu` topic doesn't exist**:
1. Check Gazebo logs for sensor errors
2. Verify sensor plugins are loaded: `[INFO] [gz_ros_control]: Loading sensor: imu_sensor`
3. Ensure Sensors system plugin is in world file
4. Check bridge is running: `ros2 node list | grep parameter_bridge`

**If Gazebo crashes with sensor plugins**:
- This was the original issue - if it still crashes, it may be due to:
  - GPU/rendering issues with ogre2
  - Missing Gazebo Harmonic sensor plugin libraries
  - Try removing `<render_engine>ogre2</render_engine>` from Sensors plugin

**If IMU data looks wrong**:
- Check orientation: `w` should be close to 1.0 when upright
- Check acceleration: `z` should be ~9.81 when standing
- If all zeros: Sensor may not be initialized yet (wait a few seconds)

### RL Environment Integration

The RL environment (`pidog_rl_env_sensors.py`) already subscribes to `/imu`:

```python
self.imu_sub = self.node.create_subscription(
    Imu,
    '/imu',
    self._imu_callback,
    10
)
```

**Observation space includes**:
- IMU orientation (roll, pitch, yaw from quaternion)
- IMU angular velocity (wx, wy, wz)
- IMU linear acceleration (ax, ay, az)

No code changes needed - it works with both simulation and real hardware!

### Files Modified

- `pidog_description/worlds/pidog.sdf` - Added Sensors and IMU system plugins
- `pidog_description/launch/gazebo.launch.py` - Enabled IMU bridge
- `pidog_description/urdf/pidog.urdf` - IMU sensor configuration (already present)

### References

- [Gazebo Harmonic Sensors](https://gazebosim.org/api/sim/8/sensors.html)
- [ros_gz_bridge](https://github.com/gazebosim/ros_gz/tree/ros2/ros_gz_bridge)
- [SDF Sensor Spec](http://sdformat.org/spec?elem=sensor)
