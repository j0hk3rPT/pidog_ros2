# PiDog Sensor Integration & Fast Running Gait Research

**Date**: 2025-11-08
**Goal**: Integrate PiDog's sensor suite into simulation and train a neural network to achieve maximum running speed using dog-like behaviors.

---

## Executive Summary

This document outlines a comprehensive plan to transform the PiDog simulation from a purely kinematic system into a sensor-rich, dog-like robot capable of fast running gaits. The current system uses only ground-truth joint states; we will add all hardware sensors and implement a galloping gait to maximize speed.

**Key Findings**:
- Current system: **NO sensors implemented** (only ground-truth joint states)
- Target speed: **1.8-2.2 m/s** (based on research benchmarks)
- Critical sensors: **IMU (6DOF)**, **contact sensors**, **camera** (optional for obstacles)
- Recommended gait: **Trot** (medium speed) → **Gallop** (high speed)
- RL Architecture: Extend current PPO to 50+ dimensional observation space

---

## 1. PiDog Hardware Sensor Suite

### Available Sensors (SunFounder PiDog)

| Sensor | Type | Purpose | Priority |
|--------|------|---------|----------|
| **6-DOF IMU** | MPU6050 | Gyroscope + Accelerometer for orientation/stability | **CRITICAL** |
| **Ultrasonic** | HC-SR04 | Distance measurement (0.02-4m) | High |
| **Camera** | CSI Module | Vision, obstacle detection | Medium |
| **Touch Sensor** | Capacitive | Head touch detection | Low |
| **Sound Direction** | Microphone Array | Sound source localization | Low |

### Sensor Data Formats

**IMU (6-DOF)**:
- Angular velocity: ωx, ωy, ωz (rad/s)
- Linear acceleration: ax, ay, az (m/s²)
- Orientation: Roll, Pitch, Yaw (rad) or Quaternion
- Update rate: 50-100 Hz

**Ultrasonic**:
- Range: 0.02-4.0 meters
- Cone angle: ~30° (horizontal)
- Update rate: 5-10 Hz
- Noise: σ = 0.01m (Gaussian)

**Camera**:
- Resolution: 640x480 (configurable)
- FOV: 1.4 rad (~80°) standard, 3.14 rad wide-angle
- Format: RGB8 or compressed JPEG
- Update rate: 20-30 Hz

---

## 2. Gazebo Sensor Implementation

### 2.1 IMU Sensor (CRITICAL for Fast Running)

**URDF Addition** (`pidog.urdf`):

```xml
<!-- IMU link (mounted on body) -->
<joint name="imu_joint" type="fixed">
  <origin xyz="0 0 0.02" rpy="0 0 0"/>  <!-- Center of body, 2cm up -->
  <parent link="body"/>
  <child link="imu_link"/>
</joint>

<link name="imu_link">
  <inertial>
    <mass value="0.001"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
</link>

<!-- Gazebo IMU sensor -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>1</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <topic>imu</topic>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>  <!-- 0.01 rad/s noise -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.1</stddev>  <!-- 0.1 m/s² noise -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.1</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.1</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

**World Plugin** (`pidog.sdf` or gazebo launch):
```xml
<plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
```

**ROS2 Bridge**:
```python
# In launch file
Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
    ]
)
```

**Expected ROS Topic**:
- `/imu` (sensor_msgs/Imu) - 100 Hz

---

### 2.2 Contact Sensors (Foot Sensing for Stance Phase)

**URDF Addition** (for each foot):

```xml
<!-- Example for back_right foot -->
<gazebo reference="back_right_leg_a">
  <sensor name="back_right_foot_contact" type="contact">
    <contact>
      <collision>back_right_leg_a_collision</collision>
    </contact>
    <update_rate>50</update_rate>
    <plugin name="back_right_contact_plugin" filename="libgazebo_ros_bumper.so">
      <always_on>true</always_on>
      <update_rate>50.0</update_rate>
      <bumper_topic_name>contact/back_right_foot</bumper_topic_name>
      <frame_name>back_right_leg_a</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

**Need to add for all 4 feet**:
- `contact/back_right_foot`
- `contact/front_right_foot`
- `contact/back_left_foot`
- `contact/front_left_foot`

**World Plugin**:
```xml
<plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
```

---

### 2.3 Ultrasonic/Sonar Sensor (Obstacle Detection)

**URDF Addition** (head-mounted):

```xml
<!-- Sonar link -->
<joint name="sonar_joint" type="fixed">
  <origin xyz="0.05 0 0.03" rpy="0 0 0"/>  <!-- Front of head -->
  <parent link="head"/>
  <child link="sonar_link"/>
</joint>

<link name="sonar_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
  <visual>
    <geometry><box size="0.01 0.02 0.01"/></geometry>
  </visual>
</link>

<!-- Gazebo sonar sensor -->
<gazebo reference="sonar_link">
  <sensor name="sonar" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <visualize>true</visualize>
    <ray>
      <scan>
        <horizontal>
          <samples>5</samples>
          <resolution>1</resolution>
          <min_angle>-0.12</min_angle>
          <max_angle>0.12</max_angle>
        </horizontal>
        <vertical>
          <samples>5</samples>
          <resolution>1</resolution>
          <min_angle>-0.01</min_angle>
          <max_angle>0.01</max_angle>
        </vertical>
      </scan>
      <range>
        <min>0.02</min>
        <max>4.0</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </ray>
    <topic>sonar</topic>
    <gz_frame_id>sonar_link</gz_frame_id>
  </sensor>
</gazebo>
```

**ROS2 Bridge**:
```python
'/sonar@sensor_msgs/msg/Range[gz.msgs.LaserScan',
```

---

### 2.4 Camera (Optional - for Vision-Based Navigation)

**URDF Addition** (head-mounted):

```xml
<!-- Camera link -->
<joint type="fixed" name="camera_joint">
  <origin xyz="0.04 0 0.025" rpy="0 0 0"/>  <!-- Front of head -->
  <child link="camera_link"/>
  <parent link="head"/>
</joint>

<link name='camera_link'>
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
  <visual>
    <geometry><box size="0.01 0.02 0.01"/></geometry>
  </visual>
</link>

<!-- Optical frame for ROS -->
<joint type="fixed" name="camera_optical_joint">
  <origin xyz="0 0 0" rpy="-1.5707 0 -1.5707"/>
  <child link="camera_link_optical"/>
  <parent link="camera_link"/>
</joint>

<link name="camera_link_optical"/>

<!-- Gazebo camera -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.3962634</horizontal_fov>  <!-- ~80° -->
      <image>
        <width>320</width>  <!-- Lower res for faster processing -->
        <height>240</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.05</near>
        <far>10</far>
      </clip>
      <optical_frame_id>camera_link_optical</optical_frame_id>
    </camera>
    <always_on>1</always_on>
    <update_rate>20</update_rate>
    <topic>camera/image</topic>
  </sensor>
</gazebo>
```

**ROS2 Bridge**:
```python
Node(
    package='ros_gz_image',
    executable='image_bridge',
    arguments=['/camera/image'],
)
```

---

## 3. Neural Network Architecture for Sensor-Based Control

### 3.1 Current vs. Proposed Observation Space

**Current** (`pidog_rl_env.py`):
```
Observation (36D):
- gait_cmd: [gait_type, direction, turn] (3)
- phase: [0.0-1.0] (1)
- joint_pos: [12 motors] (12)
- joint_vel: [12 motors] (12)
- body_pos: [x, y, z] (3)
- body_quat: [qx, qy, qz, qw] (4)
- head_contact: [bool] (1)
```

**Proposed** (Extended with Real Sensors):
```
Observation (54D):
- gait_cmd: [gait_type, direction, turn] (3)
- phase: [0.0-1.0] (1)
- joint_pos: [12 motors] (12)
- joint_vel: [12 motors] (12)
- imu_orientation: [roll, pitch, yaw] (3)  ← NEW
- imu_angular_vel: [ωx, ωy, ωz] (3)        ← NEW
- imu_linear_acc: [ax, ay, az] (3)         ← NEW
- body_linear_vel: [vx, vy, vz] (3)        ← Keep from ground truth
- foot_contact: [BR, FR, BL, FL] (4)       ← NEW (binary)
- sonar_range: [distance] (1)              ← NEW
- terrain_height_est: [h] (1)              ← NEW (derived from IMU+contacts)
```

**For Vision** (Optional CNN branch):
- Camera image: 320x240x3 → CNN encoder → 64D feature vector
- Total observation: 54D + 64D = 118D

---

### 3.2 Network Architecture Options

#### Option A: Extended LSTM (Recommended for Phase 1)

**Best for**: Real-time control, sim-to-real transfer, temporal dynamics

```python
class SensorGaitNetLSTM(nn.Module):
    def __init__(self, obs_dim=54, action_dim=12):
        super().__init__()

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # Action head
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, obs, hidden=None):
        lstm_out, hidden = self.lstm(obs, hidden)
        x = F.elu(self.fc1(lstm_out))
        action = torch.tanh(self.fc2(x)) * 1.57  # ±90°
        return action, hidden
```

**Parameters**: ~45K
**Training time**: 30-45 min (GPU, 100K steps)
**Expected performance**: Good for trot (1.0-1.5 m/s)

---

#### Option B: Multi-Modal Network (For Vision Integration)

```python
class MultiModalGaitNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Proprioceptive branch (LSTM)
        self.proprio_lstm = nn.LSTM(54, 128, 2, batch_first=True)

        # Vision branch (CNN encoder)
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),  # 320x240 → 158x118
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), # → 77x57
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),  # → 512D
            nn.Linear(512, 64)
        )

        # Fusion layer
        self.fusion = nn.Linear(128 + 64, 128)
        self.action_head = nn.Linear(128, 12)

    def forward(self, proprio_obs, image):
        # Process proprioception
        lstm_out, _ = self.proprio_lstm(proprio_obs)

        # Process vision
        vision_feat = self.vision_cnn(image)

        # Fuse and output
        fused = torch.cat([lstm_out, vision_feat], dim=-1)
        x = F.elu(self.fusion(fused))
        action = torch.tanh(self.action_head(x)) * 1.57
        return action
```

**Parameters**: ~150K
**Training time**: 1-2 hours (GPU)
**Use case**: Obstacle-aware navigation

---

### 3.3 Reward Function for Fast Running

**Current reward** (pidog_rl_env.py:195) focuses on stability. For fast running, we need to add:

```python
def _calculate_reward_fast_running(self, action):
    """
    Reward for maximizing forward speed while maintaining stability.
    """
    reward = 0.0

    # === PRIMARY: SPEED REWARD ===
    # Exponentially reward higher speeds (target: 2.0 m/s)
    forward_vel = self.body_linear_vel[0]
    if forward_vel > 0:
        reward += forward_vel * 10.0  # Main reward
        if forward_vel > 1.5:
            reward += 5.0  # Bonus for high speed
        if forward_vel > 2.0:
            reward += 10.0  # Big bonus for exceeding target
    else:
        reward -= abs(forward_vel) * 2.0  # Penalty for backward

    # === STABILITY CONSTRAINTS ===
    # Height: higher is better for gallop (10-12cm)
    if 0.10 < self.body_position[2] < 0.15:
        reward += 2.0
    elif self.body_position[2] < 0.08:
        reward -= 5.0  # Too low

    # Orientation: allow more pitch variation during gallop
    qx, qy, qz, qw = self.body_orientation
    roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    pitch = math.asin(2*(qw*qy - qz*qx))

    if abs(roll) < 0.4:  # Relaxed from 0.3
        reward += 1.0
    else:
        reward -= abs(roll) * 2.0

    if abs(pitch) < 0.6:  # Allow forward lean during gallop
        reward += 1.0
    else:
        reward -= abs(pitch) * 2.0

    # === GAIT EFFICIENCY ===
    # Reward ground clearance during swing phase
    feet_contacts = self.foot_contacts  # [BR, FR, BL, FL]
    num_contacts = sum(feet_contacts)

    # Gallop: alternating 2-4 feet (flight phase exists)
    if num_contacts in [0, 2]:  # Good gallop
        reward += 2.0
    elif num_contacts == 4:  # All feet down (stance phase)
        reward += 0.5
    elif num_contacts == 1:  # Single leg (risky)
        reward -= 1.0

    # === ENERGY PENALTY (reduced) ===
    # Allow higher joint velocities for gallop
    joint_vel_penalty = np.sum(np.abs(self.joint_velocities)) * 0.005
    reward -= joint_vel_penalty

    # === TERMINATION ===
    done = False
    if abs(roll) > 1.2 or abs(pitch) > 1.2 or self.body_position[2] < 0.05:
        reward -= 50.0
        done = True

    if self.head_contact:
        reward -= 20.0
        done = True

    return reward, done, {
        'speed': forward_vel,
        'height': self.body_position[2],
        'contacts': num_contacts,
    }
```

**Key differences from walk/trot reward**:
- 10x higher velocity reward coefficient
- Allow flight phase (0 contacts)
- Relaxed pitch constraint (±0.6 rad vs ±0.3)
- Higher target body height (10-15cm vs 8-10cm)

---

## 4. Fast Running Gaits: Trot vs. Gallop

### 4.1 Gait Comparison

| Gait | Speed Range | Foot Pattern | Flight Phase | Energy |
|------|-------------|--------------|--------------|--------|
| **Walk** | 0.2-0.5 m/s | Sequential (3 feet down) | No | Low CoT |
| **Trot** | 0.5-1.8 m/s | Diagonal pairs (BR+FL, FR+BL) | Minimal | **Most efficient** |
| **Gallop** | 1.5-3.0+ m/s | Asymmetric (4→2→0→2→4) | Yes | High CoT |

### 4.2 Trot Gait (Current - KEEP)

**Implementation**: `pidog_gaits/trot_gait.py`

**Characteristics**:
- Phase offset: [0.0, 0.5, 0.5, 0.0] (diagonal pairing)
- Duty factor: ~0.5 (50% stance, 50% swing)
- Frequency: 1.5-2.0 Hz (current: ~1.5 Hz)
- Speed achieved: 0.8-1.5 m/s

**Optimization for speed**:
```python
# In trot_gait.py
TROT_FAST_PARAMS = {
    'LEG_STEP_HEIGHT': 30,      # 20→30mm (higher lift)
    'LEG_STEP_WIDTH': 100,      # 80→100mm (longer stride)
    'STEP_COUNT': 8,            # 10→8 (faster transitions)
    'Z_ORIGIN': 100,            # 80→100mm (higher stance)
}
```

**Expected improvement**: 0.8 m/s → **1.2-1.5 m/s**

---

### 4.3 Gallop Gait (NEW - HIGH PRIORITY)

**Implementation**: Create `pidog_gaits/gallop_gait.py`

**Characteristics**:
- Phase pattern: Asymmetric quadrupedal
- Sequence: Back legs → Flight → Front legs → Flight
- Duty factor: ~0.25-0.35 (short stance, long flight)
- Frequency: 2.0-3.0 Hz
- Target speed: **1.8-2.5 m/s**

**Foot sequence** (one cycle):
```
Time 0.00-0.15s: Both back legs push (BR, BL contact)
Time 0.15-0.35s: FLIGHT PHASE (0 contacts)
Time 0.35-0.50s: Both front legs land (FR, FL contact)
Time 0.50-0.65s: FLIGHT PHASE (0 contacts)
→ Repeat (cycle = 0.65s @ 1.5 Hz)
```

**Key parameters**:
```python
# gallop_gait.py
GALLOP_PARAMS = {
    'LEG_STEP_HEIGHT': 40,      # High lift for clearance
    'LEG_STEP_WIDTH': 120,      # Very long stride
    'BACK_PUSH_TIME': 0.15,     # Back legs push duration
    'FLIGHT_TIME': 0.20,        # Flight phase duration
    'FRONT_LAND_TIME': 0.15,    # Front legs landing duration
    'Z_ORIGIN': 110,            # High stance (11cm)
    'SPINE_FLEX': 0.2,          # Spinal flexion/extension (rad)
}
```

**Spine flexion** (dog-like behavior):
- Extension during back push (arched back)
- Flexion during front landing (curled back)
- Requires adding virtual "spine joint" or modulating neck/tail

---

### 4.4 Gait Transition Logic

**Automatic gait switching** based on commanded speed:

```python
def select_gait_for_speed(target_speed):
    """
    Automatically select gait based on target speed.
    Mimics animal behavior (Froude number).
    """
    if target_speed < 0.6:
        return 'walk_forward'
    elif 0.6 <= target_speed < 1.5:
        return 'trot_forward'
    else:  # target_speed >= 1.5
        return 'gallop_forward'
```

**Neural network learns when to switch**:
- Input: `[desired_speed]` (1D)
- RL reward encourages efficiency → naturally learns transitions
- Research shows this emerges without explicit programming

---

## 5. Implementation Roadmap

### Phase 1: Critical Sensors (Week 1-2)

**Goal**: Add IMU and foot contact sensors, achieve trot at 1.2 m/s

**Tasks**:
1. ✅ Research complete
2. Add IMU sensor to `pidog.urdf`
3. Add contact sensors to 4 feet
4. Update `pidog_rl_env.py` observation space (36D → 52D)
5. Create ROS2 subscriber nodes for `/imu` and `/contact/*`
6. Extend `GaitNetSimpleLSTM` to 52D input
7. Optimize trot parameters for speed
8. Collect training data with sensor noise
9. Train RL policy (50K steps, ~20 min)
10. Test in Gazebo, measure top speed

**Expected outcome**:
- Trot speed: 1.2-1.5 m/s
- IMU-based stability control
- Contact-aware stance/swing transitions

---

### Phase 2: Gallop Gait (Week 3-4)

**Goal**: Implement galloping gait, achieve 2.0+ m/s

**Tasks**:
1. Create `pidog_gaits/gallop_gait.py`
2. Implement asymmetric foot pattern
3. Add spine flexion (modulate neck/tail joints)
4. Update reward function for high-speed stability
5. Add gait transition logic (walk→trot→gallop)
6. Collect gallop demonstration data
7. Train RL with gallop gait (100K steps, ~45 min)
8. Fine-tune reward weights for speed vs. stability
9. Benchmark top speed in flat terrain
10. Test recovery from perturbations

**Expected outcome**:
- Gallop speed: 1.8-2.5 m/s
- Automatic gait transitions
- Flight phase with 0 ground contacts

---

### Phase 3: Sonar & Obstacle Avoidance (Week 5-6)

**Goal**: Add forward-looking sonar, avoid obstacles while running

**Tasks**:
1. Add ultrasonic sensor to `pidog.urdf`
2. Create `/sonar` subscriber in RL env
3. Add sonar distance to observation (52D → 53D)
4. Update reward: penalty for sonar < 0.5m at high speed
5. Create Gazebo world with obstacles (walls, boxes)
6. Collect data with obstacle scenarios
7. Train RL with obstacle avoidance (150K steps)
8. Test in cluttered environment
9. Measure success rate (% obstacle-free runs)

**Expected outcome**:
- Slows down when approaching obstacles
- Learns to stop or turn before collision
- Maintains 1.5+ m/s in open spaces

---

### Phase 4: Vision-Based Navigation (Week 7-8)

**Goal**: Add camera, learn to navigate using vision

**Tasks**:
1. Add camera sensor to `pidog.urdf`
2. Create CNN encoder for image processing
3. Implement `MultiModalGaitNet` architecture
4. Use transfer learning (pretrained ResNet features)
5. Create visually varied Gazebo worlds (textures, colors)
6. Collect image+proprio dataset
7. Train multi-modal policy (200K steps, ~2 hours)
8. Test visual obstacle detection
9. Compare vision vs. sonar performance
10. Benchmark in real-world-like scenarios

**Expected outcome**:
- Vision-guided navigation
- Detect obstacles from appearance (not just range)
- More robust to sensor failures (multi-modal fusion)

---

### Phase 5: Dog-Like Behaviors (Week 9-10)

**Goal**: Implement playful/expressive behaviors using sensors

**Tasks**:
1. **Head tracking**: Use camera to track objects/people
2. **Sound localization**: Turn head toward sound source
3. **Touch response**: React to head touch (stop, sit)
4. **Playful running**: Random direction changes, jumps
5. **Energy management**: Slow down when "tired" (long episode)
6. **Social behavior**: Approach detected person (camera)
7. Create behavior state machine
8. Train hierarchical RL (high-level behavior selection)
9. User study: evaluate "dog-likeness"

**Expected outcome**:
- Engaging, lifelike behaviors
- Interactive pet robot
- Sensor-driven decision making

---

## 6. Technical Challenges & Solutions

### Challenge 1: Sim-to-Real Transfer

**Problem**: Neural network trained in simulation may fail on real hardware due to:
- Sensor noise differences
- Servo dynamics (lag, backlash)
- Ground friction variations

**Solutions**:
1. **Domain randomization**:
   - Randomize physics parameters (friction, damping)
   - Add realistic sensor noise to simulation
   - Vary robot mass (±10%)

2. **System identification**:
   - Measure real servo response time (likely 20-50ms)
   - Add servo delay model to simulation
   - Tune URDF parameters to match real dynamics

3. **Robust training**:
   - Use enhanced data collector with noise
   - Train on varied terrains (carpet, wood, tile)
   - Test in simulation with 2x expected noise

---

### Challenge 2: High-Speed Stability

**Problem**: At 2+ m/s, small errors cause catastrophic falls

**Solutions**:
1. **IMU feedback**: Real-time orientation correction
2. **Predictive control**: LSTM predicts next state, adjusts early
3. **Safety bounds**: Hard limits on roll/pitch during training
4. **Progressive difficulty**: Start slow (0.5 m/s), gradually increase
5. **Recovery behaviors**: Explicit "stumble recovery" training

---

### Challenge 3: Computational Cost (Vision)

**Problem**: Camera at 30 Hz + CNN inference may exceed real-time on Raspberry Pi

**Solutions**:
1. **Reduce resolution**: 320x240 → 160x120 (4x faster)
2. **Lower frame rate**: 30 Hz → 10 Hz (good enough for obstacles)
3. **Lightweight CNN**: MobileNetV2 instead of ResNet
4. **Edge TPU**: Use Coral USB accelerator (4 TOPS, 2W)
5. **Hybrid**: Use sonar for real-time, camera for planning

---

### Challenge 4: Contact Sensor Accuracy

**Problem**: Gazebo contact sensors can be noisy, false positives

**Solutions**:
1. **Force threshold**: Only count contact if force > 0.5N
2. **Temporal filtering**: Contact = True if detected for 2+ consecutive frames
3. **Calibration**: Record contact patterns from working gaits
4. **Redundancy**: Use both contact sensor + joint torque estimation

---

## 7. Performance Benchmarks

### Speed Targets

| Milestone | Gait | Target Speed | Stability | Timeline |
|-----------|------|--------------|-----------|----------|
| Baseline | Trot (current) | 0.8 m/s | Good | Week 0 |
| Phase 1 | Trot (optimized) | 1.2 m/s | Good | Week 2 |
| Phase 2 | Gallop | 2.0 m/s | Moderate | Week 4 |
| Phase 3 | Gallop + avoid | 1.8 m/s | Good | Week 6 |
| Stretch | Gallop (max) | 2.5+ m/s | Risky | Week 8 |

**World record** (comparable robots):
- MIT Mini Cheetah: **2.45 m/s** (research platform, $10K+)
- Ghost Robotics Vision 60: **1.8 m/s** (military, $100K)
- Boston Dynamics Spot: **1.6 m/s** (commercial, $75K)
- **PiDog target**: **2.0 m/s** (hobbyist, $200)

---

### Success Metrics

**Quantitative**:
- Top speed (m/s) on flat terrain
- Stability (% episodes without falls)
- Gait efficiency (Cost of Transport: J/kg/m)
- Obstacle avoidance (% successful runs)

**Qualitative**:
- Visual gait quality (smooth, natural)
- Dog-likeness (user survey)
- Robustness (handles pushes, uneven terrain)

---

## 8. Next Steps

### Immediate Actions (This Week)

1. **Start with IMU sensor**:
   ```bash
   # Edit pidog.urdf to add IMU sensor (Section 2.1)
   # Add world plugin for gz-sim-imu-system
   # Test in Gazebo, verify /imu topic
   ```

2. **Add contact sensors**:
   ```bash
   # Add contact sensors to 4 feet (Section 2.2)
   # Verify /contact/* topics
   ```

3. **Update RL environment**:
   ```bash
   # Modify pidog_rl_env.py observation space
   # Add IMU and contact subscribers
   # Test observation collection
   ```

4. **Collect baseline data**:
   ```bash
   # Run enhanced data collector with new sensors
   # Verify sensor data is recorded
   ```

5. **Optimize trot gait**:
   ```bash
   # Edit trot_gait.py parameters (Section 4.2)
   # Test manually, measure speed with RViz
   ```

---

### Code Files to Create/Modify

**New files**:
- `pidog_gaits/pidog_gaits/gallop_gait.py` - Gallop gait generator
- `pidog_gaits/pidog_gaits/sensor_gait_net_lstm.py` - Extended neural network

**Modified files**:
- `pidog_description/urdf/pidog.urdf` - Add IMU, contacts, sonar, camera
- `pidog_description/worlds/pidog.sdf` - Add world sensor plugins
- `pidog_gaits/pidog_gaits/pidog_rl_env.py` - Extend observation space
- `pidog_gaits/pidog_gaits/trot_gait.py` - Optimize parameters
- `pidog_gaits/pidog_gaits/gait_generator_node.py` - Add gallop gait
- `pidog_gaits/launch/gait_demo.launch.py` - Add sensor bridge nodes

---

## 9. References

**Academic Papers**:
1. "Learning Quadrupedal High-Speed Running on Uneven Terrain" (2024) - PMC10813166
   - Trotting at 1.8 m/s, PPO with IMU feedback

2. "A Quadruped Robot Exhibiting Spontaneous Gait Transitions" (2017) - Nature
   - Walk→Trot→Gallop transitions emerge from CPG + sensory feedback

3. "Viability leads to emergence of gait transitions" (2024) - PMC11271497
   - Gait transitions improve energy efficiency and prevent falls

4. "Foot trajectory as key factor for diverse gait patterns" (2024) - Nature
   - Trotting, pacing, galloping achieved via foot trajectory shaping

**Implementation Resources**:
- Gazebo Harmonic Sensors: https://gazebosim.org/docs/harmonic/sensors/
- MOGI-ROS Gazebo Sensors Tutorial: https://github.com/MOGI-ROS/Week-5-6-Gazebo-sensors
- SunFounder PiDog Docs: https://docs.sunfounder.com/projects/pidog/

**Quadruped Benchmarks**:
- MIT Cheetah: 6 m/s sprint, 2.45 m/s sustained
- ANYmal: 1.0 m/s outdoor navigation
- Unitree A1: 1.6 m/s, 11.4 kg, $10K

---

## 10. Summary

**Current State**:
- PiDog has rich sensor suite in hardware
- Simulation has ZERO sensors (only ground-truth)
- RL training uses simplified observations
- Top speed: ~0.8 m/s (conservative trot)

**Proposed State**:
- Full sensor simulation (IMU, contacts, sonar, camera)
- Sensor-driven RL with 50+ dimensional observation
- Optimized trot (1.2 m/s) + gallop gait (2.0+ m/s)
- Dog-like behaviors: obstacle avoidance, head tracking, playfulness

**Development Time**: 8-10 weeks (part-time)

**Difficulty**:
- Phase 1 (IMU + contacts): **Moderate** - well-established methods
- Phase 2 (Gallop): **High** - requires gait design expertise
- Phase 3 (Sonar): **Low** - straightforward integration
- Phase 4 (Vision): **Very High** - computationally intensive, complex

**Recommended Approach**:
1. Start with **Phase 1** (IMU + contacts) - highest ROI
2. Test trot optimization - quick win
3. Implement gallop gait - high risk, high reward
4. Add sonar only if needed (gallop may be fast enough)
5. Vision is optional (nice-to-have, not critical for speed)

---

**Let's make PiDog run like a real dog!** 🐕💨
