# Correct PiDog Sensor Layout

Based on real SunFounder PiDog hardware specifications.

## Head Sensor Configuration (Front View)

```
                    [Touch Sensor]
                  (behind the eyes)
                   _______________
                  |               |
                  |   /   |   \   |  Camera (nose)
                  |  ( ) | ( )   |  Eyes = HC-SR04 Ultrasonic
                  |   \___|___/  |  (2 transducers look like eyes)
                  |_______________|
                        HEAD
```

## Sensor Positions (Relative to Head Link)

### 1. Ultrasonic Sensor (HC-SR04) - "The Eyes"
**Location:** Front of head at eyes position
**Position:** `xyz="0.0 0.0 0.036"` (same as eyes link)
**Visual:** Two silver cylinders (8mm diameter each, spaced 12mm apart)
- Left transducer: `y=+0.006`
- Right transducer: `y=-0.006`

**ROS2 Topic:** `/ultrasonic` (sensor_msgs/LaserScan)

**Why it looks like eyes:**
- HC-SR04 has 2 cylindrical transducers (transmitter + receiver)
- They're positioned like eyes on the front of the head
- This is the actual design of the real PiDog!

---

### 2. Touch Sensor (Single)
**Location:** Behind the eyes/ultrasonic, on top of head
**Position:** `xyz="-0.005 0 0.037"`
- 5mm behind the ultrasonic sensor
- Same height as eyes (37mm up from head origin)

**ROS2 Topic:** `/touch_sensor/contacts` (gazebo_msgs/Contacts)

**Visual:** Small red box (15mm × 20mm × 1mm) - semi-transparent

**Purpose:** Detects when someone pets the dog's head

---

### 3. Camera (OV5647)
**Location:** Nose (front-center, slightly down from eyes)
**Position:** `xyz="0.03 0 0.02"` (relative to head)

**ROS2 Topic:** `/camera` (sensor_msgs/Image)

**Visual:** Small black box (5mm × 10mm × 5mm)

**Purpose:** Vision for RL, object detection, navigation

---

### 4. IMU (MPU6050)
**Location:** Center of body, 2cm up
**Position:** `xyz="0 0 0.02"` (relative to body link)

**ROS2 Topic:** `/imu` (sensor_msgs/Imu)

**Purpose:** Balance, orientation, motion tracking

---

### 5. Microphone Array (3 mics)
**Location:** Around the head (triangular array)
**Positions (relative to head):**
- Left: `xyz="0.02 0.015 0.025"`
- Center: `xyz="0.025 0 0.025"`
- Right: `xyz="0.02 -0.015 0.025"`

**Visual:** Three small black spheres (2mm radius)

**Status:** Visual markers only (audio simulation TBD)

---

### 6. RGB LED Strip
**Location:** Front chest
**Position:** `xyz="0.045 0 0.01"` (relative to body)

**Visual:** Blue glowing panel (40mm × 20mm)

**Purpose:** Visual feedback, emotions (WS2812B in real hardware)

---

### 7. Speaker
**Location:** Bottom center of body
**Position:** `xyz="-0.03 0 -0.015"` (relative to body)

**Visual:** Dark gray disc (8mm diameter)

**Purpose:** Audio output (visual representation in simulation)

---

## Key Corrections from Previous Version

### ❌ What was wrong:
- Had 2 touch sensors (head + body) ❌
- Ultrasonic was separate from eyes ❌
- Eyes were decorative, ultrasonic was elsewhere ❌

### ✅ What's correct now:
- **1 touch sensor** on head behind eyes ✅
- **Ultrasonic sensor IS the eyes** (HC-SR04 transducers) ✅
- **Camera on nose** (separate from eyes) ✅

---

## Sensor Summary Table

| Sensor | Count | Location | ROS2 Topic | Purpose |
|--------|-------|----------|------------|---------|
| **IMU** | 1 | Body center | `/imu` | Balance/orientation |
| **Camera** | 1 | Nose | `/camera` | Vision/navigation |
| **Ultrasonic** | 1 (2 transducers) | Eyes position | `/ultrasonic` | Obstacle detection |
| **Touch** | 1 | Behind eyes | `/touch_sensor/contacts` | Petting detection |
| **Microphones** | 3 | Around head | N/A (visual) | Sound direction |
| **RGB LED** | 1 strip | Chest | N/A (visual) | Visual feedback |
| **Speaker** | 1 | Body bottom | N/A (visual) | Audio output |

---

## Why HC-SR04 Looks Like Eyes

The HC-SR04 ultrasonic sensor has a distinctive appearance:

```
  ___________________
 |  _____   _____   |
 | |     | |     |  |  ← Two cylindrical transducers
 | | (T) | | (R) |  |    T = Transmitter (sends ultrasonic)
 | |_____| |_____|  |    R = Receiver (receives echo)
 |___________________|
      HC-SR04
```

When mounted on a robot head, these two transducers naturally look like **eyes**!

**Real PiDog Design:**
- The HC-SR04's two transducers serve as the dog's "eyes"
- Camera is positioned on the "nose" below
- Touch sensor is behind/between the "eyes" for petting
- This creates a friendly, dog-like face appearance!

---

## Testing Sensor Layout

```bash
# Launch Gazebo
ros2 launch pidog_description gazebo.launch.py

# Check sensor topics
ros2 topic list | grep -E 'ultrasonic|touch|camera|imu'

# Expected output:
# /camera
# /imu
# /ultrasonic
# /touch_sensor/contacts

# Test each sensor
ros2 topic echo /ultrasonic --once          # Should show distance
ros2 topic echo /touch_sensor/contacts      # Click on red box in Gazebo
ros2 topic echo /camera --once              # Should show image data
ros2 topic echo /imu --once                 # Should show orientation
```

---

## Visual Appearance in Gazebo

When you launch the simulation, you'll see:

**Head:**
- 👀 **Two silver "eyes"** = HC-SR04 ultrasonic transducers
- 📷 **Small black box on nose** = Camera
- 🔴 **Small red box behind eyes** = Touch sensor (semi-transparent)
- 🎤 **Three tiny black dots** = Microphones

**Body:**
- 💡 **Blue glowing panel on chest** = RGB LED strip
- 🔊 **Dark disc on bottom** = Speaker

**This matches the real PiDog's cute, friendly appearance!** 🐕

---

## RL Observation Space with Correct Sensors

```python
observation = {
    'image': camera_image,              # 84×84×3 RGB from nose camera
    'vector': [
        gait_command,                   # 4D
        joint_positions,                # 12D
        joint_velocities,               # 12D
        body_position,                  # 3D
        body_orientation,               # 4D (quat)
        imu_orientation,                # 4D (quat)
        imu_angular_velocity,           # 3D
    ],                                  # Total: 42D
    'ultrasonic': distance_reading,     # 1D (from eyes)
    'touch': is_being_petted,           # 1D (binary)
}
```

**Total:** Vision (84×84×3) + Proprioception (42D) + Exteroception (2D) = **Complete dog sensory system!**

---

**The sensor layout now perfectly matches the real SunFounder PiDog!** ✅🐕
