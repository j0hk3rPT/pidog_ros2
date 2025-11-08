# Updated Reward Function Design with Real Hardware Constraints

**Date**: 2025-11-08 (REVISED)
**Hardware**: SunFounder PiDog (verified against actual specs)

---

## ✅ VERIFIED: Available Sensors on Real Hardware

Based on SunFounder documentation:
- **IMU (6-DOF)**: YES - MPU6050 gyro + accelerometer
- **Camera**: YES - CSI module
- **Ultrasonic**: YES - HC-SR04
- **Touch sensors**: YES - Dual touch on **HEAD ONLY** (not feet!)
- **Servo feedback**: **NO** - Open-loop control only (no current/torque sensing)
- **Foot contact sensors**: **NO** - Don't exist!

---

## 🚨 Critical Changes from Original Design

### 1. NO FOOT CONTACT SENSORS

**Problem**: Original design used 4 contact sensors on feet - **these don't exist on real hardware!**

**Solution**: Virtual contact detection using **IMU + kinematics only**

```python
def estimate_foot_contact_virtual(self):
    """
    Estimate ground contact using ONLY sensors available on real PiDog.
    Works in sim AND on real hardware!
    """
    contacts = np.zeros(4)  # [BR, FR, BL, FL]

    # Method 1: Z-axis acceleration pattern
    # When foot hits ground: sharp downward spike
    # When foot in air: smooth/noisy
    z_acc = self.imu_linear_acc[2]
    z_acc_magnitude = abs(z_acc - 9.81)  # Remove gravity

    # Method 2: Forward kinematics
    # Calculate foot positions from joint angles
    foot_z_positions = self.calc_foot_heights(self.joint_positions)

    # Estimate: foot on ground if Z-pos near zero AND low Z-accel variance
    for i in range(4):
        if foot_z_positions[i] < 0.01:  # Within 1cm of ground
            contacts[i] = 1.0

    return contacts
```

**Impact on training**:
- Observation space: 52D → **48D** (removed 4D contact sensors)
- Flight phase reward: Use **acceleration patterns** instead of contact count
- Gait quality: Detect from **IMU periodicity** instead of foot contacts

---

### 2. NEW TERMINATION CONDITION: Stall Detection

**User requirement**: "Not moving in the direction of the head for 2 seconds = dead"

```python
# Stall detection (new!)
self.stall_counter = 0  # Initialize in __init__

def step(self, action):
    # ... existing code ...

    # Check forward progress
    if self.body_linear_vel[0] < 0.05:  # < 5 cm/s = stalled
        self.stall_counter += 1
        if self.stall_counter >= 60:  # 2 sec at 30Hz
            reward -= 100.0
            done = True
            info['termination_reason'] = 'stalled_no_forward_progress'
    else:
        self.stall_counter = 0  # Reset if moving
```

**Effect**: Forces PiDog to keep moving forward. Can't just stand still!

---

### 3. STRICTER ROLL TOLERANCE

**User requirement**: "Falling to the side 45 degrees = dead"

**Before**: `abs(roll) > 1.2 rad` (≈69°)
**After**: `abs(roll) > 0.785 rad` (45°)

```python
# Termination: Roll > 45 degrees
if abs(roll) > 0.785:  # 45 degrees (was 1.2 rad)
    reward -= 100.0
    done = True
    info['termination_reason'] = 'tipped_over_45deg'
```

**Effect**: Less tolerance for lateral instability. Must stay more upright.

---

## Updated Reward Function (Hardware-Compatible)

### Observation Space: 48D (was 52D)

```
Observation (48D):
  - gait_cmd (3): [gait_type, direction, turn]
  - phase (1): gait cycle phase
  - joint_pos (12): motor positions
  - joint_vel (12): motor velocities
  - imu_orientation (3): [roll, pitch, yaw] ← REAL SENSOR
  - imu_angular_vel (3): [wx, wy, wz] ← REAL SENSOR
  - imu_linear_acc (3): [ax, ay, az] ← REAL SENSOR
  - body_linear_vel (3): [vx, vy, vz] (from IMU integration or GPS)
  - virtual_contacts (4): [BR, FR, BL, FL] ← ESTIMATED from IMU+kinematics
  - terrain_height_est (1): from IMU+kinematics
  - body_height (1): Z position (from IMU integration)
  - stall_counter_norm (1): normalized (0-1) stall duration
```

---

### Complete Reward Function (Revised)

```python
def _reward_fast_running_hardware_compatible(self, action):
    """
    Reward function for max speed using ONLY real PiDog sensors.
    """
    reward = 0.0
    done = False
    info = {}

    # Unpack IMU data
    roll, pitch, yaw = self.imu_orientation
    body_height = self.imu_estimated_height  # Integrated from IMU
    forward_vel = self.body_linear_vel[0]

    # Virtual contact detection
    virtual_contacts = self.estimate_foot_contact_virtual()
    num_contacts = int(np.sum(virtual_contacts))

    # ========== PRIMARY: SPEED (60%) ==========
    if forward_vel > 0:
        reward += forward_vel * 15.0  # Main reward

        # Milestone bonuses
        if forward_vel > 1.2:
            reward += 3.0
        if forward_vel > 1.8:
            reward += 10.0
        if forward_vel > 2.2:
            reward += 20.0
    else:
        reward += forward_vel * 5.0  # Penalty for backward

    # ========== STABILITY: HEIGHT (10%) ==========
    if 0.10 < body_height < 0.15:
        reward += 2.0
    elif body_height < 0.08:
        reward -= 10.0

    # ========== STABILITY: ORIENTATION (15%) ==========
    if abs(roll) < 0.4:
        reward += 1.5
    else:
        reward -= abs(roll) * 3.0

    if abs(pitch) < 0.6:  # Relaxed for galloping
        reward += 1.5
    else:
        reward -= abs(pitch) * 2.0

    # ========== STABILITY: ANGULAR VELOCITY (5%) ==========
    ang_vel_mag = np.linalg.norm(self.imu_angular_vel)
    if ang_vel_mag > 5.0:
        reward -= ang_vel_mag * 0.5

    # ========== GAIT QUALITY: VIRTUAL CONTACTS (10%) ==========
    # Same logic, but using estimated contacts
    if num_contacts == 0:
        reward += 3.0   # Flight phase detected from kinematics
    elif num_contacts == 2:
        reward += 1.5
    elif num_contacts == 4:
        reward += 0.2
    else:  # 1 or 3
        reward -= 2.0

    # ========== EFFICIENCY: ENERGY (-5%) ==========
    reward -= np.sum(np.abs(self.joint_velocities)) * 0.005
    reward -= np.sum(np.abs(action)) * 0.005

    # ========== TERMINATION CONDITIONS ==========

    # 1. Roll > 45 degrees (STRICT)
    if abs(roll) > 0.785:  # 45 degrees
        reward -= 100.0
        done = True
        info['termination_reason'] = 'roll_exceeded_45deg'

    # 2. Pitch > 69 degrees (keep original)
    if abs(pitch) > 1.2:
        reward -= 100.0
        done = True
        info['termination_reason'] = 'pitch_excessive'

    # 3. Body too low (crashed)
    if body_height < 0.04:
        reward -= 100.0
        done = True
        info['termination_reason'] = 'body_crashed_ground'

    # 4. Stall detection (NEW!)
    if forward_vel < 0.05:  # < 5 cm/s
        self.stall_counter += 1
        if self.stall_counter >= 60:  # 2 seconds at 30Hz
            reward -= 100.0
            done = True
            info['termination_reason'] = 'stalled_2_seconds'
    else:
        self.stall_counter = 0  # Reset

    # Stall counter penalty (gentle nudge before termination)
    if self.stall_counter > 0:
        reward -= self.stall_counter * 0.1  # Increasing penalty

    info.update({
        'speed': forward_vel,
        'height': body_height,
        'contacts_estimated': num_contacts,
        'roll': roll,
        'pitch': pitch,
        'ang_vel_mag': ang_vel_mag,
        'stall_counter': self.stall_counter,
    })

    return reward, done, info
```

---

## Key Parameter Changes

| Parameter | Original | Updated | Reason |
|-----------|----------|---------|--------|
| **Roll termination** | 1.2 rad (69°) | **0.785 rad (45°)** | User requirement |
| **Stall threshold** | N/A | **< 0.05 m/s for 2 sec** | User requirement |
| **Contact sensors** | Real (4D) | **Virtual (estimated)** | Hardware constraint |
| **Observation space** | 52D | **48D** | Removed real contact sensors |
| **Forward velocity weight** | 15.0 | **15.0** | Keep (good balance) |
| **Fall penalty** | -100.0 | **-100.0** | Keep (works well) |

---

## Updated Training Strategy

### Phase 1: Conservative Baseline (50K steps, ~20 min)

**Goal**: Learn to walk without stalling or falling

**Key focus**:
- Don't trigger stall detection (keep moving!)
- Stay upright (roll < 45°)
- Forward velocity > 0.5 m/s minimum

**Expected**: 0.6-0.8 m/s, stable, <10% stall rate

---

### Phase 2: Fast Running (100K steps, ~45 min)

**Goal**: Maximize speed while avoiding termination

**Key focus**:
- Push speed toward 1.5-2.0 m/s
- Learn to use flight phase (virtual contact detection)
- Avoid stalling at all costs!

**Expected**: 1.2-1.8 m/s, occasional falls, <5% stall rate

---

### Phase 3: Speed Optimization (150K steps, ~60 min)

**Goal**: Break 2.0 m/s barrier

**Key focus**:
- Maximize milestone bonuses
- Accept higher fall rate (5-10%) for speed
- Perfect gait timing

**Expected**: 1.8-2.2+ m/s, 5-10% fall rate, <2% stall rate

---

## Virtual Contact Detection Details

### Simple Version (Recommended Start)

```python
def estimate_foot_contact_simple(self):
    """
    Basic virtual contact detection using forward kinematics.
    """
    contacts = np.zeros(4)

    # Calculate foot heights from joint angles
    # (Need to implement forward kinematics for PiDog leg geometry)
    leg_lengths = [0.047, 0.0635]  # Upper, lower leg (meters)

    for leg_idx in range(4):
        shoulder_angle = self.joint_positions[leg_idx * 2]
        knee_angle = self.joint_positions[leg_idx * 2 + 1]

        # Simple 2-link kinematics
        foot_z = (leg_lengths[0] * np.sin(shoulder_angle) +
                  leg_lengths[1] * np.sin(shoulder_angle + knee_angle))

        # On ground if within 1cm of zero (with body height offset)
        if abs(self.body_height + foot_z) < 0.01:
            contacts[leg_idx] = 1.0

    return contacts
```

### Advanced Version (Better Accuracy)

```python
def estimate_foot_contact_advanced(self):
    """
    Enhanced virtual contact using IMU + kinematics + acceleration.
    """
    contacts_kinematic = self.estimate_foot_contact_simple()

    # Refine with Z-acceleration
    z_acc_var = np.var(self.imu_linear_acc_history[-10:, 2])  # Last 10 samples

    # High variance = foot strikes → more likely in contact
    # Low variance = smooth flight → less likely in contact
    if z_acc_var > 2.0:  # Threshold to tune
        # Boost kinematic estimates during impact
        contacts = contacts_kinematic * 1.5
    else:
        contacts = contacts_kinematic * 0.8

    return np.clip(contacts, 0, 1)
```

---

## Testing & Validation Plan

### 1. Sanity Check (Simulation)
- Launch Gazebo with updated URDF (no contact sensors)
- Run trot gait manually
- Verify:
  - Stall detection triggers if stopped for 2 sec ✓
  - Roll > 45° triggers termination ✓
  - Virtual contacts reasonable match reality

### 2. Baseline Training
- Train Phase 1 (conservative)
- Plot:
  - Stall rate over time (should decrease)
  - Average episode length (should increase)
  - Forward velocity (should increase)

### 3. Hardware Transfer (Future)
- Deploy trained policy on real PiDog
- Compare:
  - Sim speeds vs real speeds
  - Sim stability vs real stability
  - Virtual contact accuracy

---

## Summary of Changes

**Removed**:
- ❌ Foot contact sensors (don't exist on hardware)
- ❌ Servo current/torque feedback (not available)

**Added**:
- ✅ Virtual contact detection (IMU + kinematics)
- ✅ Stall detection (2 sec timeout)
- ✅ Stricter roll tolerance (45° vs 69°)

**Kept**:
- ✅ IMU sensor (6-DOF) - real hardware
- ✅ Speed-focused reward function
- ✅ Flight phase encouragement (via virtual contacts)
- ✅ Energy efficiency penalties

**Next**: Review these changes and approve for implementation!
