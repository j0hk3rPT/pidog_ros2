# PiDog Fast Running Reward Function Design

**Date**: 2025-11-08
**Goal**: Design reward function to train PiDog to run as fast as possible while maintaining stability

---

## Overview

The reward function is the "brain" that teaches PiDog what behaviors are good and bad. Every timestep (~30Hz), PiDog receives a reward (positive points) or penalty (negative points) based on its current state and actions. The neural network learns to maximize total reward over time.

**Key principle**: Balance speed vs. stability
- Too much speed reward → PiDog falls trying to go fast
- Too much stability reward → PiDog stands still (safe but boring!)
- **Sweet spot**: Reward high speed BUT heavily penalize falling

---

## Reward Function Components

### PRIMARY GOAL: SPEED (60% of total reward)

**Objective**: Make PiDog run forward as fast as possible

```python
# Forward velocity reward (MAIN DRIVER)
forward_vel = imu_linear_vel_x  # From IMU sensor (m/s)

if forward_vel > 0:
    speed_reward = forward_vel * 15.0  # ← TUNABLE WEIGHT

    # Bonus milestones (encourage breaking speed barriers)
    if forward_vel > 1.2:  # Fast trot
        speed_reward += 3.0
    if forward_vel > 1.8:  # Galloping speed
        speed_reward += 10.0
    if forward_vel > 2.2:  # VERY fast
        speed_reward += 20.0
else:
    # Penalty for going backward
    speed_reward = forward_vel * 5.0  # Negative reward
```

**Why this works**:
- Linear scaling: Faster = more reward
- Bonuses create "plateaus" to aim for
- 15.0 weight means 1.5 m/s = +22.5 points per step
- At 30 Hz, that's +675 points/second when running fast!

**Tunable parameters**:
- `forward_vel_weight`: 15.0 (higher = prioritize speed more)
- `milestone_1.2m/s_bonus`: 3.0
- `milestone_1.8m/s_bonus`: 10.0
- `milestone_2.2m/s_bonus`: 20.0

---

### SAFETY CONSTRAINTS: STABILITY (30% of total reward)

**Objective**: Prevent falling, maintain upright posture

#### 1. Body Height (Z position)

```python
# Keep body at 10-15cm height (good for running)
body_height = imu_position_z  # meters

if 0.10 < body_height < 0.15:
    height_reward = 2.0  # Good height
elif body_height < 0.08:
    height_reward = -10.0  # Too low (dragging belly)
elif body_height > 0.18:
    height_reward = -5.0  # Too high (unstable)
else:
    height_reward = 0.0
```

**Tunable parameters**:
- `target_height_min`: 0.10 m
- `target_height_max`: 0.15 m
- `good_height_reward`: 2.0
- `too_low_penalty`: -10.0

---

#### 2. Body Orientation (Roll & Pitch from IMU)

```python
# Roll = side-to-side tilt
# Pitch = forward-backward tilt
roll, pitch, yaw = imu_orientation  # radians

# Allow more pitch variation during fast running
# (dogs lean forward when sprinting)
pitch_tolerance = 0.6  # ±34 degrees
roll_tolerance = 0.4   # ±23 degrees

orientation_reward = 0.0

if abs(roll) < roll_tolerance:
    orientation_reward += 1.5
else:
    orientation_reward -= abs(roll) * 3.0  # Penalize rolling

if abs(pitch) < pitch_tolerance:
    orientation_reward += 1.5
else:
    orientation_reward -= abs(pitch) * 2.0  # Penalize pitching
```

**Tunable parameters**:
- `roll_tolerance`: 0.4 rad (±23°)
- `pitch_tolerance`: 0.6 rad (±34°) - relaxed for galloping
- `good_orientation_reward`: 1.5 each
- `roll_penalty_weight`: 3.0
- `pitch_penalty_weight`: 2.0

**Why relaxed pitch?** During galloping, dogs naturally pitch forward during flight phase, then backward when landing. We allow this natural motion.

---

#### 3. IMU Angular Velocity (Stability)

```python
# Penalize rapid spinning/wobbling
ang_vel_x, ang_vel_y, ang_vel_z = imu_angular_velocity  # rad/s

ang_vel_magnitude = sqrt(ang_vel_x² + ang_vel_y² + ang_vel_z²)

if ang_vel_magnitude > 5.0:  # Spinning too fast
    ang_vel_penalty = -ang_vel_magnitude * 0.5
else:
    ang_vel_penalty = 0.0
```

**Tunable parameters**:
- `max_safe_ang_vel`: 5.0 rad/s
- `ang_vel_penalty_weight`: 0.5

---

### GAIT QUALITY: CONTACT PATTERNS (10% of total reward)

**Objective**: Encourage proper galloping gait with flight phase

```python
# Contact sensors: [back_right, front_right, back_left, front_left]
feet_contacts = [br_contact, fr_contact, bl_contact, fl_contact]  # booleans
num_contacts = sum(feet_contacts)

# Gallop pattern: 0, 2, or 4 feet touching ground
# 0 = flight phase (GOOD for speed)
# 2 = diagonal pair or front/back pair (GOOD)
# 1 or 3 = asymmetric/unstable (BAD)
# 4 = all feet down (OK during stance)

if num_contacts == 0:
    # FLIGHT PHASE - excellent for galloping!
    contact_reward = 3.0
elif num_contacts == 2:
    # Good push-off or landing
    contact_reward = 1.5
elif num_contacts == 4:
    # All feet down - stable but slow
    contact_reward = 0.2
else:  # 1 or 3 feet
    # Bad balance
    contact_reward = -2.0
```

**Tunable parameters**:
- `flight_phase_reward`: 3.0 (encourage airtime)
- `two_feet_reward`: 1.5
- `four_feet_reward`: 0.2 (small positive, not penalized)
- `asymmetric_penalty`: -2.0

**Why reward flight phase?** Research shows galloping quadrupeds achieve max speed with significant airtime (0 ground contacts). This is the signature of fast running!

---

### EFFICIENCY: ENERGY COST (penalty, -5% to -10%)

**Objective**: Prevent thrashing, encourage smooth movements

```python
# Joint velocity penalty (don't flail)
joint_vels = [motor_0_vel, motor_1_vel, ..., motor_7_vel]  # 8 leg joints
joint_vel_penalty = sum(abs(v) for v in joint_vels) * 0.005

# Action smoothness (don't make huge sudden commands)
action_penalty = sum(abs(a) for a in action) * 0.005

total_efficiency_penalty = -(joint_vel_penalty + action_penalty)
```

**Tunable parameters**:
- `joint_vel_penalty_weight`: 0.005 (small, speed matters more)
- `action_penalty_weight`: 0.005

**Why so small?** We want speed, and speed requires fast joint movements. This is just a gentle nudge toward smoothness.

---

### TERMINATION: CATASTROPHIC FAILURE (immediate episode end)

**Objective**: End episode early if robot falls

```python
# Check for fall conditions
fallen = False

if abs(roll) > 1.2 or abs(pitch) > 1.2:  # ±69 degrees
    fallen = True
    fall_penalty = -100.0

if body_height < 0.04:  # Body touching ground
    fallen = True
    fall_penalty = -100.0

if head_contact:  # Head touching ground (ouch!)
    fallen = True
    fall_penalty = -100.0

return reward + fall_penalty, fallen
```

**Tunable parameters**:
- `max_roll_before_fall`: 1.2 rad
- `max_pitch_before_fall`: 1.2 rad
- `min_height_before_fall`: 0.04 m
- `fall_penalty`: -100.0 (catastrophic)

**Why -100?** This is a HUGE penalty that ends the episode. It teaches PiDog: "Don't fall, no matter what!"

---

## Complete Reward Function (Pseudocode)

```python
def calculate_reward_fast_running(self):
    reward = 0.0
    done = False

    # ========== PRIMARY: SPEED (60%) ==========
    forward_vel = self.imu_linear_vel[0]  # X-axis

    if forward_vel > 0:
        reward += forward_vel * 15.0  # Main reward
        if forward_vel > 1.2:
            reward += 3.0   # Milestone 1
        if forward_vel > 1.8:
            reward += 10.0  # Milestone 2
        if forward_vel > 2.2:
            reward += 20.0  # Milestone 3
    else:
        reward += forward_vel * 5.0  # Penalty for backward

    # ========== STABILITY: HEIGHT (10%) ==========
    if 0.10 < self.body_height < 0.15:
        reward += 2.0
    elif self.body_height < 0.08:
        reward -= 10.0

    # ========== STABILITY: ORIENTATION (15%) ==========
    roll, pitch, yaw = self.imu_orientation

    if abs(roll) < 0.4:
        reward += 1.5
    else:
        reward -= abs(roll) * 3.0

    if abs(pitch) < 0.6:
        reward += 1.5
    else:
        reward -= abs(pitch) * 2.0

    # ========== STABILITY: ANGULAR VEL (5%) ==========
    ang_vel_mag = np.linalg.norm(self.imu_angular_vel)
    if ang_vel_mag > 5.0:
        reward -= ang_vel_mag * 0.5

    # ========== GAIT QUALITY: CONTACTS (10%) ==========
    num_contacts = sum(self.foot_contacts)

    if num_contacts == 0:
        reward += 3.0   # Flight phase!
    elif num_contacts == 2:
        reward += 1.5
    elif num_contacts == 4:
        reward += 0.2
    else:  # 1 or 3
        reward -= 2.0

    # ========== EFFICIENCY: ENERGY (-5%) ==========
    reward -= np.sum(np.abs(self.joint_velocities)) * 0.005
    reward -= np.sum(np.abs(action)) * 0.005

    # ========== TERMINATION: FALL CHECK ==========
    if abs(roll) > 1.2 or abs(pitch) > 1.2:
        reward -= 100.0
        done = True

    if self.body_height < 0.04:
        reward -= 100.0
        done = True

    if self.head_contact:
        reward -= 100.0
        done = True

    return reward, done, {
        'speed': forward_vel,
        'height': self.body_height,
        'contacts': num_contacts,
        'roll': roll,
        'pitch': pitch,
    }
```

---

## Expected Reward Ranges

**Good running episode** (1.5 m/s, stable):
- Speed: +22.5/step (1.5 * 15.0)
- Milestone: +3.0/step
- Height: +2.0/step
- Orientation: +3.0/step (roll + pitch)
- Contacts: +1.5/step (average)
- Energy: -0.5/step
- **Total: ~+31.5 points/step**
- **Per second**: +945 points (at 30 Hz)
- **Per episode** (500 steps): +15,750 points

**Failed episode** (falls after 50 steps):
- Speed: +10/step × 50 = +500
- Stability: ~+100 (before falling)
- Fall penalty: -100
- **Total: ~+500 points** (much worse!)

**This teaches**: Run fast, but DON'T FALL!

---

## Tuning Strategy

### Phase 1: Learn to Walk Stable (Baseline)

Use conservative weights to learn balance first:

```python
WEIGHTS_CONSERVATIVE = {
    'forward_vel': 5.0,      # Low (focus on stability)
    'height_reward': 5.0,    # High
    'orientation_reward': 3.0,  # High
    'flight_reward': 0.5,    # Low (don't encourage flight yet)
    'fall_penalty': -100.0,  # Keep high
}
```

Train for 50K steps → Should achieve 0.8-1.0 m/s

---

### Phase 2: Encourage Speed (Our Plan)

Increase speed reward, relax stability:

```python
WEIGHTS_FAST_RUNNING = {
    'forward_vel': 15.0,     # HIGH (main goal)
    'height_reward': 2.0,    # Lower
    'orientation_reward': 1.5,  # Lower
    'roll_tolerance': 0.4,   # Relaxed
    'pitch_tolerance': 0.6,  # Very relaxed
    'flight_reward': 3.0,    # HIGH (encourage gallop)
    'fall_penalty': -100.0,  # Keep high
}
```

Train for 100K steps → Target 1.5-2.0 m/s

---

### Phase 3: Maximum Speed (Risky)

Push the limits:

```python
WEIGHTS_MAX_SPEED = {
    'forward_vel': 25.0,     # VERY HIGH
    'height_reward': 1.0,    # Low (allow crouching)
    'orientation_reward': 1.0,  # Low
    'roll_tolerance': 0.5,   # More relaxed
    'pitch_tolerance': 0.8,  # Very relaxed
    'flight_reward': 5.0,    # VERY HIGH
    'fall_penalty': -100.0,  # Keep high
}
```

Train for 150K steps → Target 2.2+ m/s (may be unstable)

---

## Questions for You to Consider

**1. Speed vs. Stability Trade-off**
- How important is it that PiDog never falls? (e.g., running indoors vs. outdoor grass)
- Are you OK with occasional falls if it means higher top speed?

**2. Gait Preference**
- Do you want to force galloping, or let it emerge naturally?
- Should we reward specific foot patterns (diagonal trot vs. rotary gallop)?

**3. Training Time**
- Willing to train longer (200K+ steps) for better performance?
- Or prefer faster results (50K steps) with "good enough" speed?

**4. Tuning Knobs - Which Matter Most?**
Here are the **TOP 5** most important parameters to tune:

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `forward_vel_weight` | 15.0 | Higher = more speed, more risk |
| `fall_penalty` | -100.0 | Higher = more cautious |
| `flight_reward` | 3.0 | Higher = more airtime |
| `pitch_tolerance` | 0.6 rad | Higher = allow forward lean |
| `milestone_1.8m/s_bonus` | 10.0 | Creates "speed barrier" to break |

**My recommendation**: Start with the values shown (Phase 2), then we can tune based on what you see in training!

---

## Visualization During Training

We'll plot these metrics every 1000 steps:

1. **Average speed** (m/s) - should increase over time
2. **Episode length** - longer = more stable (max 500 steps)
3. **Total reward** - should increase
4. **Contact pattern** - histogram of 0/1/2/3/4 feet
5. **Fall rate** - % episodes ending in fall

**Target metrics after 100K steps**:
- Avg speed: 1.5-2.0 m/s
- Avg episode length: 400+ steps (stable)
- Total reward: 15,000+ per episode
- Flight phase: 20%+ of time (0 contacts)
- Fall rate: <5%

---

## Next Steps

1. **Review this design** - tell me what you think!
2. **Adjust weights** if you want different priorities
3. **Implement in `pidog_rl_env.py`**
4. **Train Phase 1** (conservative, 50K steps, ~20 min)
5. **Evaluate** - watch videos, check metrics
6. **Tune and iterate**!

---

## Summary Table: Reward Components

| Component | Weight | Purpose | Typical Value |
|-----------|--------|---------|---------------|
| Forward velocity | 15.0 | **Main goal** | +15-30/step |
| Speed milestone 1.2 | 3.0 | Encourage trot | +3/step |
| Speed milestone 1.8 | 10.0 | Encourage gallop | +10/step |
| Speed milestone 2.2 | 20.0 | Maximum speed | +20/step |
| Body height | 2.0 | Stay upright | +2/step |
| Roll stability | 1.5 | Don't tip sideways | +1.5/step |
| Pitch stability | 1.5 | Allow forward lean | +1.5/step |
| Angular velocity | -0.5 | Don't spin | -0 to -3/step |
| Flight phase (0 feet) | 3.0 | **Gallop signature** | +3/step |
| Two feet contact | 1.5 | Good push-off | +1.5/step |
| Four feet contact | 0.2 | Stable stance | +0.2/step |
| Asymmetric contact | -2.0 | Bad balance | -2/step |
| Joint velocity | -0.005 | Smooth motion | -0.2/step |
| Action penalty | -0.005 | Smooth control | -0.2/step |
| **FALL PENALTY** | **-100.0** | **Don't crash!** | **-100 (episode end)** |

**Typical successful step**: +30 to +35 points
**Typical failed episode**: +500 points (crashes early)
**Typical successful episode**: +15,000 to +18,000 points (runs fast, doesn't fall)

---

**Ready to tune the points system? Let me know what you think!** 🎮🔧
