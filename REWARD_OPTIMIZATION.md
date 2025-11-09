# Reward Function Optimization for Speed + Safety

## 🎯 Your Goals
1. **Balanced speed+stability** (leaning towards speed)
2. **CRITICAL obstacle avoidance** (hardware safety)
3. **Sim-to-real robustness** (will deploy to hardware)

## 📊 Changes Made

### **1. STABILITY (Hardware Safety)**

| Component | Original | Optimized | Reason |
|-----------|----------|-----------|--------|
| Height tolerance | 8cm min | 7cm min | Allow faster, lower stance |
| Height penalty | error × 8.0 | error × 6.0 | More tolerant for speed |
| Tilt tolerance | ±11° | ±17° | Allow leaning in turns |
| Tilt penalty | ×2.0 | ×1.5 | Less penalty for dynamics |
| Fall penalty | -15.0 | **-20.0** | **INCREASED for hardware** |
| Head contact | -8.0 | **-10.0** | **CRITICAL for safety** |

**Why**: Slightly relaxed stability for speed, but HARSHER penalties for catastrophic failures (falls, head contact) to protect hardware.

---

### **2. OBSTACLE AVOIDANCE (CRITICAL!)** 🚨

| Distance Range | Original | Optimized | Change |
|---------------|----------|-----------|--------|
| Critical (<15cm) | N/A | **-15.0 + stop moving** | **NEW: Emergency zone** |
| Danger (<30cm) | -5.0 | **-8.0 + slow down** | **More aggressive** |
| Warning (<60cm) | <50cm: -2.0 | **<60cm: -4.0** | **Larger safe zone** |
| Safe (>60cm) | +0.3 | **+0.5** | Better reward |

**Approaching penalty**:
- Original: -vel × 3.0 in danger zone
- Optimized: **-vel × 10.0** in critical, **-vel × 5.0** in danger

**Why**: Hardware safety is CRITICAL. Robot must stop/slow when obstacles detected.

---

### **3. SPEED REWARDS (Primary Objective)** 🏃

| Component | Original | Optimized | Reason |
|-----------|----------|-----------|--------|
| Base speed multiplier | ×8.0 | **×10.0** | **More speed focus** |
| Speed milestone >10cm/s | None | **+1.0** | Progressive learning |
| Speed milestone >15cm/s | +2.0 | **+2.0** | Same |
| Speed milestone >20cm/s | None | **+5.0** | **Ambitious goal** |
| Speed milestone >25cm/s | None | **+10.0** | **Stretch goal** |
| Backward penalty | -3.0 | **-5.0** | **Stronger aversion** |
| Lateral drift penalty | -1.5 | **-2.0** | **Stay straighter** |
| Episode avg speed bonus | None | **+avg_speed × 20** | **Consistency reward** |
| Personal best bonus | None | **+10.0** | **Encourage improvement** |

**Why**: Stronger speed incentives with progressive milestones. Curriculum learning approach.

---

### **4. SIM-TO-REAL TRANSFER** 🤖

| Component | Original | Optimized | Reason |
|-----------|----------|-----------|--------|
| Jerk penalty | ×0.015 | **×0.020** | **Smoother for hardware** |
| Energy penalty | ×0.02 | **×0.03** | **Battery efficiency** |
| Bounce penalty | ×2.0 | **×3.0** | **Less bouncing** |
| Extreme joints penalty | None | **-0.5 per joint >80°** | **Hardware limits** |
| Drift penalty | ×1.0 | **×1.5** | **Better tracking** |

**Why**: Hardware has physical limits (torque, battery, range of motion). Penalize behaviors that work in sim but fail on hardware.

---

## 📈 Expected Behavior Changes

### **Training Progression**

**Phase 1 (0-50k steps)**: Learn stable trotting
- Target: 0.10-0.15 m/s
- Focus: Don't fall, stay upright
- Obstacle: Start learning avoidance

**Phase 2 (50k-100k steps)**: Increase speed
- Target: 0.15-0.20 m/s
- Focus: Speed milestones, smoother gait
- Obstacle: Consistent avoidance

**Phase 3 (100k-200k steps)**: Optimize performance
- Target: 0.20-0.25 m/s (stretch goal)
- Focus: Best speed, efficiency
- Obstacle: Perfect avoidance

### **Speed Targets**

| Goal | Speed (m/s) | Speed (cm/s) | Real-world |
|------|-------------|--------------|------------|
| Minimum | 0.10 | 10 | Slow walk |
| Good | 0.15 | 15 | Comfortable trot ✅ |
| Great | 0.20 | 20 | Fast trot 🎯 |
| Excellent | 0.25 | 25 | Sprint! 🚀 |

**Realistic target for 200k steps**: 0.15-0.20 m/s

---

## 🔬 Key Optimizations

### **1. Progressive Speed Milestones**

Original: Only rewarded speed linearly
```python
reward += forward_vel * 8.0
```

Optimized: Milestone bonuses for breakthroughs
```python
reward += forward_vel * 10.0  # Base reward
if forward_vel > 0.10: reward += 1.0   # First milestone
if forward_vel > 0.15: reward += 2.0   # Good speed
if forward_vel > 0.20: reward += 5.0   # Great speed!
if forward_vel > 0.25: reward += 10.0  # Excellent!
```

**Why**: Encourages breaking through plateaus.

---

### **2. Critical Obstacle Zones**

Original: Only 2 zones (danger <20cm, safe >50cm)

Optimized: 4 zones with graduated response
```
|--------|---------|---------|---------|
0cm     15cm      30cm      60cm     infinity
CRITICAL DANGER  WARNING    SAFE
-15.0    -8.0     -4.0      +0.5
STOP!    SLOW    CAREFUL    GO!
```

**Why**: Hardware needs progressive warnings, not binary safe/unsafe.

---

### **3. Episode Performance Tracking**

Original: Only step-by-step rewards

Optimized: Track entire episode + personal bests
```python
avg_speed = np.mean(episode_speeds)
reward += avg_speed * 20.0  # Consistency bonus

if avg_speed > best_speed:
    reward += 10.0  # Beat personal record!
```

**Why**: Encourages consistent fast performance, not just brief bursts.

---

### **4. Hardware-Aware Penalties**

New penalties for sim-to-real transfer:
- Extreme joint angles (>80°) → Joint limits
- High jerk → Mechanical stress
- Energy waste → Battery life
- Bouncing → Sensor noise

**Why**: Behaviors that work in sim often fail on hardware. Penalize these early.

---

## 🚀 How to Use

### **Option 1: Update Production Pipeline**

Edit `train_production_pipeline.sh` line ~18:
```bash
# Change from:
from .pidog_rl_env_vision import PiDogVisionEnv

# To:
from .pidog_rl_env_vision_optimized import PiDogVisionEnvOptimized as PiDogVisionEnv
```

### **Option 2: Test Optimized Rewards First**

```bash
# Quick test with optimized rewards (10k steps)
python3 -m pidog_gaits.train_rl_vision_optimized \
    --pretrained ./models/best_model.pth \
    --output ./models/rl_optimized_test \
    --timesteps 10000 \
    --envs 1
```

### **Option 3: Compare Both**

Train with both reward functions and compare:
```bash
# Original rewards
./train_production_pipeline.sh "original_rewards"

# Optimized rewards (after editing)
./train_production_pipeline.sh "optimized_rewards"

# Compare in TensorBoard
tensorboard --logdir experiments/
```

---

## 📊 Expected Results

### **Original Rewards**
- Pros: Very stable, safe
- Cons: Might be slow (~0.10-0.15 m/s)
- Best for: Cautious deployment

### **Optimized Rewards**
- Pros: Faster (~0.15-0.20 m/s), better obstacles
- Cons: Might need more training steps
- Best for: Speed-focused + hardware deployment

---

## ⚠️ Important Notes

1. **Obstacle avoidance is MUCH more aggressive** - This is intentional for hardware safety
2. **Speed milestones encourage exploration** - Robot will try to go faster
3. **Sim-to-real penalties added** - Smoother, more hardware-friendly behaviors
4. **May need more training steps** - 200k instead of 100k for best results

---

## 🎛️ Fine-Tuning

If after training you find:

**Too slow?**
- Increase speed multiplier: 10.0 → 12.0
- Reduce stability penalties

**Too unstable?**
- Increase fall penalty: -20.0 → -25.0
- Tighten tilt tolerance: 0.3 → 0.25

**Not avoiding obstacles?**
- Increase critical penalty: -15.0 → -20.0
- Expand safe zone: 0.60 → 0.75

**Too jerky for hardware?**
- Increase jerk penalty: 0.020 → 0.025
- Add more smoothness rewards

---

## ✅ Ready to Train

```bash
./train_production_pipeline.sh "speed_optimized_v1"
```

Monitor these metrics in TensorBoard:
- `rollout/ep_rew_mean` - Should increase to 1000+
- `info/forward_vel` - Should reach 0.15-0.20 m/s
- `info/speed` - Average speed per episode
- `info/ultrasonic_range` - Should stay >0.30m

Good luck! 🚀
