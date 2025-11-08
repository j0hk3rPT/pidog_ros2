# Quick Start: RL Training for Fast Running

**Goal**: Train PiDog to run as fast as possible using pure Reinforcement Learning

**Your Hardware**: 20GB GPU VRAM ✅ (Perfect for parallel training!)

---

## Prerequisites

You need:
- ✅ Unified Docker container (ROCm + ROS2 + Gazebo)
- ✅ GPU with 20GB VRAM (you have this!)
- ✅ Rebuilt workspace with sensor integration

---

## Step 1: Rebuild Workspace (5 minutes)

```bash
# Inside unified container
cd /home/user/pidog_ros2
./rebuild.sh
source install/setup.bash
```

**Expected**: Clean build, no errors

---

## Step 2: Install Python Dependencies (2 minutes)

```bash
# Install RL libraries
pip install stable-baselines3[extra] tensorboard

# Verify GPU is detected
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**Expected output**:
```
CUDA: True
Device: AMD Radeon ... (or NVIDIA ...)
```

---

## Step 3: Start Gazebo (Background)

**Terminal 1** (keep running):
```bash
cd /home/user/pidog_ros2
source install/setup.bash
ros2 launch pidog_description gazebo.launch.py
```

**Wait for**: "Gazebo started" message

**Verify sensors**:
```bash
# Terminal 2
ros2 topic list | grep imu
# Should show: /imu

ros2 topic echo /imu --once
# Should show IMU data
```

---

## Step 4: Start Training Phase 1 (20-30 minutes)

**Terminal 2**:
```bash
cd /home/user/pidog_ros2
source install/setup.bash

# Easy way (recommended)
./train_phase1.sh

# Or manual way (more control)
python3 train_fast_running.py \
    --phase 1 \
    --timesteps 50000 \
    --num-envs 16 \
    --batch-size 512 \
    --device cuda  # or 'auto' for auto-detection
```

**What this does**:
- Creates 16 parallel simulations (maxes out your GPU!)
- Trains for 50,000 steps
- Saves checkpoints every 10,000 steps
- Evaluates every 5,000 steps
- Target: Learn stable trot at 0.8-1.0 m/s

**While training**:
- Watch Gazebo: PiDog will try random movements, gradually learn to trot
- Progress bar shows current timestep
- Checkpoints saved to: `./rl_models/phase1_TIMESTAMP/`

---

## Step 5: Monitor Training (Optional)

**Terminal 3**:
```bash
cd /home/user/pidog_ros2

# Start TensorBoard
tensorboard --logdir ./rl_models

# Open browser: http://localhost:6006
```

**What to watch**:
- `rollout/ep_len_mean`: Episode length (should increase)
- `rollout/ep_rew_mean`: Reward per episode (should increase)
- `train/value_loss`: Value function loss (should decrease)
- Custom metrics:
  - `eval/mean_reward`: Evaluation performance
  - `eval/mean_ep_length`: How long episodes last

**Good signs**:
- Episode length increasing (robot survives longer)
- Reward increasing (robot going faster)
- Less falls over time

---

## Step 6: Check Results (2 minutes)

After Phase 1 completes:

```bash
# Best model location
ls ./rl_models/phase1_*/best_model/

# Should see:
# best_model.zip  ← This is your trained policy!
```

**Expected Phase 1 metrics**:
- Average episode length: 350-450 steps (was ~200)
- Average reward: 8,000-12,000 per episode
- Top speed: 0.8-1.0 m/s
- Fall rate: <15%
- Stall rate: <10%

---

## Step 7: Train Phase 2 (40-60 minutes)

**Terminal 2** (after Phase 1 done):
```bash
# Automatically resumes from Phase 1 best model
./train_phase2.sh

# Or manual
python3 train_fast_running.py \
    --phase 2 \
    --timesteps 100000 \
    --num-envs 16 \
    --resume ./rl_models/phase1_*/best_model/best_model.zip
```

**What changes in Phase 2**:
- Speed reward weight: 10.0 → 15.0 (higher priority)
- Milestone bonuses: +2, +5, +15 for 1.0, 1.5, 2.0 m/s
- Flight phase reward: +3.0 (encourages galloping)
- Same termination rules (stall, 45° roll)

**Target**: 1.5-2.0 m/s with galloping gait

---

## Step 8: Test Trained Model (5 minutes)

```bash
# Test Phase 2 best model
python3 -c "
from stable_baselines3 import PPO
from pidog_gaits.pidog_rl_env_hardware import PiDogHardwareEnv

# Load model
model = PPO.load('./rl_models/phase2_*/best_model/best_model.zip')

# Create environment
env = PiDogHardwareEnv(reward_mode='fast_running')

# Run 5 episodes
for episode in range(5):
    obs, info = env.reset()
    total_reward = 0
    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    print(f'Episode {episode+1}: Reward={total_reward:.1f}, Speed={info[\"speed\"]:.2f} m/s')

env.close()
"
```

**Expected output**:
```
Episode 1: Reward=15234.5, Speed=1.67 m/s
Episode 2: Reward=16891.2, Speed=1.82 m/s
...
```

---

## GPU Memory Usage

With your 20GB VRAM:

| Parallel Envs | VRAM Usage | Training Speed | Recommended |
|---------------|------------|----------------|-------------|
| 8 envs | ~3-4 GB | Slower | If low VRAM |
| **16 envs** | **~6-8 GB** | **Fast** | ✅ **Recommended** |
| 32 envs | ~12-15 GB | Faster | If you want max speed |
| 64 envs | ~18-20 GB | Fastest | Use all VRAM! |

**To change**:
```bash
# Use all 20GB (64 parallel environments!)
python3 train_fast_running.py --phase 1 --num-envs 64 --batch-size 1024
```

---

## Troubleshooting

### Problem: GPU not detected

```bash
# Check PyTorch GPU support
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with ROCm
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

### Problem: Training very slow

**Check**:
1. Is GPU being used? `nvidia-smi` or `rocm-smi`
2. Enough parallel envs? Try `--num-envs 32`
3. Is Gazebo GUI open? Close it for faster training

**Speed up**:
```bash
# Close Gazebo GUI, run headless
ros2 launch pidog_description gazebo.launch.py use_rviz:=false

# Or increase parallel envs
--num-envs 32  # Use more VRAM for speed
```

### Problem: Robot keeps falling

**Solutions**:
1. Lower speed weight: `--learning-rate 1e-4` (learn slower, more stable)
2. Increase episodes before eval: `--eval-freq 10000`
3. Check termination rules aren't too strict

### Problem: Out of memory

```bash
# Reduce parallel environments
--num-envs 8

# Reduce batch size
--batch-size 256

# Or reduce network size (edit train_fast_running.py)
net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Smaller network
```

---

## File Structure After Training

```
pidog_ros2/
├── rl_models/
│   ├── phase1_20251108_143022/
│   │   ├── best_model/
│   │   │   └── best_model.zip       ← Phase 1 best
│   │   ├── checkpoints/
│   │   │   ├── pidog_rl_10000_steps.zip
│   │   │   ├── pidog_rl_20000_steps.zip
│   │   │   └── ...
│   │   ├── tensorboard/             ← TensorBoard logs
│   │   └── final_model.zip          ← Phase 1 final
│   │
│   └── phase2_20251108_163045/
│       ├── best_model/
│       │   └── best_model.zip       ← Phase 2 best ⭐ USE THIS
│       ├── final_model.pth          ← PyTorch weights
│       └── ...
│
├── train_fast_running.py            ← Training script
├── train_phase1.sh                  ← Quick Phase 1
└── train_phase2.sh                  ← Quick Phase 2
```

---

## What to Expect During Training

### Phase 1 (Conservative - 50K steps, ~25 min)

**Timeline**:
- 0-10K steps: Random flailing, lots of falls
- 10K-20K steps: Starts standing, occasional forward movement
- 20K-30K steps: Consistent trotting, some falls
- 30K-50K steps: Stable trot, rarely falls, 0.8-1.0 m/s

**In Gazebo**:
- Watch PiDog go from chaotic to coordinated
- Diagonal leg pairs will synchronize (trot pattern)
- Less wobbling over time

### Phase 2 (Fast Running - 100K steps, ~50 min)

**Timeline**:
- 0-20K steps: Applies Phase 1 knowledge, tries faster
- 20K-50K steps: Discovers speed bonuses, pushes limits
- 50K-80K steps: Flight phase appears (0 contacts!), faster gait
- 80K-100K steps: Optimizes galloping, 1.5-2.0+ m/s

**In Gazebo**:
- Longer strides
- More airtime between steps
- Forward lean during sprint
- Possible galloping pattern

---

## Next Steps After Training

1. **Evaluate in Gazebo**: Watch your trained PiDog run!
2. **Tune if needed**: Adjust reward weights, retrain
3. **Deploy to hardware**: Transfer policy to real PiDog
4. **Iterate**: Collect real-world data, continue training

---

## Quick Command Reference

```bash
# Phase 1 (conservative baseline)
./train_phase1.sh

# Phase 2 (fast running)
./train_phase2.sh

# Monitor training
tensorboard --logdir ./rl_models

# Test model
python3 test_model.py --model ./rl_models/phase2_*/best_model/best_model.zip

# Resume interrupted training
python3 train_fast_running.py --phase 1 --resume ./rl_models/phase1_*/checkpoints/pidog_rl_30000_steps.zip
```

---

## Expected Timeline

| Task | Time | GPU Usage |
|------|------|-----------|
| Rebuild workspace | 5 min | 0% |
| Install dependencies | 2 min | 0% |
| Phase 1 training | 25 min | **80-90%** |
| Phase 2 training | 50 min | **80-90%** |
| **Total** | **~1.5 hours** | **Maxed out!** |

---

## Success Criteria

**Phase 1 Complete** ✅ when:
- Avg episode length > 350 steps
- Avg speed > 0.7 m/s
- Stall rate < 10%
- Fall rate < 15%

**Phase 2 Complete** ✅ when:
- Avg episode length > 300 steps
- **Top speed > 1.5 m/s** ⭐
- Flight phase detected
- Stable at high speeds

---

**Ready to train? Start with: `./train_phase1.sh`** 🚀
