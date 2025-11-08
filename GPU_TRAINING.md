# GPU-Accelerated Training for PiDog

**Quick Start:** Use the unified container for everything!

---

## One-Time Setup

```bash
cd ~/pidog_ros2

# Build unified container (ROCm GPU + ROS2 + Gazebo)
./setup_unified_container.sh

# Time: ~10-15 minutes
```

---

## Daily Workflow

### Start Container

```bash
# Allow X11 for Gazebo
xhost +local:docker

# Start unified container
docker-compose --profile unified run --rm pidog-unified

# Inside container: Build workspace (first time)
colcon build
source install/setup.bash
```

### Complete Training Pipeline

**All commands run inside the unified container:**

#### 1. Collect Training Data

```bash
# Option A: Use launch file (enhanced with noise)
ros2 launch pidog_gaits collect_data_enhanced.launch.py

# Option B: Use script with custom duration
./collect_training_data.sh 90  # 90 seconds per gait
```

**Output:** `./training_data/gait_data_enhanced_*.npz`

---

#### 2. Train Imitation Model (GPU)

```bash
python3 -m pidog_gaits.train \
    --data ./training_data/gait_data_enhanced_*.npz \
    --model simple_lstm \
    --epochs 200 \
    --batch_size 1024 \
    --device cuda \
    --save_dir ./models

# Time: ~30 min with 20GB GPU
# Output: ./models/best_model.pth
```

**Key points:**
- ✅ Uses GPU for fast training (batch=1024)
- ✅ LSTM model for sim-to-real transfer
- ✅ ~20K parameters, validation loss < 0.04

---

#### 3. RL Fine-Tuning (GPU + 32 Parallel Envs)

```bash
python3 -m pidog_gaits.train_rl \
    --pretrained ./models/best_model.pth \
    --output ./models/rl \
    --timesteps 100000 \
    --envs 32

# Time: ~10-15 min with 32 parallel environments
# Output: ./models/rl/final_model.zip
```

**Key points:**
- ✅ GPU used for batched updates
- ✅ 32 parallel Gazebo simulations (CPU)
- ✅ PPO algorithm with physics-based rewards
- ✅ Default device=cuda (GPU always used)

**Note:** You may see a warning about "MLP+GPU" - this is expected and safe to ignore. GPU is still being used for tensor operations.

---

#### 4. Test Trained Model

```bash
# Launch Gazebo with neural network controller
ros2 launch pidog_gaits nn_demo.launch.py

# In another terminal, send commands:
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
```

---

## Monitor Training

### TensorBoard (for RL training)

```bash
# Inside container
tensorboard --logdir ./models/rl/tensorboard --bind_all

# On host, open browser:
# http://localhost:6006
```

**Metrics to watch:**
- `ep_rew_mean` - Average episode reward (should increase)
- `ep_len_mean` - Average episode length
- `policy_loss` - Policy network loss
- `value_loss` - Value function loss

### Training Plots (for Imitation)

```bash
# View training history plot
ls -lh ./models/training_history.png

# Copy to host to view:
# File is already on host at ~/pidog_ros2/models/training_history.png
```

---

## Performance with 20GB GPU

| Task | Config | Time | GPU/VRAM |
|------|--------|------|----------|
| Data collection | 90s per gait, 12 gaits | ~20 min | N/A |
| Imitation training | batch=1024, 200 epochs | ~30 min | ✅ ~4 GB |
| RL training | 100k steps, 32 envs | ~10-15 min | ✅ ~8 GB |
| **Total** | **Full pipeline** | **~60 min** | **GPU accelerated** |

---

## Troubleshooting

### GPU not detected

```bash
# Inside container, verify GPU:
python3 -c "import torch; print('GPU:', torch.cuda.is_available())"
rocm-smi

# Should show your AMD GPU
```

### Gazebo won't start

```bash
# On host:
xhost +local:docker

# Inside container:
export DISPLAY=$DISPLAY
ros2 launch pidog_description gazebo.launch.py
```

### Code changes not applying

```bash
# Full rebuild:
./rebuild.sh
source install/setup.bash

# Nuclear option:
./clean_all.sh
colcon build
source install/setup.bash
```

### RL training stuck at 100%

**This is normal!** With many parallel environments, PPO collects full rollouts:
- With 32 envs: minimum ~65k steps (2048 × 32)
- Progress bar shows >100% but keeps going
- Wait until the counter stops climbing
- Models save when training actually finishes

Reduce environments for quick tests:
```bash
# Quick test with 4 envs
python3 -m pidog_gaits.train_rl \
    --pretrained ./models/best_model.pth \
    --output ./models/test_rl \
    --timesteps 10000 \
    --envs 4
```

---

## Files Overview

**Main documentation:**
- `GPU_TRAINING.md` - This file (quick reference)
- `UNIFIED_CONTAINER.md` - Detailed unified container guide
- `CLAUDE.md` - Project instructions for AI assistants

**Setup:**
- `Dockerfile.rocm-ros2` - Unified container definition
- `docker-compose.yml` - Container orchestration
- `setup_unified_container.sh` - Build script

**Training scripts** (in `pidog_gaits/pidog_gaits/`):
- `train.py` - Imitation learning (GPU)
- `train_rl.py` - RL fine-tuning (GPU + parallel envs)
- `pidog_rl_env.py` - Gymnasium environment for RL
- `neural_network.py` - Model architectures

---

## Summary

**Everything in ONE container with your 20GB GPU:**

```bash
# Build once
./setup_unified_container.sh

# Daily use
xhost +local:docker
docker-compose --profile unified run --rm pidog-unified

# Inside: Full training pipeline
colcon build && source install/setup.bash
ros2 launch pidog_gaits collect_data_enhanced.launch.py
python3 -m pidog_gaits.train --data ./training_data/gait_data_*.npz --model simple_lstm --device cuda
python3 -m pidog_gaits.train_rl --pretrained ./models/best_model.pth --envs 32
```

**Total time: ~60 minutes for complete training!** 🚀
