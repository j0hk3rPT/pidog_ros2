# PiDog GPU-Accelerated Training

Quick start guide for training PiDog gaits with GPU acceleration.

---

## 🚀 Quick Start

### One-Time Setup (~15 min)

```bash
cd ~/pidog_ros2
./setup_unified_container.sh
```

### Daily Workflow

```bash
# Start container
xhost +local:docker
docker-compose --profile unified run --rm pidog-unified

# Inside container - complete training pipeline:
colcon build && source install/setup.bash

# 1. Collect data (~20 min)
ros2 launch pidog_gaits collect_data_enhanced.launch.py

# 2. Train imitation (GPU, ~30 min)
python3 -m pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model simple_lstm \
    --device cuda

# 3. RL fine-tuning (GPU + 32 parallel envs, ~15 min)
python3 -m pidog_gaits.train_rl \
    --pretrained ./models/best_model.pth \
    --envs 32

# Total time: ~65 minutes 🚀
```

---

## 📚 Documentation

- **`GPU_TRAINING.md`** - Complete workflow with troubleshooting
- **`UNIFIED_CONTAINER.md`** - Detailed container guide
- **`CLAUDE.md`** - Full project documentation

---

## 💡 Key Features

✅ **20GB GPU fully utilized** - Batch size 1024, 32 parallel environments
✅ **All-in-one container** - ROS2 + Gazebo + ROCm GPU
✅ **~1 hour** total training time (vs 6+ hours CPU-only)
✅ **Sim-to-real ready** - LSTM models, noise augmentation

---

## 🆘 Troubleshooting

**GPU not detected:**
```bash
python3 -c "import torch; print('GPU:', torch.cuda.is_available())"
rocm-smi
```

**Gazebo won't start:**
```bash
xhost +local:docker  # On host
```

**Code changes not applying:**
```bash
./rebuild.sh  # Inside container
source install/setup.bash
```

**More help:** See `GPU_TRAINING.md`
