# 🚀 PiDog GPU Training - Start Here

Everything you need for GPU-accelerated quadruped robot training.

---

## Quick Start

```bash
# One-time: Build unified container (~15 min)
./setup_unified_container.sh

# Daily: Start container
xhost +local:docker
docker-compose --profile unified run --rm pidog-unified

# Inside container: Train!
colcon build && source install/setup.bash
ros2 launch pidog_gaits collect_data_enhanced.launch.py
python3 -m pidog_gaits.train --data ./training_data/gait_data_*.npz --device cuda
python3 -m pidog_gaits.train_rl --pretrained ./models/best_model.pth --envs 32
```

**Total time: ~60 minutes** ⚡

---

## Documentation

📖 **Read these in order:**

1. **`README_GPU_TRAINING.md`** - Ultra-quick start (you are here!)
2. **`GPU_TRAINING.md`** - Complete workflow with troubleshooting
3. **`UNIFIED_CONTAINER.md`** - Detailed container guide
4. **`CLAUDE.md`** - Full project documentation

---

## Key Files

**Setup:**
- `setup_unified_container.sh` - Build container script
- `Dockerfile.rocm-ros2` - Container definition
- `docker-compose.yml` - Container config

**Training:**
- `pidog_gaits/pidog_gaits/train.py` - Imitation learning
- `pidog_gaits/pidog_gaits/train_rl.py` - RL fine-tuning
- `pidog_gaits/pidog_gaits/pidog_rl_env.py` - RL environment

**Helpers:**
- `rebuild.sh` - Rebuild ROS2 workspace
- `clean_all.sh` - Nuclear clean (when rebuild fails)
- `collect_training_data.sh` - Data collection script

---

## What You Get

✅ **20GB GPU fully utilized** - Batch 1024, 32 parallel envs
✅ **All-in-one container** - No complex multi-container setup
✅ **~60 min** total training (vs 6+ hours CPU)
✅ **LSTM models** - Sim-to-real ready with noise augmentation

---

## Troubleshooting

See `GPU_TRAINING.md` for full troubleshooting.

**Quick fixes:**
```bash
# GPU not detected
python3 -c "import torch; print('GPU:', torch.cuda.is_available())"

# Gazebo won't start
xhost +local:docker  # On host

# Code changes not applying
./rebuild.sh  # Inside container
```

---

**Happy training!** 🤖
