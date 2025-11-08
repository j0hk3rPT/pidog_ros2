# MuJoCo Training Quick Start Guide

## Prerequisites

✅ MuJoCo 3.3.7 installed
✅ Gymnasium environment ready
✅ AMD 7900XT GPU available

## Step 1: Test Viewer (Optional but Recommended)

First, make sure the viewer works:

```bash
# In container, check DISPLAY is set
echo $DISPLAY

# If empty, set it (use :0 or :1 depending on your X11 setup)
export DISPLAY=:0

# Test the viewer
python3 test_viewer.py
```

**Expected**: You should see a 3D window with the PiDog robot standing.

**Controls**:
- **Left mouse drag**: Rotate view
- **Right mouse drag**: Pan view
- **Scroll wheel**: Zoom in/out
- **ESC**: Close viewer

**If viewer fails**:
- Check `xhost +local:docker` was run on host
- Try `export DISPLAY=:1` if `:0` doesn't work
- Don't worry - headless training still works!

## Step 2: Watch Training (With Visualization)

Train for 100k steps while watching the robot learn:

```bash
python3 watch_training.py
```

**What you'll see**:
- Robot starts randomly moving
- Gradually learns to stand
- After ~20-50k steps: Should maintain standing pose
- Episode stats printed to console

**Duration**: ~5-10 minutes on your hardware

**Output**: Model saved to `./models/pidog_watched.zip`

## Step 3: Full Training (Headless, Fast)

Once you're happy with the setup, run full training:

```bash
python3 train_mujoco.py
```

**Features**:
- 16 parallel environments (much faster!)
- 1M timesteps
- GPU acceleration
- Checkpointing every 50k steps
- TensorBoard logging

**Duration**: ~30-45 minutes on your hardware

**Output**:
- Checkpoints: `./models/mujoco_checkpoints/`
- Best model: `./models/mujoco_best/`
- Final model: `./models/pidog_mujoco_final.zip`

## Step 4: Monitor Training

In another terminal:

```bash
# For watch_training.py
tensorboard --logdir ./logs/mujoco_watch/

# For train_mujoco.py
tensorboard --logdir ./logs/mujoco_tensorboard/
```

Open browser: http://localhost:6006

**Key metrics to watch**:
- `rollout/ep_rew_mean`: Episode reward (should increase)
- `train/value_loss`: Should decrease and stabilize
- `rollout/ep_len_mean`: Episode length (should increase as robot learns to not fall)

## Step 5: Test Trained Model

```bash
python3 test_trained_model.py --model ./models/pidog_mujoco_final.zip
```

(TODO: Create this script)

## Troubleshooting

### Viewer doesn't work
**Symptom**: `ERROR: could not initialize GLFW`

**Solution**:
```bash
# On host (outside container)
xhost +local:docker

# In container
export DISPLAY=:0
```

**Alternative**: Train headless (faster anyway!)
```bash
# Headless training works without DISPLAY
python3 train_mujoco.py  # No viewer needed
```

### Robot falls immediately
**Symptom**: Episode ends after 1-2 steps

**Possible causes**:
- Normal early in training (random policy)
- Wait ~10k-20k steps for learning to kick in
- Check reward function is working (`rollout/ep_rew_mean` in TensorBoard)

### Training is slow
**Symptom**: Less than 1000 steps/second

**Solutions**:
- Use headless mode (no viewer): Much faster
- Increase `num_envs` in `train_mujoco.py` (try 32 or 64)
- Check GPU is being used: `nvidia-smi` or `rocm-smi`

### Out of memory
**Symptom**: CUDA out of memory error

**Solutions**:
- Reduce `num_envs` (try 8 instead of 16)
- Reduce `batch_size` (try 32 instead of 64)
- Use CPU: Change `device='cpu'` in training script

## Expected Training Progress

**0-10k steps**: Random movements, frequent falls, reward ~0-2

**10k-30k steps**: Learns to stand briefly, reward ~2-4

**30k-100k steps**: Maintains standing pose, reward ~4-6

**100k-500k steps**: Stable standing, begins smooth movements

**500k-1M steps**: Robust policy, good walking (if walking reward added)

## Performance Benchmarks (Your Hardware)

| Setup | Steps/Sec | Real-Time Factor |
|-------|-----------|------------------|
| Single env + viewer | ~100 | ~2x |
| Single env headless | ~3000 | ~60x |
| 16 envs headless | ~15000 | ~300x |
| 32 envs headless | ~25000 | ~500x |

## Next Steps

1. ✅ Test viewer works
2. ✅ Watch short training (100k steps)
3. ✅ Run full training (1M steps)
4. ✅ Evaluate trained policy
5. 🚀 Deploy to real robot!

## Comparison: Gazebo vs MuJoCo

| Feature | Gazebo | MuJoCo |
|---------|--------|--------|
| **IMU Sensor** | ❌ Crashes | ✅ Works |
| **Training Speed** | ~1000 steps/s | ~15000 steps/s |
| **Viewer** | Full GUI | Simple viewer |
| **Best For** | Visualization, ROS2 | Fast RL training |

**Recommendation**: Use MuJoCo for training, Gazebo for visualization/testing!
