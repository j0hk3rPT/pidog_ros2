# PiDog Production Training Pipeline

Complete end-to-end training system for training PiDog to run fast using vision-based reinforcement learning.

## Quick Start

```bash
# Inside Docker container
./train_production_pipeline.sh "pidog_fast_runner_v1"
```

That's it! The pipeline will:
1. ✅ Collect training data (10 min)
2. ✅ Train imitation model (20-40 min)
3. ✅ Fine-tune with RL (40-60 min)
4. ✅ Package deployable model

**Total time**: ~1.5-2 hours

## What Gets Created

```
experiments/
└── pidog_fast_runner_v1_20251108_223045/
    ├── config.txt              # Training configuration
    ├── data/                   # Raw training data
    │   └── gait_data_enhanced_*.npz
    ├── imitation_model/        # Imitation learning
    │   ├── best_model.pth
    │   ├── final_model.pth
    │   └── training_history.png
    ├── rl_model/               # RL fine-tuning
    │   ├── final_model.pth     # PyTorch model
    │   ├── final_model.zip     # SB3 model
    │   ├── checkpoints/
    │   └── tensorboard/
    ├── logs/                   # All training logs
    │   ├── data_collection.log
    │   ├── imitation_training.log
    │   └── rl_training.log
    ├── final_model/            # DEPLOYABLE MODEL ⭐
    │   ├── final_model.pth     # ← Deploy this to robot
    │   ├── final_model.zip     # ← Use for testing
    │   └── README.md
    └── SUMMARY.txt             # Training summary
```

## Pipeline Stages

### Stage 1: Data Collection (10 minutes)

Collects expert demonstrations using traditional gaits:
- **Gaits**: walk_forward, walk_backward, trot_forward, stand, sit
- **Duration**: 2 minutes per gait
- **Noise**: Position σ=0.01 rad, velocity σ=0.1 rad/s (for robustness)
- **Output**: ~10,000-15,000 samples

### Stage 2: Imitation Learning (20-40 minutes)

Trains neural network to mimic expert demonstrations:
- **Model**: GaitNetSimpleLSTM (best for sim-to-real)
- **Architecture**: Input(4) → LSTM(64) → Dense(32) → Dense(12)
- **Epochs**: 300 (typically converges ~100-200)
- **Batch size**: 1024
- **Output**: Baseline policy that can reproduce gaits

### Stage 3: RL Training (40-60 minutes)

Fine-tunes policy with physics-based rewards:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Sensors**: Camera + IMU + Ultrasonic + 12 Joint Encoders
- **Parallel envs**: 4 (adjustable based on CPU cores)
- **Timesteps**: 200,000
- **Rewards**:
  - Stability (upright, no falling)
  - Speed (forward velocity)
  - Obstacle avoidance (ultrasonic)
  - Efficiency (smooth movements)

### Stage 4: Finalization (1 minute)

Packages model for deployment:
- Copies final model to `final_model/`
- Generates deployment README
- Creates summary with performance metrics

## Configuration

Edit `train_production_pipeline.sh` to customize:

```bash
# Training parameters (lines 30-37)
DATA_COLLECTION_DURATION=120  # seconds per gait
IMITATION_EPOCHS=300           # epochs
IMITATION_BATCH_SIZE=1024      # batch size
RL_TIMESTEPS=200000            # total RL steps
RL_PARALLEL_ENVS=4             # parallel environments
DEVICE="cuda"                  # cuda or cpu

# Physics quality (line 40)
PHYSICS_QUALITY="medium"       # production, medium, or fast
```

### Physics Quality Options

| Quality | Physics Steps | Solver Iters | Speed | Best For |
|---------|---------------|--------------|-------|----------|
| **production** | 1ms @ 300 iters | 1000 Hz | 1x RT | Final validation |
| **medium** ⭐ | 2ms @ 150 iters | 500 Hz | 5-10x | Main training |
| **fast** | 5ms @ 50 iters | 200 Hz | 10-30x | Quick experiments |

**Recommended**: `medium` - Best balance of quality and speed.

## Testing the Trained Model

After training completes:

```bash
# Option 1: Quick test with GUI
./test_model_in_gazebo.sh experiments/pidog_fast_runner_v1_*/final_model/final_model.zip 10

# Option 2: Headless test (faster)
./test_model_in_gazebo.sh experiments/pidog_fast_runner_v1_*/final_model/final_model.zip 10 no

# Option 3: Manual testing
python3 test_rl_model.py \
    --model experiments/pidog_fast_runner_v1_*/final_model/final_model.zip \
    --episodes 10
```

## Monitoring Training

### Real-time Monitoring

```bash
# In another terminal
tensorboard --logdir experiments/pidog_fast_runner_v1_*/rl_model/tensorboard
# Open http://localhost:6006
```

### View Logs

```bash
# Data collection
tail -f experiments/pidog_fast_runner_v1_*/logs/data_collection.log

# Imitation training
tail -f experiments/pidog_fast_runner_v1_*/logs/imitation_training.log

# RL training
tail -f experiments/pidog_fast_runner_v1_*/logs/rl_training.log
```

## Resuming/Modifying

If a stage fails, you can resume:

```bash
# Stage 2 only (already have data)
TRAINING_DATA="experiments/pidog_fast_runner_v1_*/data/gait_data_enhanced_*.npz"
python3 -m pidog_gaits.train \
    --data $TRAINING_DATA \
    --model simple_lstm \
    --epochs 300 \
    --batch_size 1024 \
    --device cuda \
    --save_dir experiments/pidog_fast_runner_v1_*/imitation_model

# Stage 3 only (already have imitation model)
IMITATION_MODEL="experiments/pidog_fast_runner_v1_*/imitation_model/best_model.pth"
python3 -m pidog_gaits.train_rl_vision \
    --pretrained $IMITATION_MODEL \
    --output experiments/pidog_fast_runner_v1_*/rl_model \
    --timesteps 200000 \
    --envs 4 \
    --device cuda
```

## Expected Performance

### Imitation Learning
- **Training loss**: Should drop to <0.01 by epoch 100-200
- **Validation loss**: Should be similar to training loss (no overfitting)
- **Result**: Policy can reproduce basic gaits

### RL Training
- **Episode reward**: Should increase from ~100 to ~1000+ over 200k steps
- **Forward velocity**: Should improve from 0.05 m/s → 0.15+ m/s
- **Stability**: Episodes should go from ~50 steps → 500 steps (full length)
- **Result**: Policy is faster, more stable, and avoids obstacles

## Hardware Deployment

1. **Copy model to robot**:
   ```bash
   scp experiments/pidog_fast_runner_v1_*/final_model/final_model.pth \
       robot@pidog:/home/robot/models/
   ```

2. **Use with ROS2 on robot**:
   ```bash
   # On robot
   ros2 run pidog_gaits nn_controller \
       --model /home/robot/models/final_model.pth
   ```

## Troubleshooting

### Pipeline Fails at Stage 1 (Data Collection)
- **Check**: Gazebo started? `ros2 topic hz /clock`
- **Check**: Controllers loaded? `ros2 control list_controllers`
- **Fix**: Rebuild workspace, source setup.bash

### Imitation Model Not Learning (loss stays high)
- **Check**: Training data quality - view samples
- **Fix**: Increase epochs to 500, try `GaitNetLarge` model
- **Fix**: Reduce batch size to 512

### RL Training Very Slow
- **Check**: CPU usage - should be 60-90% with 4 envs
- **Fix**: Increase parallel envs (8 for 16-core CPU)
- **Fix**: Use `fast` physics quality

### Low Episode Rewards in RL
- **Check**: Imitation model baseline - test it first
- **Fix**: Increase RL timesteps to 500k
- **Fix**: Adjust reward weights in `pidog_rl_env_vision.py`

### Robot Falls Immediately When Testing
- **Issue**: RL overfit to training conditions
- **Fix**: Train longer with more domain randomization
- **Fix**: Use `production` physics quality for final training

## Advanced Usage

### Multiple Experiments

Run multiple experiments with different configurations:

```bash
# Fast gait focused
./train_production_pipeline.sh "pidog_speed_demon"

# Stable gait focused
# (Edit reward weights to prioritize stability over speed)
./train_production_pipeline.sh "pidog_stable_walker"

# Obstacle avoidance focused
# (Edit reward weights to prioritize ultrasonic)
./train_production_pipeline.sh "pidog_obstacle_master"
```

### Compare Experiments

```bash
# View all experiments
ls -la experiments/

# Compare training curves
tensorboard --logdir experiments/ --port 6006
```

## Performance Benchmarks

On a typical system (16-core CPU, RTX 3080):

- **Data Collection**: ~10 minutes
- **Imitation Learning**: ~25 minutes (300 epochs)
- **RL Training**: ~45 minutes (200k steps, 4 envs)
- **Total**: ~1.5 hours

On slower hardware:
- **Without GPU**: 2-3x slower (~3-4 hours total)
- **Fewer cores**: Reduce parallel envs, 2-3x slower

## Tips for Best Results

1. **Use medium physics quality** - Best balance for production
2. **Start with 4 parallel envs** - Good for most systems
3. **Monitor training** - Use TensorBoard to catch issues early
4. **Test intermediate models** - Test imitation model before RL
5. **Keep raw data** - Useful for retraining with different architectures
6. **Version your experiments** - Use descriptive names with dates

## Files Reference

- `train_production_pipeline.sh` - Main pipeline script ⭐
- `test_model_in_gazebo.sh` - Test trained models
- `test_rl_model.py` - Model evaluation script
- `PHYSICS_QUALITY_GUIDE.md` - Physics settings explained

## Next Steps

After successful training:

1. ✅ Test model in simulation
2. ✅ Validate performance metrics
3. ✅ Deploy to hardware
4. ✅ Fine-tune on real robot (optional)
5. ✅ Share results / iterate!

---

**Questions?** Check the logs in `experiments/<name>/logs/` for detailed information.
