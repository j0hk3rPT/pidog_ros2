# PiDog Training - Quick Start Guide

## For the Impatient 🚀

**Train a complete model in one command:**

```bash
# Inside Docker container
./train_production_pipeline.sh "my_fast_pidog"
```

Wait ~1.5-2 hours, then test:

```bash
./test_model_in_gazebo.sh experiments/my_fast_pidog_*/final_model/final_model.zip
```

**Done!** Your model is in `experiments/my_fast_pidog_*/final_model/final_model.pth`

---

## What Just Happened?

The pipeline automatically:

1. ✅ **Collected training data** - Expert demonstrations from traditional gaits
2. ✅ **Trained imitation model** - Neural network learns to copy experts
3. ✅ **Fine-tuned with RL** - Physics-based rewards make it better
4. ✅ **Packaged for deployment** - Ready to use on real hardware

---

## File Organization

```
experiments/my_fast_pidog_TIMESTAMP/
├── final_model/
│   ├── final_model.pth  ← Deploy this to robot
│   └── final_model.zip  ← Use this for testing
├── SUMMARY.txt          ← Read this first
└── logs/                ← Check if something went wrong
```

---

## Common Commands

### Training

```bash
# Full production pipeline
./train_production_pipeline.sh "experiment_name"

# Quick test (fast physics, fewer steps)
# Edit train_production_pipeline.sh:
#   PHYSICS_QUALITY="fast"
#   RL_TIMESTEPS=50000
#   RL_PARALLEL_ENVS=8
```

### Testing

```bash
# Test with visualization (Gazebo GUI)
./test_model_in_gazebo.sh path/to/model.zip 10

# Test headless (faster)
./test_model_in_gazebo.sh path/to/model.zip 10 no
```

### Monitoring

```bash
# View training progress
tensorboard --logdir experiments/my_fast_pidog_*/rl_model/tensorboard

# View logs
tail -f experiments/my_fast_pidog_*/logs/rl_training.log
```

---

## Troubleshooting One-Liners

```bash
# NumPy version issue?
./fix_numpy.sh

# Check if Gazebo is running fast
./test_inside_container.sh

# Performance diagnostics
./diagnose_performance.sh

# Rebuild everything
./rebuild.sh && source install/setup.bash
```

---

## Next Steps

1. **Read**: `PRODUCTION_PIPELINE.md` - Full details
2. **Understand**: `PHYSICS_QUALITY_GUIDE.md` - Quality vs speed
3. **Optimize**: `FAST_RL_TRAINING.md` - Max performance

---

## Support

- **Logs**: Check `experiments/*/logs/` for detailed errors
- **Config**: See `experiments/*/config.txt` for what was run
- **Summary**: See `experiments/*/SUMMARY.txt` for quick overview
