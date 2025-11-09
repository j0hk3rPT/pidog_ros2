# PiDog Training System - Complete Index

## 📖 Start Here

**New user?** → `QUICK_START.md`
**Ready to train?** → `README_PRODUCTION.md`

## 🎯 Main Workflow

```bash
# 1. Train a complete model (one command!)
./train_production_pipeline.sh "my_experiment"

# 2. Test the model
./test_model_in_gazebo.sh experiments/my_experiment_*/final_model/final_model.zip

# 3. Deploy to robot
# Copy experiments/my_experiment_*/final_model/final_model.pth to hardware
```

## 📚 Documentation

### Getting Started
- **`QUICK_START.md`** - TL;DR guide for impatient users
- **`README_PRODUCTION.md`** - Complete production system overview ⭐
- **`PRODUCTION_PIPELINE.md`** - Detailed pipeline documentation

### Training Guides
- **`FAST_RL_TRAINING.md`** - Speed optimization guide
- **`PHYSICS_QUALITY_GUIDE.md`** - Quality vs speed tradeoffs
- **`CLAUDE.md`** - Project overview for AI assistant

### Technical Details
- **`GPU_TRAINING.md`** - GPU setup (if exists)
- **`HARDWARE_COMPARISON.md`** - Hardware specs
- **`SENSOR_SETUP_COMPLETE.md`** - Sensor configuration

## 🔧 Scripts

### Training
| Script | Purpose |
|--------|---------|
| `train_production_pipeline.sh` | **Full end-to-end training** ⭐ |
| `train_rl_vision_fast.sh` | Fast RL training (standalone) |
| `collect_training_data.sh` | Data collection only |

### Testing
| Script | Purpose |
|--------|---------|
| `test_model_in_gazebo.sh` | Test trained models with GUI |
| `test_rl_model.py` | Python evaluation script |
| `test_inside_container.sh` | Container diagnostics |

### Utilities
| Script | Purpose |
|--------|---------|
| `fix_numpy.sh` | Fix NumPy 2.x compatibility |
| `diagnose_performance.sh` | Check CPU/GPU usage |
| `rebuild.sh` | Rebuild ROS2 workspace |

## 🗂️ World Files (Physics)

| File | Quality | Speed | Use |
|------|---------|-------|-----|
| `pidog.sdf` | ★★★★★ | 1x | Final validation |
| `pidog_rl_medium.sdf` | ★★★★☆ | 5-10x | **Production** ⭐ |
| `pidog_rl_fast.sdf` | ★★★☆☆ | 10-30x | Experiments |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Segfault / NumPy error | `./fix_numpy.sh` |
| Low CPU/GPU | `./diagnose_performance.sh` |
| Container issues | `./test_inside_container.sh` |
| Build errors | `./rebuild.sh` |

## 📂 Important Directories

```
/workspace/
├── experiments/          # Training outputs (self-contained)
├── models/              # Legacy models
├── training_data/       # Legacy training data
├── pidog_description/   # Robot URDF, worlds, launch files
├── pidog_gaits/        # Gait generation, NN training
└── pidog_control/      # ROS2 controllers
```

## 🎓 Workflow Summary

### Quick Test (1-2 hours)
```bash
./train_production_pipeline.sh "quick_test"
```

### Production Training (1.5-2 hours)
```bash
./train_production_pipeline.sh "production_v1"
```

### Custom Training
```bash
# Edit train_production_pipeline.sh first
# Then:
./train_production_pipeline.sh "custom_experiment"
```

## 📊 Output Structure

Every training run creates:
```
experiments/<name>_<timestamp>/
├── SUMMARY.txt           ← Read this first
├── config.txt           ← What was run
├── data/                ← Training data
├── imitation_model/     ← Baseline
├── rl_model/           ← Fine-tuned
├── logs/               ← Debug info
└── final_model/        ← DEPLOY THIS ⭐
    ├── final_model.pth
    └── README.md
```

## 🚀 Common Commands

```bash
# Train
./train_production_pipeline.sh "exp_name"

# Test
./test_model_in_gazebo.sh path/to/model.zip 10

# Monitor
tensorboard --logdir experiments/exp_name_*/rl_model/tensorboard

# Debug
./diagnose_performance.sh

# Fix issues
./fix_numpy.sh
./rebuild.sh && source install/setup.bash
```

## 📞 Support

1. Check `experiments/*/logs/` for errors
2. Read `experiments/*/SUMMARY.txt`
3. Review relevant guide above
4. Check diagnostics: `./diagnose_performance.sh`

---

**Ready to start?** → `./train_production_pipeline.sh "my_first_model"`
