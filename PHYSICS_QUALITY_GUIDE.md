# Physics Quality vs Speed Tradeoffs

## Quick Comparison Table

| Setting | Step Size | Solver Iters | Update Rate | Speed | Quality | Best For |
|---------|-----------|--------------|-------------|-------|---------|----------|
| **Production** (original) | 1ms | 300 | 1000 Hz | 1x RT | ★★★★★ | Final validation, sim-to-real |
| **Medium** (recommended) | 2ms | 150 | 500 Hz | ~5-10x | ★★★★☆ | **Main training** |
| **Fast** (exploration) | 5ms | 50 | 200 Hz | ~10-30x | ★★★☆☆ | Quick experiments, debugging |

**RT = Real-time**

## What You Lose with Faster Settings

### Fast (5ms @ 50 iters)
❌ **Contact jitter** - Feet might slip/bounce more
❌ **Joint drift** - Constraints less accurate
❌ **Trajectory smoothness** - Less temporal resolution
✅ **10-30x faster training**
✅ **Still adequate for quadruped gaits** (200 Hz > typical hardware)

### Medium (2ms @ 150 iters) ⭐ **RECOMMENDED**
✅ **Good contact stability** - Minimal slipping
✅ **Accurate constraints** - Joints behave realistically
✅ **High temporal resolution** - Smooth trajectories
✅ **5-10x faster than real-time** - Still fast!
⚠️ **Slightly slower than "fast" mode**

### Production (1ms @ 300 iters)
✅ **Maximum accuracy** - Best sim-to-real transfer
✅ **Perfect for final validation**
❌ **Locked to real-time** - Very slow training
❌ **Overkill for exploration**

## Recommended Workflow

### Phase 1: Exploration (Fast)
```bash
# Use FAST settings for rapid prototyping
# Try different reward functions, hyperparameters
./train_rl_vision_fast.sh 10000 1
# ~5-10 minutes per 10k steps
```

### Phase 2: Main Training (Medium) ⭐
```bash
# Use MEDIUM settings for production training
# Good quality + still reasonably fast
# Switch world file to pidog_rl_medium.sdf
./train_rl_vision_medium.sh 100000 4
# ~30-40 minutes per 100k steps
```

### Phase 3: Validation (Production)
```bash
# Use ORIGINAL settings to validate final policy
# Run trained policy in high-fidelity simulation
# Use pidog.sdf (original world)
python3 -m pidog_gaits.test_rl_vision \
    --model ./models/rl_final/final_model.zip \
    --episodes 100
```

## Impact on Sim-to-Real Transfer

**Fast settings** (50 iters, 5ms):
- ⚠️ Policy might learn to exploit physics inaccuracies
- ⚠️ Could overfit to jittery contacts
- ✅ Still useful for prototyping reward functions

**Medium settings** (150 iters, 2ms): ⭐
- ✅ Good balance - realistic enough for transfer
- ✅ Implicit domain randomization from some noise
- ✅ Faster iteration than production mode

**Production settings** (300 iters, 1ms):
- ✅ Highest fidelity for sim-to-real
- ✅ Best for final validation before deployment
- ❌ Too slow for main training loop

## Bottom Line

**For your use case (vision-based RL with camera/IMU/sonar):**

1. **Start with FAST** to debug your pipeline ← *You are here*
2. **Switch to MEDIUM** for main training ← *Do this next*
3. **Validate with PRODUCTION** before hardware deployment

The quality loss from FAST→MEDIUM is **minimal** for gait learning.
The quality loss from MEDIUM→PRODUCTION is **negligible** for most use cases.

**Speed gain is worth it** - you can run 10x more experiments!
