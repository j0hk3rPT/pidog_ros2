# 📊 Training Data Collection Guide

Quick guide to collect training data for your PiDog neural network.

---

## 🚀 Quick Start (2 Terminals)

### Terminal 1: Start Data Collection

```bash
cd /home/user/pidog_ros2
ros2 launch pidog_gaits collect_data.launch.py
```

**You should see:**
- Webots simulator window opens
- Robot appears in the arena
- Console shows: "Data Collector started"
- **But no data collected yet!** ⚠️

**Leave this running** - data will be saved when you press Ctrl+C at the end.

---

### Terminal 2: Trigger Gait Recording

```bash
cd /home/user/pidog_ros2

# Option 1: Automated (Recommended) - Records all 12 gaits
./collect_training_data.sh 20  # 20 seconds per gait = 4 minutes total

# Option 2: Manual - Send individual commands
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
# Wait 20 seconds, then try next gait...
```

---

## ⏱️ Collection Times

| Command | Duration | Data Points | Use Case |
|---------|----------|-------------|----------|
| `./collect_training_data.sh 20` | 4 min | ~7,200 | Quick test |
| `./collect_training_data.sh 40` | 8 min | ~14,400 | Good |
| `./collect_training_data.sh 60` | 12 min | ~21,600 | Best (recommended) |

**Recommendation:** Use **40-60 seconds** for quality data with your GPU training setup.

---

## ✅ How to Know It's Working

### In Terminal 1 (Data Collector):

You should see messages like:
```
[INFO] [data_collector]: Now recording: walk_forward
[INFO] [data_collector]: Collected 100 frames, 100 data points
[INFO] [data_collector]: Collected 200 frames, 200 data points
[INFO] [data_collector]: Collected 300 frames, 300 data points
...
```

### In Webots Window:

- Robot should be moving (walking, trotting, etc.)
- Movement changes when you send different gait commands
- Physics warnings are OK (see PHYSICS_WARNINGS_README.md)

### Data NOT Being Collected?

**Problem:** Terminal 1 shows no "Collected X frames" messages

**Solution:** The data collector is waiting for gait commands!
```bash
# In Terminal 2, send a command:
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once

# You should immediately see "Now recording: walk_forward" in Terminal 1
```

---

## 📝 Complete Collection Procedure

### Step 1: Launch Data Collection (Terminal 1)

```bash
cd /home/user/pidog_ros2
ros2 launch pidog_gaits collect_data.launch.py
```

Wait for:
- ✅ Webots window opens
- ✅ "Data Collector started" message
- ✅ "PiDog extern controller: connected"

### Step 2: Start Recording Gaits (Terminal 2)

```bash
cd /home/user/pidog_ros2

# Automated - records all 12 gaits (RECOMMENDED)
./collect_training_data.sh 40  # 40 seconds per gait = 8 minutes

# Script will show progress like:
# [1/12] Recording: walk_forward
# [##########                    ] 10/40s
```

**Watch Terminal 1** - you should see "Collected X frames" counting up!

### Step 3: Save Data (Terminal 1)

When the script finishes (or whenever you want to stop):

**In Terminal 1, press `Ctrl+C`**

You should see:
```
Shutting down, saving data...
Saved 14400 data points to ./training_data/gait_data_20251105_185030.json
Saved NumPy arrays to ./training_data/gait_data_20251105_185030.npz
  Input shape: (14400, 4)
  Output shape: (14400, 8)
```

### Step 4: Verify Data

```bash
ls -lh ./training_data/

# Should show:
# gait_data_YYYYMMDD_HHMMSS.json  (human readable)
# gait_data_YYYYMMDD_HHMMSS.npz   (for training)
```

---

## 🎯 Data Format

### Input Features (4 values):
- `gait_type`: 0 (walk), 1 (trot), 2 (pose)
- `direction`: -1 (backward), 0 (none), 1 (forward)
- `turn`: -1 (left), 0 (straight), 1 (right)
- `phase`: 0.0 to 1.0 (position in gait cycle)

### Output (8 joint angles in radians):
- Motor 0-1: Back right leg
- Motor 2-3: Back left leg
- Motor 4-5: Front right leg
- Motor 6-7: Front left leg

**Sample rate:** 30 Hz (30 frames per second)

---

## 🐛 Troubleshooting

### No "Collected X frames" messages

**Cause:** Data collector hasn't received any gait commands

**Fix:**
```bash
# Send a test command
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once

# Should immediately see in Terminal 1:
# [INFO] [data_collector]: Now recording: walk_forward
```

### Robot not moving

**Cause:** Gait generator might not be running

**Check:**
```bash
ros2 node list
# Should include: /gait_generator

ros2 topic list
# Should include: /motor_pos and /gait_command
```

### Data file not created after Ctrl+C

**Cause:** No data was collected (collector never received valid gait commands)

**Check Terminal 1 logs:**
```
[WARN] [data_collector]: No data collected!
```

**Fix:** Make sure you sent gait commands before stopping!

### Physics warnings filling the console

**This is normal!** See `PHYSICS_WARNINGS_README.md`

The warnings don't affect data quality. Just make sure you can see the "Collected X frames" messages among the warnings.

---

## 📈 Recommended Workflow

### For Initial Testing (4 minutes):
```bash
./collect_training_data.sh 20
```

### For Production Training (8-12 minutes):
```bash
./collect_training_data.sh 60
```

### Manual Fine Control:
```bash
# Record specific gaits for longer
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
# Wait...
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
# Wait...
# Ctrl+C when done
```

---

## ⏭️ Next Steps After Collection

```bash
# 1. Verify data
ls -lh ./training_data/
# Should see .npz file with data

# 2. Train on GPU
./train_gpu.sh

# 3. Test trained model
ros2 launch pidog_gaits nn_demo.launch.py
```

---

## 📊 Expected Results

### Good Collection:
- ✅ 10,000+ data points
- ✅ All 12 gaits recorded
- ✅ .npz file ~500KB - 2MB
- ✅ Input shape: (N, 4)
- ✅ Output shape: (N, 8)

### Insufficient Data:
- ❌ < 1,000 data points
- ❌ Only 1-2 gaits
- ❌ File very small (<100KB)

**If insufficient:** Run collection again with longer duration per gait.

---

## 💡 Pro Tips

1. **Use the automated script** - It's faster and ensures all gaits are recorded
2. **Monitor Terminal 1** - "Collected X frames" should increment steadily
3. **Don't worry about physics warnings** - They're cosmetic (see PHYSICS_WARNINGS_README.md)
4. **Save early, save often** - You can Ctrl+C anytime to save partial data
5. **More data = better model** - 60 seconds per gait is better than 20

---

## Summary

```bash
# Terminal 1
ros2 launch pidog_gaits collect_data.launch.py

# Terminal 2
./collect_training_data.sh 40

# Terminal 1 (after script finishes)
# Press Ctrl+C to save

# Verify
ls -lh ./training_data/

# Train
./train_gpu.sh
```

**Total time:** 8-12 minutes to collect, 10-15 minutes to train on GPU

You'll have a trained model in under 30 minutes! 🚀
