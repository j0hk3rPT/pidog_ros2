# 🐕🤖 PiDog Neural Network Training - Quick Start

This guide helps you get started with training neural networks to control your PiDog robot!

## 🎯 What You'll Learn

You'll train a neural network to generate PiDog's walking movements. Instead of hand-coding gaits, the neural network will learn from expert demonstrations!

**Pipeline:**
```
Traditional Gaits → Data Collection → Neural Network Training → Deployment
```

## ⚡ Quick Start (5 Steps)

### 1️⃣ Install Dependencies

```bash
# Install PyTorch
pip3 install torch torchvision matplotlib numpy

# Optional: For GPU acceleration (if you have NVIDIA GPU)
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2️⃣ Build the Package

```bash
cd /home/user/pidog_ros2

# Source ROS2 (adjust for your ROS2 distribution)
source /opt/ros/jazzy/setup.bash  # or humble, etc.

# Build
colcon build --packages-select pidog_gaits
source install/setup.bash
```

### 3️⃣ Demo Traditional Gaits (5 minutes)

See the hand-coded gaits in action:

```bash
# Terminal 1: Launch simulator with gaits
ros2 launch pidog_gaits gait_demo.launch.py
```

```bash
# Terminal 2: Try different movements
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_left'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'sit'" --once
```

**Watch your PiDog walk!** This is what the neural network will learn to replicate.

### 4️⃣ Collect Training Data (10-15 minutes)

Record the gaits for training:

```bash
# Terminal 1: Start data collection
ros2 launch pidog_gaits collect_data.launch.py
```

```bash
# Terminal 2: Cycle through gaits (run this script)
#!/bin/bash
for gait in walk_forward walk_backward walk_left walk_right \
            trot_forward trot_backward trot_left trot_right \
            sit stand lie stretch; do
    echo "Recording: $gait"
    ros2 topic pub /gait_command std_msgs/msg/String "data: '$gait'" --once
    sleep 20  # Record each gait for 20 seconds
done
echo "Data collection complete!"
```

Press **Ctrl+C** when done. Data is saved to `./training_data/`

**Expected output:**
- Should have ~10,000+ data points
- File: `training_data/gait_data_YYYYMMDD_HHMMSS.npz`

### 5️⃣ Train Neural Network (15-30 minutes)

Train the network to mimic the gaits:

```bash
cd /home/user/pidog_ros2

python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model simple \
    --epochs 100 \
    --batch_size 64 \
    --save_dir ./models
```

**What to expect:**
- Training takes 5-30 minutes depending on your CPU/GPU
- Final validation loss should be < 0.01
- Best model saved to `./models/best_model.pth`
- Training plot saved to `./models/training_history.png`

**Example output:**
```
Epoch [10/100] Train Loss: 0.012456 Val Loss: 0.011234 LR: 1.00e-03
Epoch [20/100] Train Loss: 0.005678 Val Loss: 0.004567 LR: 1.00e-03
...
Epoch [100/100] Train Loss: 0.001234 Val Loss: 0.001456 LR: 5.00e-04
Training complete! Best val loss: 0.001234
```

### 6️⃣ Test Your Trained Model! 🎉

See your neural network in action:

```bash
# Terminal 1: Launch with neural network controller
ros2 launch pidog_gaits nn_demo.launch.py
```

```bash
# Terminal 2: Command the neural network
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
```

**You're now controlling PiDog with a neural network!** 🎊

---

## 📊 Understanding What Happened

### Input → Network → Output

```
Input (4 numbers):                 Neural Network            Output (8 angles):
├─ Gait Type: 0 (walk)        →                       →    ├─ Motor 0: 0.52 rad
├─ Direction: 1 (forward)     →   [Hidden Layers]     →    ├─ Motor 1: -0.14 rad
├─ Turn: 0 (straight)         →   [Learns patterns]   →    ├─ Motor 2: -0.76 rad
└─ Phase: 0.5 (halfway)       →                       →    └─ ... (8 total)
```

The network learned the relationship between what you want (walk forward) and how to move the joints!

---

## 🔍 Verify Your Results

### Check Training Quality

```bash
# View training plot
xdg-open ./models/training_history.png  # Linux
open ./models/training_history.png      # Mac
```

**Good training looks like:**
- Both train/val loss decrease together
- Validation loss < 0.01
- No divergence (overfitting)

### Compare: Traditional vs Neural Network

Run both side-by-side to compare:

**Traditional:**
```bash
ros2 launch pidog_gaits gait_demo.launch.py
```

**Neural Network:**
```bash
ros2 launch pidog_gaits nn_demo.launch.py
```

They should look very similar! If the NN version is jerky or wrong, you may need:
- More training data
- More epochs
- Larger model (`--model large`)

---

## 🎓 What You Learned

Congratulations! You just implemented:

✅ **Imitation Learning** - Teaching AI by demonstration
✅ **Data Collection Pipeline** - Creating training datasets
✅ **Neural Network Training** - Using PyTorch from scratch
✅ **ROS2 Integration** - Deploying ML models in robotics
✅ **End-to-End ML** - From data → training → deployment

---

## 🚀 Next Steps (Advanced)

### 1. Improve the Model
```bash
# Try larger network
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model large \
    --epochs 200

# Tune learning rate
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --lr 0.0001 \
    --epochs 200
```

### 2. Add New Gaits

Edit `pidog_gaits/pidog_gaits/gait_generator_node.py` to add custom movements, then retrain!

### 3. Reinforcement Learning

Once imitation learning works, try reinforcement learning to discover even better gaits!

### 4. Deploy to Real Hardware

Transfer the trained model to a physical PiDog robot.

---

## 🐛 Troubleshooting

### "Model doesn't learn well"
- **Collect more data** - Aim for 10,000+ samples
- **Train longer** - Try 200-300 epochs
- **Use larger model** - `--model large`

### "Robot moves erratically"
- **Check training loss** - Should be < 0.01
- **Verify data quality** - Check `training_data/*.json`
- **Re-collect data** - Sometimes helps

### "ModuleNotFoundError: torch"
```bash
pip3 install torch matplotlib numpy
```

### "colcon command not found"
```bash
source /opt/ros/jazzy/setup.bash
```

---

## 📚 Learn More

- **Full Documentation**: `pidog_gaits/README.md`
- **Neural Network Code**: `pidog_gaits/pidog_gaits/neural_network.py`
- **Training Script**: `pidog_gaits/pidog_gaits/train.py`

---

## 🎉 Congratulations!

You've successfully trained a neural network to control a quadruped robot! This is the same approach used in cutting-edge robotics research.

**Share your results!** Record a video of your trained PiDog and show the world what you built! 🐕🤖

---

*Happy Learning!*
