# PiDog Gaits - Neural Network Gait Learning

This package implements neural network-based gait learning for the PiDog quadruped robot. It includes:
- Traditional gait generators (Walk, Trot) adapted from SunFounder
- Data collection system for training
- Neural network architecture (PyTorch)
- Training pipeline
- Neural network controller for real-time inference

## 🎯 Project Goal

Train a neural network to learn PiDog's movements instead of using hand-coded gaits. The network learns the mapping:

```
[gait_type, direction, turn, phase] → [8 joint angles]
```

## 📦 Package Contents

### Core Modules
- **`inverse_kinematics.py`** - Converts leg coordinates to joint angles
- **`walk_gait.py`** - Walking gait generator (sequential leg movement)
- **`trot_gait.py`** - Trotting gait generator (diagonal leg pairing)
- **`neural_network.py`** - PyTorch model architectures

### ROS2 Nodes
- **`gait_generator_node.py`** - Traditional gait controller
- **`data_collector.py`** - Records training data
- **`nn_controller.py`** - Neural network inference controller

### Training
- **`train.py`** - Training script with visualization

## 🚀 Quick Start Guide

### Step 1: Build the Package

```bash
cd /home/user/pidog_ros2
colcon build --packages-select pidog_gaits
source install/setup.bash
```

### Step 2: Test Traditional Gaits

First, verify that the traditional gaits work:

```bash
ros2 launch pidog_gaits gait_demo.launch.py
```

This will:
- Launch Webots simulator
- Start the gait generator (default: stand pose)

To switch gaits, in another terminal:

```bash
# Try different gaits
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'sit'" --once
```

**Available gaits:**
- `walk_forward`, `walk_backward`, `walk_left`, `walk_right`
- `trot_forward`, `trot_backward`, `trot_left`, `trot_right`
- `stand`, `sit`, `lie`, `stretch`

### Step 3: Collect Training Data

Now collect data from the traditional gaits to train the neural network:

```bash
ros2 launch pidog_gaits collect_data.launch.py
```

This runs the gait generator and records data. To cycle through different gaits for diverse training data:

```bash
# In another terminal, switch gaits every 30 seconds
for gait in walk_forward walk_backward walk_left walk_right trot_forward trot_backward sit stand; do
    echo "Recording $gait..."
    ros2 topic pub /gait_command std_msgs/msg/String "data: '$gait'" --once
    sleep 30
done
```

Press **Ctrl+C** to stop collecting. Data will be saved to `./training_data/`

You should see:
- `gait_data_YYYYMMDD_HHMMSS.json` - Raw data with metadata
- `gait_data_YYYYMMDD_HHMMSS.npz` - NumPy arrays for training

### Step 4: Train the Neural Network

Install PyTorch if you haven't already:

```bash
pip3 install torch torchvision matplotlib
```

Train the model:

```bash
cd /home/user/pidog_ros2
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model simple \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --save_dir ./models
```

**Training Options:**
- `--model simple` - Smaller model (~200K parameters)
- `--model large` - Larger model (~1M parameters)
- `--epochs 100` - Number of training epochs
- `--device auto` - Automatically use GPU if available

This will:
- Train the neural network
- Save the best model to `./models/best_model.pth`
- Generate a training plot `./models/training_history.png`

**Expected results:**
- Training should converge in 50-100 epochs
- Final validation loss < 0.01 indicates good learning
- Check the plot to ensure no overfitting

### Step 5: Test the Trained Model

Now use the neural network to control PiDog:

```bash
ros2 launch pidog_gaits nn_demo.launch.py
```

This replaces the traditional controller with your trained network!

Switch gaits to see the network in action:

```bash
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
ros2 topic pub /gait_command std_msgs/msg/String "data: 'trot_forward'" --once
```

## 📊 Understanding the System

### Input Features (4 dimensions)
- **gait_type**: 0=walk, 1=trot, 2=static_pose
- **direction**: -1=backward, 0=none, 1=forward
- **turn**: -1=left, 0=straight, 1=right
- **phase**: 0.0 to 1.0 (position in gait cycle)

### Output (8 joint angles)
- Motors 0-7: Leg joints (2 per leg: hip, knee)
- Format: [motor_0, motor_1, ..., motor_7] in radians

### Network Architecture (Simple)
```
Input (4) → Dense(128) → ReLU → Dropout(0.1)
          → Dense(256) → ReLU → Dropout(0.1)
          → Dense(128) → ReLU → Dropout(0.1)
          → Dense(8)
```

## 🎓 Learning Outcomes

By using this system, you'll learn:

1. **Imitation Learning** - Training a network to mimic expert behavior
2. **ROS2 Integration** - How to use ML models in robotics
3. **PyTorch Basics** - Neural network training from scratch
4. **Data Collection** - Creating datasets from simulations
5. **End-to-End Pipeline** - From data → training → deployment

## 🔧 Troubleshooting

### Model doesn't learn well
- Collect more data (at least 10,000 samples)
- Try the larger model: `--model large`
- Increase epochs: `--epochs 200`
- Lower learning rate: `--lr 0.0001`

### Robot moves erratically
- Check validation loss (should be < 0.01)
- Ensure data was collected properly
- Verify model loaded correctly (check terminal output)

### Data collection fails
- Ensure gait_generator is running
- Check topic: `ros2 topic echo /motor_pos`
- Verify Webots is running

## 📈 Advanced: Next Steps

Once basic imitation learning works, you can:

1. **Add More Gaits** - Create new movements and retrain
2. **Reinforcement Learning** - Let the network discover better gaits
3. **Real Hardware** - Deploy to physical PiDog
4. **Terrain Adaptation** - Train on different surfaces
5. **Vision Integration** - Add camera input for obstacle avoidance

## 📝 File Structure

```
pidog_gaits/
├── pidog_gaits/
│   ├── inverse_kinematics.py     # IK solver
│   ├── walk_gait.py               # Walking gait
│   ├── trot_gait.py               # Trotting gait
│   ├── gait_generator_node.py    # Traditional controller
│   ├── data_collector.py          # Data recording
│   ├── neural_network.py          # Model architecture
│   ├── train.py                   # Training script
│   └── nn_controller.py           # NN inference node
├── launch/
│   ├── gait_demo.launch.py       # Demo traditional gaits
│   ├── collect_data.launch.py    # Collect training data
│   └── nn_demo.launch.py         # Demo trained model
├── package.xml
├── setup.py
└── README.md
```

## 🤝 Credits

- Based on SunFounder PiDog gait implementations
- Adapted for ROS2 and neural network learning
- Created as an educational project for learning ML in robotics

## 📄 License

Same as parent pidog_ros2 project.

---

**Happy Learning! 🐕🤖**
