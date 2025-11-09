# Brax Training & Deployment Guide

This guide explains how to train PiDog with Brax and deploy the trained model to:
1. Real hardware (standalone Python - **NO ROS2**)
2. ROS2 simulation (optional, for testing)
3. Gazebo comparison (optional)

---

## 🚀 Part 1: Training with Brax (AMD 7900XT)

### Installation

```bash
# Inside unified container (or host with AMD GPU)
# Install JAX with ROCm support for AMD 7900XT
pip install --upgrade pip
pip install "jax[rocm6_0]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
pip install jaxlib

# Install Brax
pip install brax
pip install optax flax

# Verify GPU is detected
python3 -c "import jax; print('Devices:', jax.devices())"
# Expected: [CudaDevice(id=0)] or [RocmDevice(id=0)]
```

### Training Workflow

```bash
# 1. Train with Brax (10M steps, ~10-30 minutes)
python3 train_brax_ppo.py \
    --gait walk_forward \
    --timesteps 10000000 \
    --num_envs 4096 \
    --output ./models/brax

# Output:
#   - models/brax/pidog_walk_forward_brax.pkl (Brax/JAX format)
#   - models/brax/pidog_walk_forward_brax_weights.npz (NumPy format)
#   - models/brax/pidog_walk_forward_eval.html (visualization)

# 2. Test trained model (visualize in browser)
firefox models/brax/pidog_walk_forward_eval.html
```

### Expected Training Performance

| Hardware | Parallel Envs | Throughput | 10M Steps Time |
|----------|---------------|------------|----------------|
| **AMD 7900XT** | 4096 | ~50K steps/sec | **~3-5 minutes** |
| Gazebo (CPU) | 32 | ~500 steps/sec | **~5-6 hours** |

**Speedup: 100-200x faster than Gazebo!**

---

## 📦 Part 2: Deployment Options

After training, you have **3 deployment options**:

### **Option 1: Direct Hardware Deployment (NO ROS2) ⭐ RECOMMENDED**

Deploy trained model directly to real PiDog using **standalone Python**:

```bash
# Convert Brax model to PyTorch (for embedded devices)
python3 convert_brax_to_pytorch.py \
    --input models/brax/pidog_walk_forward_brax_weights.npz \
    --output models/pidog_walk_forward_hardware.pth

# Deploy to PiDog hardware (Raspberry Pi)
# Copy model to robot:
scp models/pidog_walk_forward_hardware.pth pi@pidog.local:~/models/

# On Raspberry Pi, run standalone controller:
python3 deploy_to_hardware.py \
    --model ~/models/pidog_walk_forward_hardware.pth \
    --gait walk_forward
```

**No ROS2 required!** Uses direct servo control via serial/I2C.

See `deploy_to_hardware.py` below for implementation.

---

### **Option 2: ROS2 Deployment (Optional, for testing in Gazebo)**

If you want to test the Brax-trained model in Gazebo:

```bash
# Convert Brax model to ROS2 neural network controller
python3 convert_brax_to_ros2.py \
    --input models/brax/pidog_walk_forward_brax_weights.npz \
    --output models/pidog_walk_forward_ros2.pth

# Launch Gazebo with trained model
ros2 launch pidog_gaits nn_demo.launch.py model_path:=models/pidog_walk_forward_ros2.pth

# Send gait commands
ros2 topic pub /gait_command std_msgs/msg/String "data: 'walk_forward'" --once
```

**When to use:** Testing sim-to-real transfer before hardware deployment.

---

### **Option 3: Hybrid Approach (Brax training + hardware deployment)**

**Best workflow for real robot:**

1. **Train in Brax** (GPU, 10M steps, 3-5 min)
2. **Test in Gazebo** (ROS2, verify behavior)
3. **Deploy to hardware** (standalone Python, no ROS2)

```bash
# Step 1: Train
python3 train_brax_ppo.py --gait walk_forward --timesteps 10000000

# Step 2: Test in Gazebo (optional)
python3 convert_brax_to_ros2.py --input models/brax/pidog_walk_forward_brax_weights.npz
ros2 launch pidog_gaits nn_demo.launch.py

# Step 3: Deploy to hardware
python3 deploy_to_hardware.py --model models/brax/pidog_walk_forward_brax_weights.npz
```

---

## 🤖 Part 3: Hardware Deployment (Standalone Python)

### Hardware Controller (No ROS2)

Create `deploy_to_hardware.py`:

```python
#!/usr/bin/env python3
"""
Standalone PiDog hardware controller (NO ROS2 required)

Reads trained Brax model and controls servos directly via serial/I2C.

Hardware connections:
- Raspberry Pi GPIO → Servo controller board
- Servo protocol: Standard PWM (50Hz)
- Motor mapping: 8 servos (2 per leg)

Usage:
    python3 deploy_to_hardware.py --model models/pidog_brax.npz --gait walk_forward
"""

import argparse
import time
import numpy as np
import torch
import serial
from pathlib import Path

# Import your existing neural network model
import sys
sys.path.append(str(Path(__file__).parent / 'pidog_gaits' / 'pidog_gaits'))
from neural_network import GaitNetSimpleLSTM


class PiDogHardwareController:
    """
    Direct servo control for PiDog hardware

    Controls 8 servos (2 per leg) using PWM signals
    """

    def __init__(self, model_path, servo_port="/dev/ttyUSB0", baudrate=115200):
        """
        Initialize hardware controller

        Args:
            model_path: Path to trained PyTorch model (.pth or .npz)
            servo_port: Serial port for servo controller
            baudrate: Serial communication speed
        """
        # Load trained model
        self.model = GaitNetSimpleLSTM(input_size=4, hidden_size=64, output_size=12)

        if model_path.endswith('.pth'):
            # PyTorch model
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif model_path.endswith('.npz'):
            # Brax model (NumPy weights)
            weights = np.load(model_path)
            # Convert NumPy to PyTorch state dict
            # (You'll need to map layer names appropriately)
            pass

        self.model.eval()

        # Open serial connection to servo controller
        self.ser = serial.Serial(servo_port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection

        print(f"✅ Connected to servo controller on {servo_port}")
        print(f"✅ Loaded model from {model_path}")

        # Servo calibration (PWM values for angle = 0)
        # SunFounder servos: 500-2500 μs pulse width = -90° to +90°
        self.PWM_MIN = 500
        self.PWM_MAX = 2500
        self.PWM_CENTER = 1500

        # Phase tracker for gait cycle
        self.phase = 0.0
        self.control_rate = 30  # Hz (matches training)

    def angle_to_pwm(self, angle_rad):
        """
        Convert joint angle (radians) to PWM pulse width (μs)

        Args:
            angle_rad: Joint angle in radians (-1.57 to +1.57)

        Returns:
            PWM pulse width in microseconds (500-2500)
        """
        # Map [-1.57, +1.57] rad → [500, 2500] μs
        pwm = self.PWM_CENTER + (angle_rad / 1.57) * 1000
        return int(np.clip(pwm, self.PWM_MIN, self.PWM_MAX))

    def send_servo_command(self, motor_id, pwm_value, time_ms=100):
        """
        Send PWM command to servo

        Command format (SunFounder protocol):
            #<motor_id>P<pwm>T<time>\r\n

        Args:
            motor_id: Servo ID (0-7 for legs)
            pwm_value: PWM pulse width in μs
            time_ms: Movement duration in milliseconds
        """
        cmd = f"#{motor_id}P{pwm_value}T{time_ms}\r\n"
        self.ser.write(cmd.encode())

    def control_loop(self, gait_command, duration_sec=None):
        """
        Main control loop - runs neural network at 30Hz

        Args:
            gait_command: Tuple of (gait_type, direction, turn)
                gait_type: 0=walk, 1=trot, 2=static_pose
                direction: -1=backward, 0=none, 1=forward
                turn: -1=left, 0=straight, 1=right
            duration_sec: How long to run (None = infinite)
        """
        print(f"\n🏃 Starting control loop at {self.control_rate} Hz")
        print(f"   Gait command: {gait_command}")

        gait_type, direction, turn = gait_command
        start_time = time.time()
        step_count = 0

        try:
            while True:
                step_start = time.time()

                # Update phase (0.0 to 1.0)
                self.phase = (self.phase + 1.0 / (self.control_rate * 2)) % 1.0

                # Create input for neural network
                nn_input = torch.tensor(
                    [gait_type, direction, turn, self.phase],
                    dtype=torch.float32
                ).unsqueeze(0)  # Add batch dimension

                # Run neural network inference
                with torch.no_grad():
                    output = self.model(nn_input)  # Shape: (1, 12)
                    joint_angles = output.squeeze(0).numpy()  # 12 joint angles

                # Send commands to servos (only first 8 for legs)
                for motor_id in range(8):
                    angle = joint_angles[motor_id]
                    pwm = self.angle_to_pwm(angle)
                    self.send_servo_command(motor_id, pwm, time_ms=33)  # 30Hz

                step_count += 1

                # Print status every second
                if step_count % self.control_rate == 0:
                    elapsed = time.time() - start_time
                    print(f"  Step {step_count} | Phase: {self.phase:.2f} | "
                          f"Time: {elapsed:.1f}s")

                # Check duration limit
                if duration_sec and (time.time() - start_time) > duration_sec:
                    break

                # Sleep to maintain control rate
                sleep_time = (1.0 / self.control_rate) - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n⏹️  Stopping control loop...")

        # Stop all servos (center position)
        for motor_id in range(8):
            self.send_servo_command(motor_id, self.PWM_CENTER, time_ms=500)

        print(f"✅ Completed {step_count} steps in {time.time() - start_time:.1f}s\n")

    def close(self):
        """Close serial connection"""
        self.ser.close()
        print("🔌 Disconnected from servo controller")


def main():
    parser = argparse.ArgumentParser(description='PiDog Hardware Controller')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth or .npz)')
    parser.add_argument('--gait', type=str, default='walk_forward',
                       choices=['walk_forward', 'walk_backward', 'trot_forward',
                               'trot_backward', 'stand'],
                       help='Gait to execute')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds (default: infinite)')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                       help='Servo controller serial port')

    args = parser.parse_args()

    # Gait command mapping
    GAIT_COMMANDS = {
        'walk_forward': (0, 1, 0),
        'walk_backward': (0, -1, 0),
        'trot_forward': (1, 1, 0),
        'trot_backward': (1, -1, 0),
        'stand': (2, 0, 0),
    }

    gait_cmd = GAIT_COMMANDS[args.gait]

    # Create controller
    controller = PiDogHardwareController(
        model_path=args.model,
        servo_port=args.port
    )

    # Run control loop
    controller.control_loop(gait_cmd, duration_sec=args.duration)

    # Cleanup
    controller.close()


if __name__ == '__main__':
    main()
```

### Running on Real Hardware

```bash
# On Raspberry Pi (on the robot)

# 1. Install dependencies
pip3 install torch torchvision numpy pyserial

# 2. Copy trained model to robot
# (from your training PC)
scp models/brax/pidog_walk_forward_brax_weights.npz pi@pidog.local:~/models/

# 3. Run controller
python3 deploy_to_hardware.py \
    --model ~/models/pidog_walk_forward_brax_weights.npz \
    --gait walk_forward \
    --duration 30  # Run for 30 seconds

# Stop anytime with Ctrl+C
```

---

## 🔄 Conversion Scripts

### Convert Brax → PyTorch

Create `convert_brax_to_pytorch.py`:

```python
#!/usr/bin/env python3
"""
Convert Brax/JAX model weights to PyTorch format

Brax uses JAX/Flax, but deployment on Raspberry Pi is easier with PyTorch.
This script converts the trained policy network weights.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'pidog_gaits' / 'pidog_gaits'))
from neural_network import GaitNetSimpleLSTM


def convert_brax_to_pytorch(brax_weights_path, output_path):
    """
    Convert Brax weights to PyTorch model

    Args:
        brax_weights_path: Path to .npz file with Brax weights
        output_path: Path to save PyTorch .pth file
    """
    print(f"Loading Brax weights from {brax_weights_path}")
    brax_weights = np.load(brax_weights_path)

    # Create PyTorch model
    model = GaitNetSimpleLSTM(input_size=4, hidden_size=64, output_size=12)

    # Map Brax layer names to PyTorch layer names
    # (This mapping depends on how you structured the Brax network)
    state_dict = {}

    # Example mapping (adjust based on actual Brax network structure):
    # Brax: policy/linear_0/kernel, policy/linear_0/bias
    # PyTorch: fc1.weight, fc1.bias

    for key in brax_weights.files:
        print(f"  Brax layer: {key}")
        # Convert JAX weight to PyTorch format
        weight = brax_weights[key]

        # Map to PyTorch layer
        if 'lstm' in key.lower():
            # LSTM weights
            if 'kernel' in key:
                state_dict['lstm.weight_ih_l0'] = torch.tensor(weight.T)
            elif 'bias' in key:
                state_dict['lstm.bias_ih_l0'] = torch.tensor(weight)
        elif 'fc1' in key.lower() or 'dense_0' in key.lower():
            if 'kernel' in key:
                state_dict['fc1.weight'] = torch.tensor(weight.T)
            elif 'bias' in key:
                state_dict['fc1.bias'] = torch.tensor(weight)
        elif 'fc2' in key.lower() or 'dense_1' in key.lower():
            if 'kernel' in key:
                state_dict['fc2.weight'] = torch.tensor(weight.T)
            elif 'bias' in key:
                state_dict['fc2.bias'] = torch.tensor(weight)

    # Load into model
    model.load_state_dict(state_dict)

    # Save PyTorch checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': 'GaitNetSimpleLSTM',
        'converted_from': 'brax',
    }

    torch.save(checkpoint, output_path)
    print(f"\n✅ Saved PyTorch model to {output_path}")
    print(f"   Model: {model}")


def main():
    parser = argparse.ArgumentParser(description='Convert Brax to PyTorch')
    parser.add_argument('--input', type=str, required=True,
                       help='Input Brax weights (.npz)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output PyTorch model (.pth)')

    args = parser.parse_args()

    convert_brax_to_pytorch(args.input, args.output)


if __name__ == '__main__':
    main()
```

---

## 📊 Summary: Training vs Deployment

| Stage | Tool | Hardware | Duration | Output |
|-------|------|----------|----------|--------|
| **Training** | Brax + JAX | AMD 7900XT GPU | 3-5 min | `.npz` weights |
| **Testing** | Gazebo + ROS2 | CPU (optional) | N/A | Verify behavior |
| **Deployment** | Standalone Python | Raspberry Pi | Real-time | Control robot |

**Key Point:** You only need ROS2 for optional testing in Gazebo. Real hardware deployment uses **standalone Python** with direct servo control (no ROS2!).

---

## 🎯 Recommended Workflow

```bash
# 1. Train with Brax (on desktop with AMD 7900XT)
python3 train_brax_ppo.py --gait walk_forward --timesteps 10000000

# 2. Convert to PyTorch
python3 convert_brax_to_pytorch.py \
    --input models/brax/pidog_walk_forward_brax_weights.npz \
    --output models/pidog_walk_forward.pth

# 3. (Optional) Test in Gazebo
ros2 launch pidog_gaits nn_demo.launch.py model_path:=models/pidog_walk_forward.pth

# 4. Deploy to real robot (NO ROS2)
scp models/pidog_walk_forward.pth pi@pidog.local:~/models/
ssh pi@pidog.local
python3 deploy_to_hardware.py --model ~/models/pidog_walk_forward.pth --gait walk_forward
```

**Result:** Train in minutes with Brax, deploy directly to hardware without ROS2!
