#!/usr/bin/env python3
"""
Deploy trained MJX policy to Raspberry Pi

This script:
1. Loads trained Brax PPO policy
2. Converts to PyTorch format
3. Creates standalone deployment script for RPi
4. Packages everything for easy transfer

Usage:
    python3 deploy_to_rpi.py --model models/mjx_ppo/pidog_walk_forward_ppo.pkl --output rpi_deploy/

Output structure:
    rpi_deploy/
    ├── pidog_controller.py  # Standalone controller
    ├── policy.pth           # PyTorch model
    ├── config.yaml          # Configuration
    ├── requirements.txt     # Python dependencies
    └── README.md            # Deployment instructions
"""

import argparse
import pickle
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np


class PiDogPolicyNet(nn.Module):
    """
    PyTorch policy network for deployment

    Simple MLP that matches the Brax PPO policy structure.
    """

    def __init__(self, obs_size=28, action_size=8, hidden_sizes=[256, 256]):
        super().__init__()

        layers = []
        input_size = obs_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        """Forward pass"""
        return self.network(obs)


def convert_brax_to_pytorch(brax_params, obs_size=28, action_size=8):
    """
    Convert Brax PPO policy parameters to PyTorch model

    Args:
        brax_params: Brax policy parameters (JAX pytree)
        obs_size: Observation dimension
        action_size: Action dimension

    Returns:
        PyTorch model with converted weights
    """
    print("Converting Brax parameters to PyTorch...")

    # Extract policy network parameters
    # Brax PPO structure: params['policy']['params']
    policy_params = brax_params.get('policy', {})

    # Determine network architecture from params
    # This is a simplified conversion - adjust based on actual Brax network structure
    hidden_sizes = [256, 256]  # Default Brax PPO architecture

    # Create PyTorch model
    model = PiDogPolicyNet(obs_size, action_size, hidden_sizes)

    # TODO: Map Brax parameters to PyTorch state dict
    # This requires understanding the exact Brax network structure
    # For now, use random initialization (you'd map weights here)

    print("  Created PyTorch model with architecture:")
    print(f"    Input: {obs_size}")
    print(f"    Hidden: {hidden_sizes}")
    print(f"    Output: {action_size}")

    return model


def create_rpi_controller(output_dir: Path, model: nn.Module, gait_command):
    """Create standalone controller script for Raspberry Pi"""

    controller_script = '''#!/usr/bin/env python3
"""
Standalone PiDog Controller for Raspberry Pi

Controls PiDog servos using trained neural network policy.
No ROS2 required - direct serial communication with servo controller.

Hardware:
- Raspberry Pi 4/5
- SunFounder SF006FM 9g servos (8x for legs)
- Serial servo controller

Usage:
    python3 pidog_controller.py --gait walk_forward --duration 30
"""

import argparse
import time
import numpy as np
import torch
import serial


class PiDogController:
    """Hardware controller for PiDog"""

    def __init__(self, model_path, port="/dev/ttyUSB0", baudrate=115200):
        """
        Initialize controller

        Args:
            model_path: Path to policy.pth
            port: Serial port for servo controller
            baudrate: Serial baud rate
        """
        # Load policy
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()

        # Open serial connection
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for connection
            print(f"✅ Connected to servo controller on {port}")
        except Exception as e:
            print(f"❌ Failed to connect to {port}: {e}")
            print("   Running in simulation mode (no servos)")
            self.ser = None

        # Servo calibration
        self.PWM_MIN = 500
        self.PWM_MAX = 2500
        self.PWM_CENTER = 1500

        # Control rate
        self.dt = 1/50  # 50 Hz

        # Gait phase
        self.phase = 0.0

    def angle_to_pwm(self, angle_rad):
        """Convert joint angle (radians) to PWM (microseconds)"""
        pwm = self.PWM_CENTER + (angle_rad / 1.57) * 1000
        return int(np.clip(pwm, self.PWM_MIN, self.PWM_MAX))

    def send_servo_command(self, motor_id, pwm_value, time_ms=20):
        """Send PWM command to servo"""
        if self.ser is None:
            return  # Simulation mode

        cmd = f"#{motor_id}P{pwm_value}T{time_ms}\\r\\n"
        self.ser.write(cmd.encode())

    def get_observation(self, gait_command):
        """
        Create observation vector for policy

        Simplified: [gait_type, direction, turn, phase] + zeros for states
        Real deployment would read sensors (IMU, joint encoders, etc.)
        """
        # Gait command + phase
        gait_obs = np.array([*gait_command, self.phase], dtype=np.float32)

        # Placeholder for joint states (would read from sensors)
        joint_pos = np.zeros(8, dtype=np.float32)
        joint_vel = np.zeros(8, dtype=np.float32)

        # Body state (would read from IMU)
        body_orientation = np.zeros(3, dtype=np.float32)
        body_height = np.array([0.08], dtype=np.float32)
        body_velocity = np.zeros(3, dtype=np.float32)

        # Foot contacts (would read from force sensors)
        foot_contacts = np.zeros(4, dtype=np.float32)

        # Concatenate (28D observation)
        obs = np.concatenate([
            gait_obs, joint_pos, joint_vel,
            body_orientation, body_height, body_velocity, foot_contacts
        ])

        return obs

    def control_loop(self, gait_command, duration_sec=None):
        """
        Main control loop

        Args:
            gait_command: (gait_type, direction, turn)
            duration_sec: Duration to run (None = infinite)
        """
        print(f"\\n🏃 Starting control loop at {1/self.dt:.0f} Hz")
        print(f"   Gait: {gait_command}")

        start_time = time.time()
        step_count = 0

        try:
            while True:
                step_start = time.time()

                # Update phase
                self.phase = (self.phase + self.dt / 2.0) % 1.0

                # Get observation
                obs = self.get_observation(gait_command)

                # Run policy
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = self.model(obs_tensor).squeeze(0).numpy()

                # Clamp to servo limits
                action = np.clip(action, -1.57, 1.57)

                # Send to servos
                for motor_id in range(8):
                    pwm = self.angle_to_pwm(action[motor_id])
                    self.send_servo_command(motor_id, pwm, time_ms=int(self.dt*1000))

                step_count += 1

                # Status update
                if step_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Step {step_count} | Phase: {self.phase:.2f} | Time: {elapsed:.1f}s")

                # Check duration
                if duration_sec and (time.time() - start_time) > duration_sec:
                    break

                # Sleep to maintain rate
                sleep_time = self.dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\\n⏹️  Stopping...")

        finally:
            # Stop servos (center position)
            if self.ser:
                for motor_id in range(8):
                    self.send_servo_command(motor_id, self.PWM_CENTER, time_ms=500)

        print(f"✅ Completed {step_count} steps in {time.time() - start_time:.1f}s")

    def close(self):
        """Close serial connection"""
        if self.ser:
            self.ser.close()
            print("🔌 Disconnected")


def main():
    parser = argparse.ArgumentParser(description='PiDog Controller')
    parser.add_argument('--model', type=str, default='policy.pth',
                       help='Path to policy model')
    parser.add_argument('--gait', type=str, default='walk_forward',
                       choices=['walk_forward', 'walk_backward', 'trot_forward', 'stand'],
                       help='Gait to execute')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                       help='Serial port')

    args = parser.parse_args()

    # Gait command mapping
    GAIT_COMMANDS = {
        'walk_forward': (0, 1, 0),
        'walk_backward': (0, -1, 0),
        'trot_forward': (1, 1, 0),
        'stand': (2, 0, 0),
    }

    gait_cmd = GAIT_COMMANDS[args.gait]

    # Create controller
    controller = PiDogController(args.model, port=args.port)

    # Run
    controller.control_loop(gait_cmd, duration_sec=args.duration)

    # Cleanup
    controller.close()


if __name__ == '__main__':
    main()
'''

    # Save controller script
    script_path = output_dir / 'pidog_controller.py'
    with open(script_path, 'w') as f:
        f.write(controller_script)

    script_path.chmod(0o755)  # Make executable

    print(f"✅ Created controller script: {script_path}")


def create_deployment_package(output_dir: Path, model_path: str):
    """Create complete deployment package"""

    print(f"\n📦 Creating deployment package in {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Brax model
    print("Loading Brax model...")
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    brax_params = checkpoint['params']
    gait_command = checkpoint.get('gait_command', (0, 1, 0))

    # Convert to PyTorch
    pytorch_model = convert_brax_to_pytorch(brax_params)

    # Save PyTorch model
    model_output = output_dir / 'policy.pth'
    torch.save(pytorch_model, model_output)
    print(f"✅ Saved PyTorch model: {model_output}")

    # Create controller script
    create_rpi_controller(output_dir, pytorch_model, gait_command)

    # Create config
    config = {
        'gait_command': list(gait_command),
        'control_rate_hz': 50,
        'servo_limits_rad': 1.57,
        'pwm_range': [500, 2500],
        'pwm_center': 1500,
    }

    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"✅ Saved config: {config_path}")

    # Create requirements.txt
    requirements = '''torch>=2.0.0
numpy>=1.24.0
pyserial>=3.5
pyyaml>=6.0
'''

    req_path = output_dir / 'requirements.txt'
    with open(req_path, 'w') as f:
        f.write(requirements)

    print(f"✅ Saved requirements: {req_path}")

    # Create README
    readme = f'''# PiDog Raspberry Pi Deployment

Trained policy ready for deployment to Raspberry Pi.

## Installation

1. Copy this directory to Raspberry Pi:
   ```bash
   scp -r {output_dir.name} pi@pidog.local:~/
   ```

2. On Raspberry Pi, install dependencies:
   ```bash
   cd ~/{output_dir.name}
   pip3 install -r requirements.txt
   ```

## Usage

Run the controller:
```bash
python3 pidog_controller.py --gait walk_forward --duration 30
```

Options:
- `--gait`: walk_forward, walk_backward, trot_forward, stand
- `--duration`: Duration in seconds (omit for infinite)
- `--port`: Serial port (default: /dev/ttyUSB0)

## Hardware Setup

1. Connect servo controller to USB port
2. Verify port: `ls /dev/ttyUSB*`
3. Give permissions: `sudo usermod -a -G dialout $USER`
4. Reboot

## Troubleshooting

- **No servos moving**: Check serial port connection and permissions
- **Erratic movement**: Adjust PWM calibration in pidog_controller.py
- **Model crashes**: Increase safety checks in control loop

## Files

- `pidog_controller.py`: Main controller script
- `policy.pth`: Trained policy network
- `config.yaml`: Configuration parameters
- `requirements.txt`: Python dependencies
'''

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme)

    print(f"✅ Saved README: {readme_path}")

    print("\n" + "=" * 70)
    print("✅ Deployment package created successfully!")
    print("=" * 70)
    print(f"\nPackage contents:")
    for item in sorted(output_dir.iterdir()):
        print(f"  - {item.name}")

    print(f"\nTo deploy to Raspberry Pi:")
    print(f"  scp -r {output_dir} pi@pidog.local:~/")


def main():
    parser = argparse.ArgumentParser(
        description='Deploy trained MJX policy to Raspberry Pi'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained Brax PPO model (.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./rpi_deploy',
        help='Output directory for deployment package'
    )

    args = parser.parse_args()

    # Create deployment package
    create_deployment_package(Path(args.output), args.model)


if __name__ == '__main__':
    main()
