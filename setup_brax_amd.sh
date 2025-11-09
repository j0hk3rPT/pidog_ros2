#!/bin/bash
#
# Setup Brax with AMD ROCm support for PiDog training
# For AMD 7900XT GPU on Ubuntu/Debian systems
#
# Usage:
#   ./setup_brax_amd.sh
#

set -e  # Exit on error

echo "========================================"
echo "PiDog Brax Training Setup (AMD ROCm)"
echo "========================================"
echo ""

# Detect GPU
echo "🔍 Detecting AMD GPU..."
if lspci | grep -i amd | grep -i vga > /dev/null; then
    echo "✅ AMD GPU detected:"
    lspci | grep -i amd | grep -i vga
else
    echo "⚠️  WARNING: No AMD GPU detected!"
    echo "   Brax will run on CPU (much slower)"
fi
echo ""

# Check ROCm installation
echo "🔍 Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm is installed:"
    rocm-smi --showproductname || true
else
    echo "❌ ROCm not found!"
    echo ""
    echo "Please install ROCm 6.0+ first:"
    echo "  https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    echo ""
    echo "Quick install (Ubuntu 22.04):"
    echo "  wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb"
    echo "  sudo apt install ./amdgpu-install_6.0.60002-1_all.deb"
    echo "  sudo amdgpu-install --usecase=rocm"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install JAX with ROCm support
echo "   Installing JAX with ROCm..."
pip install --upgrade "jax[rocm6_0]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Install Brax
echo "   Installing Brax..."
pip install brax

# Install training dependencies
echo "   Installing training libraries..."
pip install optax flax dm-haiku

# Install PyTorch (for model conversion)
echo "   Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install utilities
echo "   Installing utilities..."
pip install numpy matplotlib tqdm

echo ""
echo "✅ Installation complete!"
echo ""

# Verify installation
echo "🧪 Verifying installation..."
python3 << EOF
import sys

try:
    import jax
    print(f"✅ JAX {jax.__version__}")
    print(f"   Devices: {jax.devices()}")

    import brax
    print(f"✅ Brax {brax.__version__}")

    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    import numpy as np
    print(f"✅ NumPy {np.__version__}")

    print("")
    print("🎉 All dependencies installed successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
echo "1. Test Brax environment:"
echo "   python3 pidog_brax_env.py"
echo ""
echo "2. Start training:"
echo "   python3 train_brax_ppo.py --gait walk_forward --timesteps 10000000"
echo ""
echo "3. Expected performance:"
echo "   - AMD 7900XT: ~50K steps/sec"
echo "   - 10M steps in ~3-5 minutes"
echo "   - 100-200x faster than Gazebo!"
echo ""
echo "See BRAX_DEPLOYMENT.md for full workflow."
echo ""
