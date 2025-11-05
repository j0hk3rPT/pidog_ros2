#!/bin/bash

# GPU Training Script for PiDog
# Automatically detects GPU type and runs optimized training

set -e

echo "=============================================="
echo "PiDog GPU Training Setup"
echo "=============================================="
echo ""

# Function to detect GPU
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        GPU_TYPE="nvidia"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    elif command -v rocm-smi &> /dev/null; then
        echo "✅ AMD GPU detected"
        GPU_TYPE="amd"
        rocm-smi --showproductname
    else
        echo "⚠️  No GPU detected, using CPU"
        GPU_TYPE="cpu"
    fi
    echo ""
}

# Function to check PyTorch GPU support
check_pytorch() {
    echo "Checking PyTorch installation..."
    python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" 2>/dev/null || {
    echo "❌ PyTorch not installed or GPU not available"
    echo ""
    echo "Installation options:"
    echo ""
    echo "For AMD GPU (ROCm):"
    echo "  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
    echo ""
    echo "For NVIDIA GPU (CUDA):"
    echo "  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "For CPU only:"
    echo "  pip3 install torch torchvision torchaudio"
    echo ""
    echo "Or use Docker:"
    echo "  docker-compose --profile rocm up -d pidog-rocm  # For AMD"
    echo "  docker-compose --profile cuda up -d pidog-cuda  # For NVIDIA"
    exit 1
}
echo ""
}

# Function to run training
run_training() {
    echo "=============================================="
    echo "Starting Training"
    echo "=============================================="
    echo ""

    # Default parameters
    DATA_PATH="${1:-./training_data/gait_data_*.npz}"
    MODEL="${2:-large}"
    EPOCHS="${3:-200}"
    BATCH_SIZE="${4:-256}"

    echo "Configuration:"
    echo "  Data: $DATA_PATH"
    echo "  Model: $MODEL"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Device: auto (GPU if available)"
    echo ""

    # Check if training data exists
    if ! ls $DATA_PATH 1> /dev/null 2>&1; then
        echo "❌ Training data not found: $DATA_PATH"
        echo ""
        echo "Please collect training data first:"
        echo "  ros2 launch pidog_gaits collect_data.launch.py"
        exit 1
    fi

    # Run training
    python3 -m pidog_gaits.pidog_gaits.train \
        --data $DATA_PATH \
        --model $MODEL \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --device auto \
        --save_dir ./models

    echo ""
    echo "=============================================="
    echo "Training Complete!"
    echo "=============================================="
    echo ""
    echo "Model saved to: ./models/best_model.pth"
    echo "Training plot: ./models/training_history.png"
    echo ""
    echo "Test your model:"
    echo "  ros2 launch pidog_gaits nn_demo.launch.py"
}

# Main execution
detect_gpu
check_pytorch

# If arguments provided, use them; otherwise prompt
if [ $# -eq 0 ]; then
    echo "Usage: $0 [data_path] [model] [epochs] [batch_size]"
    echo ""
    echo "Examples:"
    echo "  $0                                              # Use defaults"
    echo "  $0 ./training_data/gait_data_*.npz simple 100 128"
    echo ""
    read -p "Press Enter to start training with default settings (or Ctrl+C to cancel)..."
    echo ""
fi

run_training "$@"
