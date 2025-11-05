#!/usr/bin/env python3
"""
GPU Detection and Verification Script for PiDog Training
Checks PyTorch installation and GPU availability
"""

import sys

def check_gpu():
    print("=" * 70)
    print("PiDog GPU Configuration Check")
    print("=" * 70)
    print()

    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed!")
        print()
        print("Install PyTorch:")
        print()
        print("For CPU only:")
        print("  pip3 install -r requirements-training.txt")
        print()
        print("For NVIDIA GPU (CUDA):")
        print("  pip3 install -r requirements-gpu-cuda.txt")
        print()
        print("For AMD GPU (ROCm):")
        print("  pip3 install -r requirements-gpu-rocm.txt")
        print()
        print("Or use Docker:")
        print("  docker-compose --profile rocm run pidog-rocm  # AMD")
        print("  docker-compose --profile cuda run pidog-cuda  # NVIDIA")
        print()
        return False

    # Check GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}")
    print()

    if torch.cuda.is_available():
        print("🎉 GPU DETECTED!")
        print("-" * 70)
        print(f"GPU Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")

        # Detect GPU type
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'nvidia' in gpu_name or 'geforce' in gpu_name or 'rtx' in gpu_name:
            gpu_type = "NVIDIA (CUDA)"
        elif 'amd' in gpu_name or 'radeon' in gpu_name or 'rx' in gpu_name:
            gpu_type = "AMD (ROCm)"
        else:
            gpu_type = "Unknown"

        print()
        print("-" * 70)
        print(f"GPU Type: {gpu_type}")
        print()

        # Test GPU computation
        print("Testing GPU computation...")
        try:
            x = torch.randn(2000, 2000, device='cuda')
            y = torch.randn(2000, 2000, device='cuda')
            z = torch.mm(x, y)
            print("✅ GPU tensor operations successful!")
            del x, y, z  # Free memory
        except Exception as e:
            print(f"❌ GPU computation failed: {e}")
            return False

        print()
        print("-" * 70)
        print("🚀 GPU READY FOR TRAINING!")
        print()
        print("Recommended training command:")

        vram_gb = props.total_memory / 1e9
        if vram_gb >= 20:
            batch_size = 256
            model = "large"
        elif vram_gb >= 12:
            batch_size = 128
            model = "large"
        elif vram_gb >= 8:
            batch_size = 64
            model = "simple"
        else:
            batch_size = 32
            model = "simple"

        print()
        print(f"  python3 -m pidog_gaits.pidog_gaits.train \\")
        print(f"      --data ./training_data/gait_data_*.npz \\")
        print(f"      --model {model} \\")
        print(f"      --epochs 200 \\")
        print(f"      --batch_size {batch_size} \\")
        print(f"      --device auto")
        print()
        print(f"Or use the convenience script:")
        print(f"  ./train_gpu.sh")

    else:
        print("⚠️  No GPU detected - training will use CPU")
        print()
        print("This is fine but training will be slower (~6-8x)")
        print()
        print("If you have a GPU, check:")
        print("  1. GPU drivers installed (nvidia-smi or rocm-smi)")
        print("  2. Correct PyTorch version installed")
        print()
        print("For NVIDIA: pip3 install -r requirements-gpu-cuda.txt")
        print("For AMD:    pip3 install -r requirements-gpu-rocm.txt")
        print()
        print("Training command (CPU):")
        print()
        print("  python3 -m pidog_gaits.pidog_gaits.train \\")
        print("      --data ./training_data/gait_data_*.npz \\")
        print("      --model simple \\")
        print("      --epochs 100 \\")
        print("      --batch_size 32 \\")
        print("      --device cpu")

    print()
    print("=" * 70)
    return torch.cuda.is_available()


if __name__ == '__main__':
    has_gpu = check_gpu()
    sys.exit(0 if has_gpu else 1)
