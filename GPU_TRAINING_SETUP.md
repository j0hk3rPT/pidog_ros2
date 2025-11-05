# 🚀 GPU-Accelerated Training Setup for PiDog

This guide will help you set up GPU-accelerated training for your PiDog robot dog using either AMD (ROCm) or NVIDIA (CUDA) GPUs.

## 🎯 Quick GPU Detection

First, let's identify your GPU:

```bash
# For NVIDIA GPUs
nvidia-smi

# For AMD GPUs
rocm-smi

# Alternative: Check PCI devices
lspci | grep -i vga
```

## 🔥 Your Setup: 24GB VRAM

With 24GB VRAM, you have excellent capacity for:
- ✅ Training larger neural network models
- ✅ Bigger batch sizes (faster training)
- ✅ Reinforcement learning experiments
- ✅ Multiple training runs in parallel

---

## 📋 Installation Options

### Option 1: Docker (⭐ RECOMMENDED for ROCm)

Docker provides the most reliable setup, especially for AMD GPUs with ROCm.

**Benefits:**
- Pre-configured environment
- No dependency conflicts
- Tested and optimized by AMD/NVIDIA
- Easy to replicate

See **[Docker Setup](#docker-setup-rocm--cuda)** below.

---

### Option 2: Native Installation (pip)

Install PyTorch directly on your system.

#### For AMD GPUs (ROCm)

```bash
# Install ROCm PyTorch (recommended for AMD)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Additional ML dependencies
pip3 install matplotlib numpy

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Supported AMD GPUs (ROCm 6.2+):**
- RX 7900 XTX / XT (24GB / 20GB)
- RX 7800 XT (16GB)
- RX 7700 XT (12GB)
- RX 7600 (8GB)
- Instinct MI series (datacenter)

#### For NVIDIA GPUs (CUDA)

```bash
# Install CUDA PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Additional ML dependencies
pip3 install matplotlib numpy

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

#### For CPU Only (no GPU)

```bash
pip3 install torch torchvision torchaudio matplotlib numpy
```

---

## 🐳 Docker Setup (ROCm & CUDA)

### ROCm Docker Setup (AMD GPUs)

Create a Docker container with PyTorch + ROCm pre-installed:

```bash
# Pull official ROCm PyTorch image
docker pull rocm/pytorch:latest

# Run container with GPU access
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace \
  -w /workspace \
  rocm/pytorch:latest \
  /bin/bash

# Inside container: Verify GPU
python3 -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### CUDA Docker Setup (NVIDIA GPUs)

```bash
# Pull official PyTorch image
docker pull pytorch/pytorch:latest

# Run container with GPU access
docker run --gpus all -it \
  --ipc=host \
  --shm-size 8G \
  -v $(pwd):/workspace \
  -w /workspace \
  pytorch/pytorch:latest \
  /bin/bash

# Inside container: Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 🏋️ Training with GPU

Once PyTorch + GPU support is installed, training automatically uses GPU:

### Basic Training

```bash
# Train with automatic GPU detection
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model simple \
    --epochs 100 \
    --batch_size 64 \
    --device auto
```

The `--device auto` flag will automatically use GPU if available, otherwise CPU.

### Advanced Training (GPU-Optimized)

With 24GB VRAM, you can use larger configurations:

```bash
# Larger model + bigger batch size
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model large \
    --epochs 200 \
    --batch_size 256 \
    --lr 0.001 \
    --device cuda
```

**Optimizations:**
- `--batch_size 256`: Larger batches = faster training with 24GB VRAM
- `--model large`: 532K parameters vs 74K (simple model)
- `--epochs 200`: More training iterations for better accuracy

### Force Specific Device

```bash
# Force GPU
python3 -m pidog_gaits.pidog_gaits.train --device cuda [other args...]

# Force CPU (for testing)
python3 -m pidog_gaits.pidog_gaits.train --device cpu [other args...]
```

---

## ⚡ Performance Comparison

| Hardware | Simple Model (100 epochs) | Large Model (200 epochs) |
|----------|---------------------------|--------------------------|
| **CPU** (Ryzen/Intel) | ~15-30 min | ~60-120 min |
| **GPU** (24GB VRAM) | ~2-5 min | ~10-20 min |

**GPU = 6-8x faster!** 🚀

---

## 🎯 ROCm vs CUDA: Which Should You Use?

### Use ROCm if you have:
- AMD Radeon RX 7000/9000 series
- AMD Instinct MI series
- 24GB VRAM suggests RX 7900 XTX

**Advantages:**
- Open-source
- Excellent VRAM efficiency
- Strong PyTorch support (2025+)
- Cost-effective (AMD GPUs cheaper than NVIDIA)

**Considerations:**
- Slightly more complex setup
- Docker recommended
- 10-30% slower than CUDA in some workloads

### Use CUDA if you have:
- NVIDIA RTX 4090/4080 (24GB/16GB)
- NVIDIA RTX 3090 (24GB)
- Any NVIDIA datacenter GPU

**Advantages:**
- Easier setup
- Best performance
- Widest ecosystem support

---

## 🔍 Verify Your GPU Setup

Run this diagnostic script:

```python
import torch

print("=" * 60)
print("GPU Configuration Check")
print("=" * 60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA/ROCm Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test tensor operation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y  # Matrix multiplication on GPU
    print(f"\n✅ GPU tensor operations working!")
else:
    print("\n⚠️  No GPU detected - training will use CPU")
    print("This is fine but will be slower (~6-8x)")

print("=" * 60)
```

Save as `check_gpu.py` and run:
```bash
python3 check_gpu.py
```

---

## 🎓 Expected Training Times (24GB VRAM)

### Simple Model (74K parameters)
```bash
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model simple \
    --epochs 100 \
    --batch_size 128
```
- **GPU:** ~2-3 minutes
- **CPU:** ~20-30 minutes

### Large Model (532K parameters)
```bash
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model large \
    --epochs 200 \
    --batch_size 256
```
- **GPU:** ~10-15 minutes
- **CPU:** ~90-120 minutes

---

## 🐛 Troubleshooting

### "RuntimeError: No CUDA/ROCm GPUs are available"

**Fix:** Check GPU access
```bash
# For NVIDIA
nvidia-smi

# For AMD
rocm-smi

# Check PyTorch sees GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

### "CUDA out of memory" / "ROCm out of memory"

**Fix:** Reduce batch size
```bash
# Try smaller batch size
python3 -m pidog_gaits.pidog_gaits.train \
    --batch_size 32 \
    [other args...]
```

### ROCm Installation Issues

**Solution:** Use Docker (most reliable)
```bash
# Pull and run ROCm container
docker pull rocm/pytorch:latest
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/pytorch:latest
```

### Performance Lower Than Expected

**Checks:**
1. Verify GPU utilization: `watch -n 1 nvidia-smi` or `watch -n 1 rocm-smi`
2. Increase batch size: `--batch_size 256`
3. Check for CPU bottlenecks (data loading)

---

## 📊 Monitoring GPU Usage

### During Training (Real-time)

**NVIDIA:**
```bash
watch -n 1 nvidia-smi
```

**AMD:**
```bash
watch -n 1 rocm-smi
```

You should see:
- GPU utilization: ~90-100%
- Memory usage: Several GB used
- Temperature: 60-80°C (normal)

---

## 🚀 Next Steps

After setting up GPU training:

1. **Collect Training Data:**
   ```bash
   ros2 launch pidog_gaits collect_data.launch.py
   ```

2. **Train with GPU:**
   ```bash
   python3 -m pidog_gaits.pidog_gaits.train \
       --data ./training_data/gait_data_*.npz \
       --model large \
       --epochs 200 \
       --batch_size 256 \
       --device auto
   ```

3. **Experiment:**
   - Try different learning rates: `--lr 0.0001`
   - Train longer: `--epochs 500`
   - Use larger models for better accuracy

---

## 📚 Additional Resources

- [ROCm Official Docs](https://rocm.docs.amd.com/)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [AMD GPU Support Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## 🎉 Summary

✅ **For 24GB AMD GPU:** Use ROCm with Docker (recommended)
✅ **Training Speed:** 6-8x faster than CPU
✅ **Batch Size:** Can use 128-256 (vs 32-64 on CPU)
✅ **Models:** Both simple and large models work great

Your 24GB VRAM is excellent for robot learning! Happy training! 🐕🤖
