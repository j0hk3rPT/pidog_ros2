# 🚀 Quick Start: GPU Training for PiDog

Get your robot dog training on GPU in 5 minutes!

## 📋 Prerequisites

- 24GB VRAM GPU (AMD RX 7900 XTX or NVIDIA RTX 4090/3090)
- Docker (recommended) OR native Python installation
- ~10GB disk space

---

## ⚡ Fast Track (3 Steps)

### Step 1: Check Your GPU

```bash
./check_gpu.py
```

This will tell you:
- GPU type (AMD or NVIDIA)
- Available VRAM
- Recommended settings

### Step 2: Install PyTorch

**Option A: Docker (Recommended for AMD/ROCm)**
```bash
# For AMD GPU
docker-compose --profile rocm run pidog-rocm

# For NVIDIA GPU
docker-compose --profile cuda run pidog-cuda
```

**Option B: Native Installation**
```bash
# For AMD GPU (24GB VRAM)
pip3 install -r requirements-gpu-rocm.txt

# For NVIDIA GPU
pip3 install -r requirements-gpu-cuda.txt
```

Verify installation:
```bash
./check_gpu.py
```

### Step 3: Train!

```bash
# Automated training with optimal settings
./train_gpu.sh
```

That's it! 🎉

---

## 📖 Detailed Walkthrough

### 1. Collect Training Data (10-15 min)

```bash
# Terminal 1: Start data collection
ros2 launch pidog_gaits collect_data.launch.py

# Terminal 2: Cycle through gaits
for gait in walk_forward walk_backward trot_forward trot_backward; do
    echo "Recording: $gait"
    ros2 topic pub /gait_command std_msgs/msg/String "data: '$gait'" --once
    sleep 20
done
```

### 2. Train Neural Network (5-15 min on GPU)

**Automatic (Recommended):**
```bash
./train_gpu.sh
```

**Manual Control:**
```bash
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model large \
    --epochs 200 \
    --batch_size 256 \
    --device auto
```

### 3. Test Your Model

```bash
ros2 launch pidog_gaits nn_demo.launch.py
```

---

## 🎯 Configuration for 24GB VRAM

Your GPU is perfect for fast training! Recommended settings:

### AMD RX 7900 XTX (24GB) - ROCm
```bash
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model large \
    --epochs 200 \
    --batch_size 256 \
    --lr 0.001 \
    --device cuda
```

**Expected time:** 10-15 minutes (vs 90-120 min on CPU)

### NVIDIA RTX 4090/3090 (24GB) - CUDA
```bash
python3 -m pidog_gaits.pidog_gaits.train \
    --data ./training_data/gait_data_*.npz \
    --model large \
    --epochs 200 \
    --batch_size 256 \
    --lr 0.001 \
    --device cuda
```

**Expected time:** 8-12 minutes (vs 90-120 min on CPU)

---

## 🔧 Installation Methods

### Method 1: Docker (Easiest for ROCm)

**For AMD GPU:**
```bash
# One-time setup
docker-compose --profile rocm build

# Run training environment
docker-compose --profile rocm run pidog-rocm

# Inside container
./check_gpu.py
./train_gpu.sh
```

**For NVIDIA GPU:**
```bash
# One-time setup
docker-compose --profile cuda build

# Run training environment
docker-compose --profile cuda run pidog-cuda

# Inside container
./check_gpu.py
./train_gpu.sh
```

### Method 2: Native Installation

**For AMD GPU (ROCm):**
```bash
pip3 install -r requirements-gpu-rocm.txt
./check_gpu.py
```

**For NVIDIA GPU (CUDA):**
```bash
pip3 install -r requirements-gpu-cuda.txt
./check_gpu.py
```

---

## ⚡ Performance Comparison

| Setup | Simple Model (100 epochs) | Large Model (200 epochs) |
|-------|--------------------------|--------------------------|
| **CPU** | ~25 minutes | ~120 minutes |
| **AMD RX 7900 XTX** | ~3 minutes | ~15 minutes |
| **NVIDIA RTX 4090** | ~2 minutes | ~10 minutes |

**GPU = 8-12x faster!** 🚀

---

## 🐛 Troubleshooting

### "PyTorch not installed"
```bash
pip3 install -r requirements-gpu-rocm.txt  # AMD
# or
pip3 install -r requirements-gpu-cuda.txt  # NVIDIA
```

### "No GPU detected"
Check drivers:
```bash
nvidia-smi  # NVIDIA
rocm-smi    # AMD
```

### "Out of memory"
Reduce batch size:
```bash
./train_gpu.sh ./training_data/gait_data_*.npz large 200 128
```

### ROCm installation issues
Use Docker (most reliable):
```bash
docker-compose --profile rocm run pidog-rocm
```

---

## 📊 Monitor Training

### GPU Usage (Real-time)

**NVIDIA:**
```bash
watch -n 1 nvidia-smi
```

**AMD:**
```bash
watch -n 1 rocm-smi
```

Look for:
- GPU Utilization: 90-100%
- Memory Used: ~4-8 GB
- Temperature: 60-80°C

---

## 🎓 Next Steps

After successful training:

1. **Compare Models:**
   ```bash
   # Test traditional gaits
   ros2 launch pidog_gaits gait_demo.launch.py

   # Test neural network
   ros2 launch pidog_gaits nn_demo.launch.py
   ```

2. **Experiment:**
   - Try different learning rates: `--lr 0.0001`
   - Train longer: `--epochs 500`
   - Collect more data for better accuracy

3. **Advanced:**
   - Implement reinforcement learning
   - Add new gaits
   - Deploy to real hardware

---

## 📚 Documentation

- **Complete Guide:** `GPU_TRAINING_SETUP.md`
- **Docker Setup:** `docker-compose.gpu.yml`
- **ROCm Dockerfile:** `Dockerfile.rocm`
- **Training Script:** `train_gpu.sh`

---

## ✅ Quick Checklist

- [ ] GPU detected: `./check_gpu.py`
- [ ] PyTorch installed with GPU support
- [ ] Training data collected: `./training_data/gait_data_*.npz`
- [ ] Training completed: `./models/best_model.pth`
- [ ] Model tested: `ros2 launch pidog_gaits nn_demo.launch.py`

---

## 🎉 You're Ready!

With 24GB VRAM, you have excellent capacity for fast training and experimentation.

**Simple command to start:**
```bash
./train_gpu.sh
```

Happy training! 🐕🤖
