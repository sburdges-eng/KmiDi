# iDAW Hardware Training Specifications

**Date**: 2025-12-29
**Comparison**: M4 Pro MacBook vs. $500 NVIDIA Budget Build
**Goal**: Actionable setup instructions for both platforms

---

## Quick Comparison

| Metric | M4 Pro 16GB | RTX 4060 (Budget) | Winner |
|--------|-------------|------------------|--------|
| **Training Speed** | 12-15 hours | 3-4 hours | 🔵 RTX 4060 (4x faster) |
| **Initial Cost** | $1600-2000 | $500-650 | 🔵 RTX 4060 |
| **Portability** | ✓ (laptop) | ✗ (desktop) | 🟠 M4 Pro |
| **Power Draw** | ~30W (GPU only) | ~350W (total system) | 🔵 M4 Pro |
| **Setup Complexity** | Easy | Medium | 🔵 M4 Pro |
| **Inference Speed** | 100-150ms | 50-80ms | 🔵 RTX 4060 |

---

## Platform 1: M4 Pro MacBook Pro (16GB RAM, 10-core)

### Hardware Requirements

```
✓ Apple M4 Pro (minimum: 10-core CPU, 16-core GPU)
✓ 16GB unified memory
✓ 512GB+ SSD (for datasets)
✓ macOS 14.6+
```

### Environment Setup (5 minutes)

```bash
# 1. Install Miniforge (ARM64 native)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate

# 2. Create environment
conda create -n idaw-m4 python=3.11 -y
conda activate idaw-m4

# 3. Install PyTorch with MPS support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# 4. Verify MPS
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Expected output: MPS available: True

# 5. Install audio libraries
pip install librosa torchaudio pyyaml tensorboard scipy scikit-learn

# 6. Clone iDAW
git clone https://github.com/yourusername/iDAW.git
cd iDAW
pip install -e .
```

### Optimized Training Configuration

**File**: `config_m4_pro.yaml`

```yaml
device:
  type: "mps"
  fallback: "cpu"  # If MPS fails

model:
  use_compile: true  # torch.compile() for 5-10% speedup
  dtype: "float32"   # MPS stable with float32

training:
  batch_size: 16              # Reduced from 64 to fit 16GB
  accumulation_steps: 4       # Effective batch = 16*4 = 64
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  early_stopping_patience: 10
  early_stopping_min_delta: 0.001

  mixed_precision: false      # MPS prefers full precision
  gradient_clipping: 1.0

optimizer:
  type: "AdamW"
  epsilon: 1e-8
  betas: [0.9, 0.999]

checkpoint:
  save_best: true
  save_latest: true
  save_interval: 100          # Save every 100 steps

data:
  num_workers: 2              # M4 efficient multi-threading
  pin_memory: false           # Not beneficial on MPS
  prefetch_factor: 2
  shuffle_buffer_size: 1000

logging:
  tensorboard: true
  log_interval: 10            # Log every 10 steps
```

### Training Commands

```bash
# Train EmotionRecognizer (3-4 hours)
python scripts/train_emotion_model.py \
  --config config_m4_pro.yaml \
  --data-dir /Volumes/Extreme\ SSD/kelly-audio-data/raw/raw/nsynth \
  --output-dir ./checkpoints/emotion_recognizer \
  --experiment emotion_m4_pro

# Train MelodyTransformer (7-8 hours)
python scripts/train_all_models.py \
  --config config_m4_pro.yaml \
  --models melody_transformer \
  --data-dir /Volumes/Extreme\ SSD/kelly-audio-data/raw/chord_progressions/lakh \
  --output-dir ./checkpoints/melody

# Monitor training
tensorboard --logdir=./runs --port=6006
# Open http://localhost:6006 in browser
```

### Performance Expectations

| Model | Batch Size | Batch Time | Epochs | Total Time |
|-------|------------|-----------|--------|-----------|
| EmotionRecognizer | 16 | 3-4 min | 50 | 2.5-3.5 hrs |
| MelodyTransformer | 16 | 8-10 min | 50 | 6.5-8.5 hrs |
| GroovePredictor | 16 | 1-2 min | 50 | 50-100 min |
| **Combined** | - | - | - | **10-13 hours** |

### Memory Optimization (if running out of memory)

```python
# Reduce batch size further
batch_size = 8
accumulation_steps = 8  # Still effective batch = 64

# Enable gradient checkpointing (trades memory for compute)
model.gradient_checkpointing_enable()

# Use smaller hidden dimensions
model_config = {
    'hidden_dim': 128,      # Reduce from 256
    'num_layers': 2,        # Reduce from 4
    'dropout': 0.2
}
```

### Troubleshooting M4

**Problem**: `RuntimeError: "mps" not available`
```bash
# Solution: Update PyTorch
pip install --upgrade torch torchvision torchaudio
```

**Problem**: Slower than expected (< 50% GPU utilization)
```bash
# Solution: Use torch.compile
model = torch.compile(model, backend="inductor")
# 5-10% speedup expected

# Check Activity Monitor
# GPU should show 80%+ utilization during training
```

**Problem**: Hanging during training (MPS deadlock)
```bash
# Temporary workaround: Disable MPS for that operation
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Or fall back to CPU for problematic layer
x = x.cpu()
x = custom_layer(x)  # Run on CPU
x = x.to(device)
```

---

## Platform 2: Budget NVIDIA RTX 4060 Build ($500-650)

### Hardware Bill of Materials

```
GPU:      RTX 4060 (8GB GDDR6)           $250
CPU:      Ryzen 5 5600X (used)           $120
RAM:      32GB DDR4                      $100
SSD:      1TB NVMe                       $60
PSU:      550W Bronze                    $50
Case:     NZXT H510 Flow                 $80
Cooler:   Arctic Freezer 34 eSports      $30
─────────────────────────────────────────────
Total:                                   $690

(Savings: Reuse old monitor/keyboard/mouse; buy used CPU → $550-600)
```

### Assembly & BIOS Setup (30 minutes)

1. **Install CPU**: Align gold triangle, insert, secure bracket
2. **Install RAM**: Open clips, insert at 45°, press down until clicks
3. **Install SSD**: M.2 slot, screw, no thermal pads needed
4. **Install GPU**: PCIe x16 slot (top), remove bracket covers, secure
5. **Connect PSU**: 24-pin + 8-pin (CPU) + 6-pin (GPU)
6. **BIOS**: F12 during boot → Enable XMP (RAM speed), enable PCIe 4.0

### CUDA Installation (Ubuntu 22.04)

```bash
# 1. Download CUDA Installer
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
chmod +x cuda_12.1.0_530.30.02_linux.run

# 2. Install (requires reboot)
sudo bash cuda_12.1.0_530.30.02_linux.run --silent --driver --toolkit
# Answer prompts: accept EULA, yes to driver, yes to toolkit, no to NSIGHT

# 3. Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. Verify
nvidia-smi
# Expected: NVIDIA RTX 4060, CUDA Version 12.1
```

### Python Environment

```bash
# 1. Create environment
conda create -n idaw-cuda python=3.11 -y
conda activate idaw-cuda

# 2. Install PyTorch CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA available: True, GPU: NVIDIA RTX 4060

# 4. Dependencies
pip install librosa torchaudio pyyaml tensorboard scipy scikit-learn

# 5. Clone iDAW
git clone https://github.com/yourusername/iDAW.git
cd iDAW
pip install -e .
```

### Optimized Training Configuration

**File**: `config_rtx4060.yaml`

```yaml
device:
  type: "cuda"
  cuda_device: 0
  allow_tf32: true          # Minor precision loss, ~10% speedup

model:
  use_compile: true         # torch.compile() for 5% speedup
  dtype: "float32"

training:
  batch_size: 64            # Fits in 8GB with gradient checkpointing
  accumulation_steps: 1
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  early_stopping_patience: 10

  mixed_precision: true     # Automatic Mixed Precision (AMP)
  amp_dtype: "float16"      # Compute in float16, accumulate float32
  scaler_init_scale: 65536
  gradient_checkpointing: false  # Not needed; batch_size=64 fits
  gradient_clipping: 1.0

optimizer:
  type: "AdamW"
  epsilon: 1e-8
  betas: [0.9, 0.999]

checkpoint:
  save_best: true
  save_latest: true
  save_interval: 10         # Save more frequently for safety

data:
  num_workers: 4            # CPU cores → GPU transfer
  pin_memory: true          # DMA acceleration
  prefetch_factor: 2
  shuffle_buffer_size: 5000

logging:
  tensorboard: true
  log_interval: 10
```

### Training Commands

```bash
# Train EmotionRecognizer (50-70 minutes)
python scripts/train_emotion_model.py \
  --config config_rtx4060.yaml \
  --data-dir /path/to/kelly-audio-data/raw/raw/nsynth \
  --output-dir ./checkpoints/emotion_recognizer \
  --experiment emotion_rtx4060

# Monitor GPU
watch -n 1 nvidia-smi
# Expected: 7-8GB VRAM usage, 95%+ utilization

# Monitor training in another terminal
tensorboard --logdir=./runs --port=6006
```

### Performance Expectations

| Model | Batch Size | Batch Time | Epochs | Total Time |
|-------|------------|-----------|--------|-----------|
| EmotionRecognizer | 64 | 1-2 min | 50 | 50-100 min |
| MelodyTransformer | 64 | 2-3 min | 50 | 100-150 min |
| GroovePredictor | 64 | 20-30 sec | 50 | 17-25 min |
| **Combined** | - | - | - | **3-4 hours** |

**Speed vs M4**: 3-5x faster than M4 Pro

### Memory Optimization (Gradient Checkpointing)

If you hit OOM (out of memory):

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or use gradient accumulation
effective_batch_size = 64
batch_size = 32
accumulation_steps = 2  # 2 * 32 = 64 effective

# Update optimizer step
for batch_idx, (x, y) in enumerate(dataloader):
    outputs = model(x)
    loss = criterion(outputs, y) / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Monitoring & Troubleshooting

```bash
# Monitor VRAM usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk "{print \$1/\$2*100 \"%\"}"'

# Monitor power/thermal
watch -n 1 'nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv,noheader,nounits'
# RTX 4060: Should draw 35W GPU only, <65°C

# If throttling occurs: Improve case airflow, add case fans
# RTX 4060 stays cool; unlikely to be thermal issue
```

**Problem**: "CUDA out of memory"
```bash
# Solution 1: Reduce batch_size
batch_size = 32  # Halve it

# Solution 2: Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Solution 3: Restart training
# CUDA memory fragments; restarting helps
```

**Problem**: "CuBLAS error" or CUDA errors
```bash
# Solution: Update drivers
sudo apt update
sudo apt install nvidia-driver-545  # Latest driver

# Verify CUDA compatibility
cuda_12.1 requires nvidia-driver >= 530
nvidia-smi | grep "Driver Version"  # Should be 530+
```

---

## Inference Performance (Both Platforms)

### Latency Benchmarks

```python
# Benchmark script
import torch
import time
from idaw.models import EmotionRecognizer

model = EmotionRecognizer()
model.load_state_dict(torch.load("emotion_recognizer.pt"))
model.eval()

# Warmup
with torch.no_grad():
    for _ in range(10):
        _ = model(torch.randn(1, 128))

# Benchmark
times = []
with torch.no_grad():
    for _ in range(100):
        x = torch.randn(1, 128)
        start = time.time()
        _ = model(x)
        times.append(time.time() - start)

print(f"M4 Pro: {1000*sum(times)/len(times):.1f}ms")
print(f"RTX 4060: {1000*sum(times)/len(times):.1f}ms")
```

| Model | M4 Pro | RTX 4060 |
|-------|--------|----------|
| EmotionRecognizer | 50-100ms | 30-50ms |
| MelodyTransformer | 150-200ms | 80-120ms |
| Combined pipeline | 250-350ms | 150-200ms |

**For real-time inference**: Both are acceptable (< 1 second for interactive UI)

---

## Cost-Benefit Analysis

### Total Cost of Ownership (3 years)

| Aspect | M4 Pro | RTX 4060 |
|--------|--------|----------|
| Hardware | $1800 | $600 |
| Electricity (training 100 hrs/yr × 3yr) | $30 | $250 |
| Maintenance | $0 | $50 (fans) |
| **Total** | **$1830** | **$900** |

### Recommendation Matrix

| Use Case | Recommendation | Rationale |
|----------|---|---|
| **Solo researcher** (training occasionally) | **M4 Pro** | Portable, low power, no setup |
| **Active development** (daily training) | **RTX 4060** | 4x speed, $900 total cost |
| **Production inference** (real-time app) | **Either** | Latency acceptable on both |
| **Budget-conscious** | **RTX 4060** | 3x cheaper, faster |
| **Laptop-first workflow** | **M4 Pro** | Integrated, no desk needed |

---

## Summary Commands

### M4 Pro (One-liner)
```bash
conda create -n idaw-m4 python=3.11 -y && conda activate idaw-m4 && \
pip install torch torchvision torchaudio librosa pyyaml && \
git clone https://github.com/yourusername/iDAW.git && cd iDAW && \
pip install -e . && python scripts/train_emotion_model.py --config config_m4_pro.yaml
```

### RTX 4060 (One-liner)
```bash
curl -L -O https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run && \
sudo bash cuda_12.1.0_530.30.02_linux.run --silent --driver --toolkit && \
conda create -n idaw-cuda python=3.11 -y && conda activate idaw-cuda && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install librosa pyyaml && \
git clone https://github.com/yourusername/iDAW.git && cd iDAW && \
pip install -e . && python scripts/train_emotion_model.py --config config_rtx4060.yaml
```

---

**Version**: 1.0
**Last Updated**: 2025-12-29
**Questions**: Check iDAW docs or open GitHub issue
