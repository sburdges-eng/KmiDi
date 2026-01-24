# 🚀 KmiDi Neural Model Training Sessions

## Overview

This directory contains complete training configurations for two primary paths:

1. **Online GPU Training (Project Vision)** - Train Spectocloud + MIDI Generator models on cloud GPUs.
2. **M4 Mac Metal Session** - Run both models locally with strict 16GB RAM optimization.

An **MLX-only experimental workflow** is also provided to run the full lifecycle on Apple Silicon without CUDA.

---

## 🌐 Online Training Options (Primary Vision)

Use online training for the heavy lifting, then deploy to the M4 for inference and fine-tuning.

**Recommended approach**:
- Train full models on cloud GPUs (faster + cheaper per epoch).
- Export ONNX/CoreML, then run inference + light LoRA fine-tunes on M4.

**Provider shortlist**:
- **Lambda Labs** (RTX 4090): best price/perf for mid-size runs.
- **Vast.ai** (RTX 4090): lowest cost if you can manage spot instances.
- **RunPod** (A100): fastest if budget allows.
- **AWS g5.xlarge** (A10G): reliable for enterprise workflows.

---

## 📁 File Structure

```
training/
├── cuda_session/                    # NVIDIA GPU training
│   ├── spectocloud_training_config.yaml    # Spectocloud ViT config
│   ├── midi_generator_training_config.yaml # MIDI Transformer config
│   ├── train_spectocloud.py               # Spectocloud training script
│   └── train_midi_generator.py            # MIDI Generator training script
│
├── metal_m4_session/                # Apple M4 inference + fine-tuning
│   └── dual_model_config.yaml       # Dual model orchestration config
└── mlx_session/                     # MLX-only experimental workflow
    ├── mlx_workflow.yaml            # Full MLX pipeline config
    └── README.md                    # MLX-only runbook
```

---

## 🎯 Session 1: CUDA GPU Training ($50)

### Recommended Provider & Hardware

| Provider | GPU | VRAM | Cost/hr | Est. Total |
|----------|-----|------|---------|------------|
| **Lambda Labs** | RTX 4090 | 24GB | ~$1.10 | ~$7 |
| **Vast.ai** | RTX 4090 | 24GB | ~$0.80 | ~$5 |
| **RunPod** | A100 | 40GB | ~$2.00 | ~$12 |
| **AWS** | g5.xlarge | A10G 24GB | ~$1.00 | ~$6 |

**Recommended**: Lambda Labs or Vast.ai with RTX 4090 (best price/performance)

### Model Choices

#### 1. Spectocloud Vision Transformer (SpectocloudViT)

| Property | Value |
|----------|-------|
| **Base Architecture** | Swin Transformer (tiny) |
| **Parameters** | ~12M |
| **Input** | Spectrogram (128 mels × 64 frames) + Emotion (64d) + MIDI (32d) |
| **Output** | Point cloud (1200 points × 10 properties) |
| **Inference Target** | 16ms (60 FPS) |
| **Training Time** | ~4-6 hours on RTX 4090 |

**Why This Model?**
- Swin Transformer handles spatial features efficiently
- Small enough for real-time inference on consumer GPUs
- Point cloud output matches Spectocloud visualization structure

#### 2. MIDI Generator Transformer (EmotionMIDITransformer)

| Property | Value |
|----------|-------|
| **Base Architecture** | GPT-2 style decoder |
| **Parameters** | ~25M |
| **Input** | MIDI tokens + Emotion context |
| **Output** | Next token probabilities (388 vocab) |
| **Inference Target** | 5ms per token |
| **Training Time** | ~3-4 hours on RTX 4090 |

**Why This Model?**
- Decoder-only transformer is perfect for autoregressive generation
- Rotary embeddings handle musical timing better than absolute positions
- Cross-attention at select layers enables emotion conditioning without overhead

### Quick Start

```bash
# 1. Connect to GPU instance (example: Lambda Labs)
ssh ubuntu@<your-instance-ip>

# 2. Clone repository
git clone https://github.com/sburdges-eng/KmiDi.git
cd KmiDi

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers timm wandb tensorboard einops pyyaml

# 4. Navigate to training directory
cd training/cuda_session

# 5. Train Spectocloud (first ~4 hours)
python train_spectocloud.py --config spectocloud_training_config.yaml

# 6. Train MIDI Generator (next ~3 hours)
python train_midi_generator.py --config midi_generator_training_config.yaml

# 7. Export models
python export_models.py  # (Creates ONNX + CoreML exports)

# 8. Download trained models
scp -r ubuntu@<ip>:~/KmiDi/checkpoints ./local_checkpoints/
```

### Audio emotion training (optional)
- To train `emotion_recognizer` on real audio instead of synthetic data, provide a JSONL manifest:
  ```
  {"path": "/abs/path/to/audio.wav", "label": "happy"}
  {"path": "/abs/path/to/other.wav", "label": "sad"}
  ```
- Run:
  ```
  python training/train_integrated.py --model emotion_recognizer \
    --audio-manifest data/manifests/emotion_audio.jsonl \
    --use-augmentation
  ```
- Flags: `--sample-rate 16000`, `--n-mels 128`, `--use-augmentation` (time-stretch, pitch-shift, EQ tilt).

### Training Parameters Summary

#### Spectocloud ViT

```yaml
# Key training parameters
batch_size: 32
learning_rate: 1e-4
epochs: 20
max_steps: 50000
warmup_steps: 1000

# Loss weights
position_loss: 1.0 (Chamfer distance)
color_loss: 0.5 (MSE)
property_loss: 0.3 (Smooth L1)

# Augmentation
time_stretch: [0.9, 1.1]
pitch_shift: [-3, 3] semitones
noise_snr: [20, 40] dB
```

#### MIDI Generator

```yaml
# Key training parameters
batch_size: 64
learning_rate: 3e-4
epochs: 15
max_steps: 30000
warmup_steps: 500

# Architecture
hidden_dim: 384
num_layers: 8
num_heads: 6
vocab_size: 388

# Generation
temperature: 0.85
top_k: 40
top_p: 0.92
```

### Data Requirements

The training scripts will auto-generate synthetic data if manifests don't exist. For better results:

1. **Spectocloud Data** (~50K samples recommended):
   - Audio spectrograms paired with emotion labels
   - Ground truth 3D point cloud visualizations
   - Use `scripts/generate_spectocloud_data.py` to create from existing audio

2. **MIDI Data** (~100K samples recommended):
   - MIDI files with emotion annotations
   - Chord progression labels
   - Sources: Lakh MIDI Dataset, personal collection

### Expected Outputs

After training completes:

```
checkpoints/
├── spectocloud/
│   ├── best.pt           # Best PyTorch checkpoint (~50MB)
│   ├── best.onnx         # ONNX export (~50MB)
│   └── best.mlpackage/   # CoreML export for Mac
│
└── midi_generator/
    ├── best.pt           # Best checkpoint (~100MB)
    ├── best.onnx         # ONNX export
    └── best.mlpackage/   # CoreML export
```

---

## 🍎 Session 2: M4 Mac Metal Inference ($50)

### Hardware Requirements

| Spec | Minimum | Recommended |
|------|---------|-------------|
| **Chip** | M1 | M4 Pro/Max |
| **Memory** | 16GB | 24-48GB |
| **macOS** | 14.0 | 15.0+ |

### Setup on M4 Mac

```bash
# 1. Clone repo (if not already)
git clone https://github.com/sburdges-eng/KmiDi.git
cd KmiDi

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Mac-optimized dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install coremltools transformers

# 4. Copy trained models from CUDA session
cp -r local_checkpoints/* models/

# 5. Convert to CoreML (if not already done)
python scripts/convert_to_coreml.py \
    --pytorch models/spectocloud/best.pt \
    --output models/spectocloud/best.mlpackage

python scripts/convert_to_coreml.py \
    --pytorch models/midi_generator/best.pt \
    --output models/midi_generator/best.mlpackage

# 6. Run dual model inference
python -m music_brain.inference.dual_runner \
    --config training/metal_m4_session/dual_model_config.yaml
```

### M4 16GB Optimization Profile

Use these defaults when memory is constrained:

```yaml
batch_size: 2
grad_accum_steps: 8   # Effective batch = 16
precision: fp16
sequence_length: 512
gradient_checkpointing: true
```

### Performance Targets

| Model | Target Latency | Expected on M4 |
|-------|---------------|----------------|
| Spectocloud | 16ms | ~12ms ✅ |
| MIDI Generator | 5ms/token | ~4ms ✅ |
| Combined | 16.67ms (60 FPS) | ~15ms ✅ |

### Fine-tuning on M4 (50-100 Epochs)

The config supports extended LoRA-based fine-tuning for thorough personal style adaptation:

```yaml
# In dual_model_config.yaml
finetune:
  enabled: true
  strategy: lora
  lora_config:
    r: 16
    alpha: 32
    dropout: 0.1
  training:
    batch_size: 2
    grad_accum_steps: 8          # Effective batch = 16
    learning_rate: 1e-4
    min_epochs: 50               # Minimum epochs
    max_epochs: 100              # Maximum epochs
    warmup_epochs: 5
    lr_schedule: "cosine_with_restarts"
    num_cycles: 3                # 3 LR restarts
    early_stopping:
      enabled: true
      patience: 15               # Stop if no improvement for 15 epochs
    save_every_epochs: 10        # Checkpoint every 10 epochs
```

**Estimated Training Time on M4:**
| Dataset Size | Time (50 epochs) | Time (100 epochs) |
|--------------|------------------|-------------------|
| 1,000 samples | ~2 hours | ~4 hours |
| 5,000 samples | ~8 hours | ~16 hours |
| 10,000 samples | ~16 hours | ~32 hours |

Run fine-tuning:
```bash
python -m music_brain.finetune.lora_trainer \
    --config training/metal_m4_session/dual_model_config.yaml \
    --data data/personal/
```

---

## 🍏 Session 3: MLX-Only Workflow (Experimental)

Use MLX to run **the entire project workflow** without CUDA or PyTorch. This is an experiment and should not replace online GPU training.

```bash
cat training/mlx_session/README.md
```

---

## 📊 Model Comparison

| Aspect | Spectocloud ViT | MIDI Generator |
|--------|-----------------|----------------|
| **Type** | Vision/Encoder | Language/Decoder |
| **Parameters** | 12M | 25M |
| **Training Data** | Audio + Emotion | MIDI + Emotion |
| **Output** | Point cloud | Token sequence |
| **Inference** | Single forward pass | Autoregressive |
| **Use Case** | Visualization | Music generation |

---

## 🔧 Troubleshooting

### CUDA Session Issues

**Out of Memory (OOM)**
```bash
# Reduce batch size in config
batch_size: 16  # Instead of 32

# Or enable gradient checkpointing
gradient_checkpointing: true
```

**Slow Training**
```bash
# Enable torch.compile
hardware:
  compile_mode: true
  
# Use Flash Attention
flash_attention: true
```

### M4 Mac Issues

**Model Not Using GPU**
```python
# Verify MPS is available
import torch
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True
```

**CoreML Conversion Fails**
```bash
# Use fp32 first, then quantize
python scripts/convert_to_coreml.py \
    --precision float32 \
    --quantize-after
```

---

## 📈 Monitoring

### WandB Integration

```bash
# Set up WandB
wandb login

# Training will log to:
# - https://wandb.ai/<your-username>/spectocloud
# - https://wandb.ai/<your-username>/midi-generator
```

### TensorBoard

```bash
# View training logs
tensorboard --logdir runs/
```

---

## 🎬 Next Steps After Training

1. **Integration**: Copy trained models to `models/` directory
2. **Validation**: Run `python -m pytest tests/` to verify integration
3. **UI Testing**: Launch Tauri app with `npm run tauri dev`
4. **Performance**: Benchmark with `python benchmarks/model_latency.py`

---

## 💰 Budget Breakdown

### CUDA Session ($50 budget)

| Item | Cost |
|------|------|
| GPU rental (6-8 hrs @ $1.10/hr) | ~$8 |
| Storage (100GB SSD) | ~$2 |
| **Buffer for reruns** | ~$40 |

### M4 Mac Session ($50 budget)

This is primarily a time investment if using your own M4 Mac. The $50 covers:
- Potential cloud Mac rental (if needed)
- Additional storage for datasets
- WandB Pro (optional)

---

## 📚 References

- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [Music Transformer (Magenta)](https://arxiv.org/abs/1809.04281)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [CoreML Optimization Guide](https://developer.apple.com/documentation/coreml)

---

## ✅ Checklist

### Before Training
- [ ] GPU instance provisioned
- [ ] Dependencies installed
- [ ] WandB configured (optional)
- [ ] Data manifests created (or use synthetic)

### After Training
- [ ] Best checkpoints saved
- [ ] ONNX exports created
- [ ] CoreML exports created (for Mac)
- [ ] Models validated with test inference
- [ ] Performance benchmarked

### M4 Deployment
- [ ] Models copied to Mac
- [ ] CoreML conversion verified
- [ ] Dual model runner tested
- [ ] 60 FPS target achieved
- [ ] Tauri integration tested

---

*Last updated: December 31, 2025*
*Version: 2.0*
