# V2 Training Pipeline Runbook

Complete guide for training Spectocloud + MIDI Generator models on CUDA GPU.

## Overview

The v2 training pipeline trains two bridge models:
1. **Spectocloud** - Generates 3D point cloud visualizations from audio + emotion
2. **MIDI Generator** - Generates MIDI sequences from emotion + musical context

Both models use real data (audio/MIDI files) when available, with synthetic fallback for smoke testing.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA 12.1+ support (24GB+ VRAM recommended)
- Recommended: RTX 4090, A100, H100
- Minimum: RTX 3080/3090 (may need reduced batch sizes)

### Software Requirements
- Docker with NVIDIA Container Runtime
- NVIDIA drivers (525.60.13+)
- 50GB+ free disk space

### Dataset Requirements
- Audio files: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`
- MIDI files: `.mid`, `.midi`
- Recommended: 1000+ matched audio-MIDI pairs for meaningful training
- For smoke testing: Pipeline generates synthetic data if manifests are missing

## Quick Start (One Command)

### Basic Training (Synthetic Data)
```bash
./scripts/train_cuda_v2.sh
```

### Training with Real Data
```bash
./scripts/train_cuda_v2.sh \
    --audio-root /path/to/audio \
    --midi-root /path/to/midi
```

### Full Production Run
```bash
./scripts/train_cuda_v2.sh \
    --audio-root /path/to/audio \
    --midi-root /path/to/midi \
    --data-mount /mnt/datasets
```

## Detailed Setup

### 1. GPU Host Setup

#### Install NVIDIA Drivers
```bash
# Check current driver
nvidia-smi

# If needed, install latest driver (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

#### Install Docker + NVIDIA Container Runtime
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# Restart Docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Dataset Preparation

#### Organize Your Data
```
/path/to/datasets/
├── audio/
│   ├── happy/
│   │   ├── track001.wav
│   │   └── track002.wav
│   ├── sad/
│   │   └── track003.wav
│   └── ...
└── midi/
    ├── happy/
    │   ├── track001.mid
    │   └── track002.mid
    ├── sad/
    │   └── track003.mid
    └── ...
```

**Important**: Audio and MIDI files are matched by filename stem (e.g., `track001.wav` matches `track001.mid`).

#### Emotion Inference

The manifest builder uses music_brain's EmotionThesaurus + production rules to generate emotion labels:

**Path-based inference** (automatic):
1. Checks parent directory name (e.g., `happy/`, `sad/`, `angry/`)
2. Checks filename for emotion keywords (e.g., `track_grief_001.wav`)
3. Maps to [valence, arousal, intensity] using Russell's Circumplex Model

**Music-brain integration**:
- Uses `scripts/emotion_helper.py` to convert emotion words to vectors
- Supports 14+ base emotions with intensity tiers
- Valence: -1 (very negative) to +1 (very positive)
- Arousal: -1 (very low energy) to +1 (very high energy)  
- Intensity: 0 (subtle) to 1 (overwhelming)

**Supported emotions**:
- **Happy family**: happy, joy, calm, peaceful, excited, content, tender, energetic
- **Sad family**: sad, melancholy, grief, despair
- **Angry family**: angry, rage, fury
- **Fear family**: fear, anxiety, terror, panic
- **Others**: surprise, disgust, neutral

**Custom emotion labels**:
You can provide your own emotion vectors in the manifest:
```json
{
  "audio_path": "/path/to/audio.wav",
  "midi_path": "/path/to/track.mid",
  "emotion": [0.7, 0.8, 0.6]  // [valence, arousal, intensity]
}
```

This replaces path-based inference with your ground-truth labels.

### 3. Build Manifests

Manifests are JSONL files that map your data for training.

#### Manual Manifest Building
```bash
python scripts/build_manifests.py \
    --audio-root /path/to/audio \
    --midi-root /path/to/midi \
    --out-dir data/manifests \
    --val-split 0.05 \
    --seed 42
```

**With pre-computed point clouds:**
```bash
python scripts/build_manifests.py \
    --audio-root /path/to/audio \
    --midi-root /path/to/midi \
    --generate-clouds \
    --clouds-dir data/clouds
```

**Output:**
- `data/manifests/spectocloud_train.jsonl` - Spectocloud training data
- `data/manifests/spectocloud_val.jsonl` - Spectocloud validation data
- `data/manifests/midi_train.jsonl` - MIDI Generator training data
- `data/manifests/midi_val.jsonl` - MIDI Generator validation data
- `data/clouds/*.npy` - Pre-computed point clouds (if --generate-clouds used)

#### Manifest Format

**Spectocloud:**
```json
{
  "audio_path": "/absolute/path/to/audio.wav",
  "midi_path": "/absolute/path/to/track.mid",
  "emotion": [0.8, 0.7, 0.7]
}
```

**MIDI Generator:**
```json
{
  "midi_path": "/absolute/path/to/track.mid",
  "emotion": [0.8, 0.7, 0.7]
}
```

### 4. Configure Training

Training configs are in `training/cuda_session/`:
- `spectocloud_training_config.yaml` - Spectocloud settings
- `midi_generator_training_config.yaml` - MIDI Generator settings

#### Key Configuration Options

**For Long Runs (75-200 epochs):**
```yaml
training:
  epochs: 150
  min_epochs: 50              # Don't stop before this
  
  early_stopping:
    enabled: true
    patience: 10              # Stop after 10 epochs without improvement
    min_delta: 0.001          # Minimum improvement threshold
```

**For Budget-Constrained Runs:**
```yaml
training:
  epochs: 20
  min_epochs: 10
  
  early_stopping:
    enabled: true
    patience: 5
```

**AMP Configuration:**
```yaml
training:
  amp: true
  amp_dtype: float16          # Use bfloat16 on A100/H100
```

**Gradient Accumulation (for smaller GPUs):**
```yaml
training:
  grad_accum_steps: 8         # Effective batch = batch_size * 8
```

### 5. Run Training

#### Using the Runner Script (Recommended)
```bash
# Full pipeline
./scripts/train_cuda_v2.sh \
    --audio-root /path/to/audio \
    --midi-root /path/to/midi

# Skip manifest building (if already done)
./scripts/train_cuda_v2.sh --skip-manifests

# Train only Spectocloud
./scripts/train_cuda_v2.sh --skip-midi --skip-export

# Train only MIDI Generator
./scripts/train_cuda_v2.sh --skip-spectocloud --skip-manifests
```

#### Manual Docker Run
```bash
# Build image
docker build -f deployment/docker/Dockerfile.cuda -t kmidi-cuda-train:v2 .

# Run training
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -v /path/to/datasets:/data \
    -w /workspace \
    kmidi-cuda-train:v2 \
    bash -c "cd training/cuda_session && python train_spectocloud.py"
```

#### Direct Python (for Debugging)
```bash
# Activate environment
pip install -r training/cuda_session/requirements.txt

# Train Spectocloud
cd training/cuda_session
python train_spectocloud.py --config spectocloud_training_config.yaml

# Train MIDI Generator
python train_midi_generator.py --config midi_generator_training_config.yaml
```

### 6. Monitor Training

Training logs are printed to stdout. Key metrics:

**Spectocloud:**
- `pos_loss` - Chamfer distance (point cloud positions)
- `color_loss` - Color MSE
- `val_loss` - Combined validation loss

**MIDI Generator:**
- `loss` - Cross-entropy loss
- `accuracy` - Token prediction accuracy
- `perplexity` - Language model perplexity

### 7. Export Models

#### Automatic Export (via Runner Script)
```bash
# Already included in full pipeline
./scripts/train_cuda_v2.sh
```

#### Manual Export
```bash
# Export both models
docker run --rm -it \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    kmidi-cuda-train:v2 \
    bash -c "cd training/cuda_session && python export_models.py --all"

# Export specific model
python training/cuda_session/export_models.py --spectocloud checkpoints/spectocloud/best.pt
python training/cuda_session/export_models.py --midi-generator checkpoints/midi_generator/best.pt
```

**Output:**
- `exports/spectocloud.onnx` - ONNX model for C++ bridge
- `exports/midi_generator.onnx` - ONNX model for C++ bridge
- `exports/*.mlpackage` - CoreML models (if coremltools installed)

### 8. Output Locations

After training completes:

```
checkpoints/
├── spectocloud/
│   ├── best.pt                    # Best validation model
│   ├── checkpoint_epoch_5.pt      # Regular checkpoints
│   └── checkpoint_epoch_10.pt
└── midi_generator/
    ├── best.pt
    └── checkpoint_epoch_5.pt

exports/
├── spectocloud.onnx
├── midi_generator.onnx
├── spectocloud.mlpackage/
└── midi_generator.mlpackage/

data/manifests/
├── spectocloud_train.jsonl
├── spectocloud_val.jsonl
├── midi_train.jsonl
└── midi_val.jsonl

cache/                             # Optional: cached features
├── spectrograms/
└── midi_tokens/
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```yaml
data:
  batch_size: 16              # Default: 32 for Spectocloud, 64 for MIDI
```

Or increase gradient accumulation:
```yaml
training:
  grad_accum_steps: 8
```

### No Matching Audio-MIDI Pairs
Check that filenames match (excluding extension):
```bash
# Should match
audio/happy/track001.wav
midi/happy/track001.mid

# Won't match
audio/happy/song_001.wav
midi/happy/track001.mid
```

### Synthetic Data Warnings
If you see:
```
WARNING: Manifest not found: data/manifests/spectocloud_train.jsonl
WARNING: Generating synthetic training data for smoke testing only!
```

This means manifests weren't built. Either:
1. Run `scripts/build_manifests.py` first, OR
2. Use `--audio-root` and `--midi-root` with the runner script

### Training Too Slow
Enable AMP and increase workers:
```yaml
training:
  amp: true
  amp_dtype: bfloat16         # On A100/H100

data:
  num_workers: 16
  prefetch_factor: 4
```

### MIDI Vocab Issues
The MIDI tokenizer uses vocab size **388** with special tokens 384-387:
- 384: PAD
- 385: BOS (beginning of sequence)
- 386: EOS (end of sequence)
- 387: BAR (bar marker)

Do not change this or you'll break compatibility with the tokenizer.

## Advanced Options

### Pre-computed Point Clouds

Generate target point clouds ahead of time to speed up training:

```bash
python scripts/build_manifests.py \
    --audio-root /path/to/audio \
    --midi-root /path/to/midi \
    --generate-clouds \
    --clouds-dir data/clouds
```

**Benefits:**
- Faster data loading (no on-the-fly generation)
- Deterministic point clouds across runs
- Can provide custom ground-truth visualizations

**Format:** Point clouds are saved as `.npy` files (1200×3 float32 arrays) with hash-based filenames.

The dataset loader automatically uses pre-computed clouds if `target_pointcloud_path` is in the manifest, otherwise generates on-the-fly.

### Training on Multiple GPUs

Use PyTorch's distributed data parallel (DDP) for multi-GPU training:

**Single-node multi-GPU:**
```bash
# Inside Docker container or on GPU host
torchrun --nproc_per_node=4 \
    training/cuda_session/train_spectocloud.py \
    --config spectocloud_training_config.yaml

torchrun --nproc_per_node=4 \
    training/cuda_session/train_midi_generator.py \
    --config midi_generator_training_config.yaml
```

**Multi-node training (advanced):**
```bash
# Node 0 (master)
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.10 \
    --master_port=29500 \
    training/cuda_session/train_spectocloud.py

# Node 1
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.10 \
    --master_port=29500 \
    training/cuda_session/train_spectocloud.py
```

**Configuration adjustments for multi-GPU:**
```yaml
data:
  batch_size: 16              # Per-GPU batch size
  num_workers: 8              # Per-GPU workers

training:
  grad_accum_steps: 1         # Usually not needed with multi-GPU
```

**Expected speedup:**
- 2 GPUs: ~1.8x faster
- 4 GPUs: ~3.5x faster
- 8 GPUs: ~6.5x faster

Note: Efficiency decreases with more GPUs due to communication overhead.

### Resuming from Checkpoint
Edit config to load checkpoint:
```yaml
training:
  resume_from: "checkpoints/spectocloud/checkpoint_epoch_20.pt"
```

### Custom Emotion Embeddings
Manifests accept emotion vectors of length 3 or 64:
- **3D:** `[valence, arousal, intensity]` in range [-1, 1]
- **64D:** Full emotion embedding (auto-padded if 3D)

## Budget Estimation

Based on RTX 4090 hourly rates (~$0.50-1.00/hr):

| Configuration | GPU Hours | Estimated Cost |
|--------------|-----------|----------------|
| Smoke test (synthetic, 5 epochs) | 0.5 | $0.50 |
| Medium run (real data, 20 epochs) | 4-6 | $3-6 |
| Production run (75-150 epochs) | 12-24 | $10-25 |

A100/H100 will be 2-3x faster but 2-3x more expensive per hour.

## Next Steps

After training:
1. **Validate exports:** Test ONNX models with `onnxruntime`
2. **Integrate with C++ bridge:** Load ONNX in penta_core
3. **(Optional) Convert to CoreML:** For Apple Silicon deployment
4. **Fine-tune:** Adjust configs based on validation metrics

## Support

For issues or questions:
1. Check training logs for errors
2. Review configs in `training/cuda_session/`
3. Verify dataset manifests are correctly formatted
4. Test with synthetic data first (no `--audio-root`/`--midi-root`)
