# M4 Mac Local Models Guide
## Optimal Local Model Combinations for Apple Silicon M4 with 3TB SSD

**Date**: 2025-12-31
**Target Hardware**: MacBook Pro / Mac Studio with M4 Pro/Max
**Storage**: 3TB External SSD (Thunderbolt 4)
**Use Case**: Post-CUDA training inference deployment

---

## Executive Summary

After training your ML models on NVIDIA CUDA hardware, this guide helps you deploy and run the most powerful combination of local models on your M4 Mac with a 3TB external SSD.

### Key Benefits
- **100% Local**: No internet required for inference
- **Privacy**: All data stays on your machine
- **Low Latency**: 50-500ms response times
- **Cost Effective**: No API fees or cloud costs

---

## Hardware Requirements

### Minimum (M4 Pro)
| Component | Specification |
|-----------|--------------|
| Chip | Apple M4 Pro |
| CPU Cores | 14 (10P + 4E) |
| GPU Cores | 20 |
| Neural Engine | 16 cores |
| Unified Memory | 24 GB |
| Storage | 3TB SSD (Thunderbolt 4) |

### Recommended (M4 Max)
| Component | Specification |
|-----------|--------------|
| Chip | Apple M4 Max |
| CPU Cores | 16 (12P + 4E) |
| GPU Cores | 40 |
| Neural Engine | 16 cores |
| Unified Memory | 48-128 GB |
| Storage | 3TB SSD (Thunderbolt 4) |

---

## Model Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Intent                                 │
│  "Create a melancholic ballad about loss with tension"          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │   OLLAMA     │ │   TRAINED    │ │   AUDIO      │
   │   LLMs       │ │   ML MODELS  │ │   PROCESSING │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          │                │                │
          ↓                ↓                ↓
   • Intent Parse    • Emotion→MIDI   • Voice Synth
   • Lyrics Gen      • Chord Predict  • Audio Features
   • Creative AI     • Groove/Timing  • Real-time FX
          │                │                │
          └────────────────┼────────────────┘
                           ↓
                 ┌──────────────────┐
                 │  Generated Music │
                 │  MIDI + Audio    │
                 └──────────────────┘
```

---

## Layer 1: Local LLMs (Ollama)

### Installation

```bash
# Install Ollama
brew install ollama

# Start Ollama service (runs in background)
ollama serve

# Verify installation
ollama --version
```

### Recommended Models by Use Case

#### Creative Writing & Lyrics (Primary)

| Model | Size | VRAM | Context | Best For |
|-------|------|------|---------|----------|
| `llama3.2:8b` | 5 GB | 8 GB | 131K | General creative, intent parsing |
| `mistral:7b` | 4 GB | 6 GB | 32K | Poetry, lyrics, metaphors |
| `llama3.2:70b` | 40 GB | 48 GB | 131K | Maximum quality (48GB+ RAM only) |

**Recommendation**: Start with `llama3.2:8b` - excellent balance of quality and speed.

#### Code & Music Theory

| Model | Size | VRAM | Context | Best For |
|-------|------|------|---------|----------|
| `codellama:7b` | 4 GB | 6 GB | 16K | MIDI parsing, chord analysis |
| `codellama:34b` | 19 GB | 24 GB | 16K | Complex theory reasoning |

**Recommendation**: `codellama:7b` for most music theory tasks.

#### Fast Inference

| Model | Size | VRAM | Context | Best For |
|-------|------|------|---------|----------|
| `phi3:3.8b` | 2 GB | 4 GB | 4K | Quick responses, classification |
| `gemma2:2b` | 1.5 GB | 3 GB | 8K | Lightweight tasks |

**Recommendation**: `phi3:3.8b` for real-time interactions.

#### Embeddings (Semantic Search)

| Model | Size | VRAM | Dimensions | Best For |
|-------|------|------|------------|----------|
| `nomic-embed-text` | 274 MB | 1 GB | 768 | Text similarity, search |
| `mxbai-embed-large` | 670 MB | 2 GB | 1024 | Higher accuracy |

**Recommendation**: `nomic-embed-text` - fast and effective.

### Model Installation Commands

```bash
# Essential models (12 GB total)
ollama pull llama3.2:8b      # 5 GB - Primary creative
ollama pull mistral:7b       # 4 GB - Lyrics/poetry
ollama pull codellama:7b     # 4 GB - Music theory/code

# Fast inference (2 GB)
ollama pull phi3:3.8b        # 2 GB - Quick responses

# Embeddings (300 MB)
ollama pull nomic-embed-text # 274 MB - Semantic search

# OPTIONAL: Maximum quality (requires 48GB+ RAM)
ollama pull llama3.2:70b     # 40 GB - Best quality

# Verify all models
ollama list
```

### Model Profiles

#### Profile: Balanced (24GB RAM)
```yaml
models:
  creative: llama3.2:8b
  lyrics: mistral:7b
  fast: phi3:3.8b
  embed: nomic-embed-text
total_vram: ~16 GB
```

#### Profile: Maximum Quality (48GB+ RAM)
```yaml
models:
  creative: llama3.2:70b
  lyrics: llama3.2:8b
  code: codellama:34b
  fast: phi3:3.8b
  embed: nomic-embed-text
total_vram: ~45 GB
```

#### Profile: Lightweight (16GB RAM)
```yaml
models:
  creative: mistral:7b
  fast: phi3:3.8b
  embed: nomic-embed-text
total_vram: ~8 GB
```

---

## Layer 2: Trained ML Models (Post-CUDA)

These are your custom-trained models from NVIDIA CUDA, deployed to the M4 for inference.

### Core Emotion Pipeline

| Model | Params | Input | Output | Inference |
|-------|--------|-------|--------|-----------|
| EmotionRecognizer | 403K | 128-dim mel | 64-dim embedding | 5 ms |
| MelodyTransformer | 641K | 64-dim emotion | 128-dim notes | 10 ms |
| HarmonyPredictor | 74K | 128-dim context | 64-dim chords | 3 ms |
| GroovePredictor | 18K | 64-dim emotion | 32-dim timing | 2 ms |
| DynamicsEngine | 13K | 32-dim context | 16-dim expression | 1 ms |

**Total Parameters**: 1.15M (~5 MB on disk)
**Total Inference Time**: ~21 ms per generation

### Advanced Models (Optional)

| Model | Params | Input | Output | Inference |
|-------|--------|-------|--------|-----------|
| InstrumentRecognizer | 2M | 128-dim audio | 160-dim dual-head | 10 ms |
| EmotionNodeClassifier | 3M | 128-dim features | 258-dim (6×6×6) | 15 ms |

### Deployment from CUDA to M4

```bash
# On CUDA training machine: Export models
python scripts/export_models.py \
  --checkpoint-dir ./checkpoints \
  --output-dir ./exports \
  --formats pytorch,onnx

# Copy to SSD
rsync -avz --progress exports/ /Volumes/Extreme\ SSD/kelly-project/models/trained/

# On M4: Verify models
python -c "
import torch
device = torch.device('mps')
print(f'MPS available: {torch.backends.mps.is_available()}')

# Load a model
model = torch.load('/Volumes/Extreme SSD/kelly-project/models/trained/emotionrecognizer_best.pt', 
                   map_location=device)
print('✓ Model loaded successfully on MPS')
"
```

### SSD Directory Structure

```
/Volumes/Extreme SSD/
├── kelly-project/
│   ├── models/
│   │   ├── trained/           # PyTorch checkpoints
│   │   │   ├── emotionrecognizer_best.pt
│   │   │   ├── melodytransformer_best.pt
│   │   │   ├── harmonypredictor_best.pt
│   │   │   ├── groovepredictor_best.pt
│   │   │   └── dynamicsengine_best.pt
│   │   ├── onnx/              # ONNX exports
│   │   │   └── *.onnx
│   │   └── coreml/            # CoreML for max Apple optimization
│   │       └── *.mlmodel
│   ├── checkpoints/           # Training checkpoints (keep for reference)
│   └── output/                # Generated content
│
├── kelly-audio-data/
│   ├── raw/
│   │   ├── chord_progressions/lakh/
│   │   ├── melodies/maestro/
│   │   └── nsynth/
│   └── processed/
│
└── kelly-cache/               # Inference cache
    ├── embeddings/
    ├── generations/
    └── models/
```

---

## Layer 3: Audio Processing

### Voice Synthesis Options

#### Option 1: Coqui TTS (Recommended)
```bash
pip install TTS

# Verify installation
python -c "from TTS.api import TTS; print('✓ Coqui TTS ready')"
```

**Models**:
- `tts_models/en/ljspeech/tacotron2-DDC` - Clear, natural voice
- `tts_models/en/vctk/vits` - Multi-speaker

#### Option 2: pyttsx3 (Fallback - Built-in)
```bash
pip install pyttsx3

# Uses macOS built-in voices
python -c "import pyttsx3; e = pyttsx3.init(); print('✓ pyttsx3 ready')"
```

### Audio Analysis

```python
# Using torchaudio with MPS backend
import torch
import torchaudio

device = torch.device('mps')

# Load audio
waveform, sample_rate = torchaudio.load("audio.wav")
waveform = waveform.to(device)

# Compute mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100,
    n_mels=128,
    hop_length=512
).to(device)

mel_spec = mel_transform(waveform)
```

---

## Configuration

### Environment Variables (.env)

```bash
# SSD Paths
SSD_MOUNT=/Volumes/Extreme SSD
DATA_ROOT=/Volumes/Extreme SSD/kelly-audio-data
MODELS_ROOT=/Volumes/Extreme SSD/kelly-project/models
OUTPUT_ROOT=/Volumes/Extreme SSD/kelly-project/output
CHECKPOINTS_ROOT=/Volumes/Extreme SSD/kelly-project/checkpoints

# Ollama Configuration
OLLAMA_MODEL=llama3.2:8b
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=120

# PyTorch MPS Settings
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PYTORCH_ENABLE_MPS_FALLBACK=1

# Cache Settings
IDAW_CACHE_DIR=~/.idaw_cache
IDAW_CACHE_SIZE_GB=10
```

### Build Configuration

Use the config file: `config/build-m4-local-inference.yaml`

```bash
# Load configuration
python -c "
from configs.config_loader import load_config
config = load_config('config/build-m4-local-inference.yaml')
print(f'Device: {config[\"device\"]}')
print(f'Models root: {config[\"paths\"][\"models_root\"]}')
"
```

---

## Performance Benchmarks

### Expected Latencies (M4 Pro, 24GB RAM)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Intent parsing (llama3.2:8b) | 200-500 ms | First token |
| Lyrics generation (mistral:7b) | 300-800 ms | 4 lines |
| Emotion recognition | 5 ms | ML model |
| Melody generation | 15 ms | ML model |
| Full pipeline | 500-1500 ms | End-to-end |

### Memory Usage

| Profile | LLM VRAM | ML VRAM | Total | Available |
|---------|----------|---------|-------|-----------|
| Lightweight | 8 GB | 0.5 GB | 8.5 GB | 7.5 GB free |
| Balanced | 16 GB | 0.5 GB | 16.5 GB | 7.5 GB free |
| Maximum | 45 GB | 0.5 GB | 45.5 GB | 2.5 GB free |

---

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
cd /Volumes/Extreme\ SSD/kelly-project/KmiDi
pip install -e .
pip install -e ".[dev]"

# Install Ollama models
ollama pull llama3.2:8b
ollama pull mistral:7b
ollama pull nomic-embed-text
```

### 2. Start Services

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Music Brain API
python -m music_brain.api --config config/build-m4-local-inference.yaml
```

### 3. Test the Stack

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test Music Brain
curl http://localhost:8000/health

# Generate music from intent
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"emotion": "grief", "genre": "ballad", "bars": 8}'
```

---

## Troubleshooting

### MPS Not Available

```bash
# Check MPS availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# If False, update PyTorch
pip install --upgrade torch torchvision torchaudio
```

### Ollama Connection Failed

```bash
# Check Ollama status
ollama list
pgrep -f ollama

# Restart Ollama
killall ollama
ollama serve
```

### Out of Memory

```bash
# Use smaller models
export OLLAMA_MODEL=mistral:7b

# Or reduce batch size
export IDAW_BATCH_SIZE=8
```

### SSD Not Mounted

```bash
# Check mount
ls /Volumes/

# If not mounted, check Disk Utility or reconnect
diskutil list | grep External
```

---

## Next Steps

1. **Train models on CUDA**: Use `config/build-train-nvidia.yaml`
2. **Export to M4**: Run `python scripts/export_models.py`
3. **Deploy to SSD**: Copy exports to SSD
4. **Configure environment**: Set up `.env` file
5. **Start local stack**: Run Ollama + Music Brain API
6. **Enjoy local AI music generation!**

---

## Related Documentation

- `BUILD_VARIANTS.md` - Hardware configuration guide
- `docs/HARDWARE_TRAINING_SPECS.md` - CUDA training specs
- `docs/TIER123_MAC_IMPLEMENTATION.md` - Full implementation details
- `config/build-m4-local-inference.yaml` - Configuration file
- `cleanup_analysis.md` - Repository cleanup status
