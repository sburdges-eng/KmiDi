# Parallel Training Across Multiple Environments

## Overview

Train all 5 models faster by distributing work across multiple environments:
- GitHub Codespaces (cloud)
- Cursor Desktop (local with GPU)
- VS Code Dev Containers (local)

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│ Codespace A: crispy-funicular   │    │ Codespace B: cautious-couscous  │
│                                 │    │                                 │
│ • emotion_recognizer (2 hrs)    │    │ • harmony_predictor (1 hr)      │
│ • dynamics_engine (1 hr)        │    │ • melody_transformer (3 hrs)    │
│                                 │    │ • groove_predictor (1 hr)       │
│ Total: ~3 hours                 │    │ Total: ~5 hours                 │
└─────────────────────────────────┘    └─────────────────────────────────┘
                    │                                    │
                    └────────────────┬───────────────────┘
                                     ▼
                            ┌───────────────────┐
                            │  Local Mac        │
                            │  /Volumes/sbdrive │
                            │  (27GB datasets)  │
                            └───────────────────┘
```

**Without parallelization:** ~8 hours
**With 2 Codespaces:** ~5 hours (bottleneck is Worker B)

## Quick Start

### Step 1: Set Up SSH Access (on Local Mac)

```bash
# Start SSH server with Tailscale (recommended)
./scripts/ssh_server_for_codespaces.sh --tailscale

# Or use ngrok for NAT traversal
./scripts/ssh_server_for_codespaces.sh --ngrok
```

### Step 2: Add Codespace Secrets

In GitHub repo Settings → Secrets → Codespaces:

| Secret | Value |
|--------|-------|
| `MAC_SSH_HOST` | Your Mac's Tailscale IP |
| `MAC_SSH_USER` | Your Mac username |

### Step 3: Start Codespaces

Open both Codespaces:
- https://crispy-funicular-4j57455g9q77c67p.github.dev/
- https://cautious-couscous-5v4w65vw5pwf4xx7.github.dev/

### Step 4: Run Training

**In Codespace A (crispy-funicular):**
```bash
python scripts/parallel_train.py --worker A
```

**In Codespace B (cautious-couscous):**
```bash
python scripts/parallel_train.py --worker B
```

### Step 5: Collect Results

After both finish, from either Codespace:
```bash
git pull  # Get other worker's models
git add models/ checkpoints/ training_results_*.json
git commit -m "All 5 models trained in parallel"
git push
```

## Model Assignments

### Worker A (Audio-focused)

| Model | Dataset | Epochs | Est. Time |
|-------|---------|--------|-----------|
| emotion_recognizer | M4Singer (WAV) | 50 | 2 hours |
| dynamics_engine | M4Singer (WAV) | 30 | 1 hour |

### Worker B (MIDI-focused)

| Model | Dataset | Epochs | Est. Time |
|-------|---------|--------|-----------|
| harmony_predictor | Lakh MIDI | 30 | 1 hour |
| melody_transformer | Lakh MIDI | 50 | 3 hours |
| groove_predictor | Lakh MIDI | 30 | 1 hour |

## Codespace Resources

Each Codespace has:
- 4 CPU cores
- 8-16 GB RAM
- 32 GB disk (but datasets mounted via SSH)

## Monitoring Progress

### Check Training Status

```bash
# View live output
python scripts/parallel_train.py --worker A

# Check results file
cat training_results_worker_A.json
```

### Check Other Worker

```bash
# From either Codespace, check what's been pushed
git fetch origin
git log origin/main --oneline -5
```

## Optimization Tips

### 1. Pre-download Dependencies

Before training, ensure all packages are installed:
```bash
pip install torch torchaudio librosa
```

### 2. Use Smaller Batch Size if OOM

If you get out-of-memory errors:
```python
# In parallel_train.py, reduce batch_size
"emotion_recognizer": {
    "batch_size": 8,  # Reduce from 16
    ...
}
```

### 3. Resume from Checkpoint

If training is interrupted:
```bash
python scripts/train.py --model emotion_recognizer --resume checkpoints/emotionrecognizer/last.pt
```

### 4. Skip Slow Models

To skip a model temporarily:
```bash
# Edit WORKER_ASSIGNMENTS in parallel_train.py
# Or train specific model:
python scripts/train.py --model emotion_recognizer --data /data/datasets
```

## Troubleshooting

### "Dataset not found"

```bash
# Check mount
ls -la /data/datasets

# Re-mount if needed
bash .devcontainer/setup-ssh-mount.sh
```

### "SSH connection refused"

```bash
# On Mac, ensure SSH is running
./scripts/ssh_server_for_codespaces.sh --tailscale

# Check Tailscale status
tailscale status
```

### "CUDA out of memory"

Codespaces use CPU only. If you see this, reduce batch size:
```bash
python scripts/train.py --model emotion_recognizer --batch-size 4
```

## Using Cursor Desktop

Cursor can run as a third worker with direct access to your Mac's datasets and GPU.

### Setup Cursor

1. Open project in Cursor Desktop
2. Datasets are at `/Volumes/sbdrive/audio/datasets` (no SSH needed)
3. Use MPS (Apple Silicon) or CPU for training

### Optimized Assignment with Cursor

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│ Cursor Desktop (Local Mac)      │    │ Codespace A: crispy-funicular   │
│ Worker C - GPU/MPS accelerated  │    │ Worker A - Cloud CPU            │
│                                 │    │                                 │
│ • melody_transformer (MPS fast) │    │ • emotion_recognizer            │
│ • harmony_predictor             │    │ • dynamics_engine               │
└─────────────────────────────────┘    └─────────────────────────────────┘
                                                      │
┌─────────────────────────────────┐                   │
│ Codespace B: cautious-couscous  │                   │
│ Worker B - Cloud CPU            │◄──────────────────┘
│                                 │    (both access datasets via SSH)
│ • groove_predictor              │
│ • (lighter workload)            │
└─────────────────────────────────┘
```

### Run Training in Cursor

```bash
# Cursor has direct access to datasets
cd "/Volumes/sbdrive/My Mac/Desktop/KmiDi-remote"
source /Volumes/sbdrive/venv/bin/activate

# Train with MPS acceleration (Apple Silicon)
python scripts/train.py --model melody_transformer \
    --data /Volumes/sbdrive/audio/datasets \
    --device mps \
    --epochs 50

# Or use parallel_train.py as Worker C
python scripts/parallel_train.py --worker C
```

### Add Cursor as Worker C

Edit `scripts/parallel_train.py`:

```python
WORKER_ASSIGNMENTS = {
    "A": {
        "name": "crispy-funicular",
        "models": ["emotion_recognizer", "dynamics_engine"],
        "description": "Cloud CPU - Audio models",
    },
    "B": {
        "name": "cautious-couscous",
        "models": ["groove_predictor"],
        "description": "Cloud CPU - Lighter workload",
    },
    "C": {
        "name": "cursor-local",
        "models": ["melody_transformer", "harmony_predictor"],
        "description": "Local MPS - Transformer models (fastest)",
    },
}
```

### Performance Comparison

| Environment | Device | melody_transformer | harmony_predictor |
|-------------|--------|-------------------|-------------------|
| Codespace | CPU (4 core) | ~3 hours | ~1 hour |
| Cursor (M1) | MPS | ~45 mins | ~20 mins |
| Cursor (M2/M3) | MPS | ~30 mins | ~15 mins |

**Cursor with MPS is 3-4x faster for transformer models!**

## Advanced: Three or More Workers

To add more environments, edit `scripts/parallel_train.py`:

```python
WORKER_ASSIGNMENTS = {
    "A": {"models": ["emotion_recognizer"]},
    "B": {"models": ["groove_predictor"]},
    "C": {"models": ["melody_transformer", "harmony_predictor"]},  # Cursor
    "D": {"models": ["dynamics_engine"]},  # Another Codespace
}
```

Then run `--worker C` on Cursor, `--worker D` on another Codespace.
