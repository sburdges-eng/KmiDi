# Cloud GPU Training — Lambda Labs / RunPod / AWS

**Quick reference:** [CLOUD_SETUP_REFERENCE.md](CLOUD_SETUP_REFERENCE.md) | **AWS EC2:** [CLOUD_AWS.md](CLOUD_AWS.md)

**Mac = control plane. Cloud = execution plane.** Never invert that.

Train where compute lives. Control from where your brain lives. Zero file sync; SSH + Docker + tmux.

## Architecture

| Layer | Role |
|-------|------|
| **Mac** | Dataset builder, repo authority, orchestration, experiment launcher |
| **Cloud GPU** | Gradient updates, dataloader workers, checkpoint writes, distributed training |

## One-Time Setup (on cloud instance)

### 1. Connect via SSH

```bash
ssh user@<cloud-ip>
# Or Cursor: Remote Explorer → SSH
```

### 2. Run setup script

```bash
# Clone from your canonical repo (replace URL if different)
./scripts/cloud_setup.sh https://github.com/sburdges-eng/KmiDi
# Or if repo is KmiDi MIDI Companion:
./scripts/cloud_setup.sh <YOUR_REPO_URL>
```

Creates: `/workspace/{datasets,checkpoints,logs,repos}`, builds Docker image.

### 3. Mount persistent volume

**CRITICAL:** Never train on ephemeral root disk. Mount block storage or network volume to `/workspace`. Abort if `/workspace` is on root disk.

## Launch Training (tmux)

```bash
./scripts/cloud_train.sh
# Attach: tmux attach -t kmidi-train
# Detach: Ctrl+B then D
```

Training runs inside tmux. Logs: `/workspace/logs/`. Checkpoints: `/workspace/checkpoints/`.

## Data Pipeline — Pre-build Shards

Before GPU boots, convert manifest to WebDataset shards:

```bash
python scripts/prepare_webdataset_shards.py \
  --manifest ~/Datasets/kmidi_jepa/manifests/aligned.jsonl \
  --output ~/Datasets/kmidi_jepa/shards \
  --shard_size_mb 750
```

Target shard size: 500MB–1GB. Upload shards to `/workspace/datasets/shards/` or symlink.

## Config

- `configs/cloud_training.yaml` — mixed precision, num_workers=8, pin_memory, checkpoint_every_n_steps
- Override manifest/shards path in config or via env

## Auto-Resume

Training script detects latest checkpoint and resumes. No human intervention on node restart.

## Safety Rules

**NEVER:**
- Train inside synced folders
- Write checkpoints to repo directory
- Rely on ephemeral disks
- Duplicate datasets

**ALWAYS:**
- Verify GPU first (`nvidia-smi`)
- Checkpoint aggressively
- Monitor VRAM
- Use tmux for long runs

---

## Cursor Orchestration Prompt

Paste into Cursor agent when setting up cloud training:

```
You are setting up a production-grade CUDA cloud training environment for KmiDi (multimodal audio+MIDI+Spectocloud JEPA).

GOALS:
- Zero file duplication
- Deterministic training
- Resume-safe checkpoints
- Containerized environment
- GPU verification before training
- High disk throughput
- Clean repo structure

EXECUTION PLAN:

1. CONNECT — SSH to cloud GPU. Verify: nvidia-smi

2. PREP STORAGE — Create /workspace, /workspace/datasets, /workspace/checkpoints, /workspace/logs, /workspace/repos. Mount persistent volume to /workspace. Abort if ephemeral root disk.

3. CLONE — git clone <REPO_URL> /workspace/repos/kmidi. Disable auto-git sync.

4. DOCKER — Build from docker/Dockerfile.kmidi-trainer (CUDA 12.1, PyTorch, accelerate, webdataset).

5. GPU VALIDATION — Inside container: python -c "import torch; assert torch.cuda.is_available()". Abort if False.

6. DATA — Pre-build WebDataset shards from manifest. Target 500MB–1GB per shard. num_workers >= 4.

7. LAUNCH — Use scripts/cloud_train.sh. Training runs in tmux (session: kmidi-train). Auto-resume from checkpoint.

8. SAFETY — Never train on ephemeral disk. Never write checkpoints to repo. Never duplicate datasets.

Output: GPU model, VRAM, disk speed, checkpoint path, then begin training.
```

---

## Artifact Storage (recommended)

Push checkpoints to S3, Backblaze B2, or Cloudflare R2 after runs. Nodes die; sadness is optional.
