# KMidi Cloud Training & Dataset Guide

This guide covers high-performance training of KMidi's scaled ML models on cloud GPU instances (RunPod, Lambda Labs, AWS).

## 1. Cloud-Optimized Models

We have scaled the core 5-model architecture to support higher resolution musical generation:

| Model | Parameters | Target Hardware |
|-------|------------|-----------------|
| **EmotionRecognizer** | 1.8M | A100/H100 |
| **MelodyTransformer** | 2.5M | A100/H100 |
| **HarmonyPredictor** | 0.8M | RTX 4090+ |
| **DynamicsEngine** | 80K | Local Mac/PC |
| **GroovePredictor** | 90K | Local Mac/PC |

## 2. Dataset Orchestration

Use the `dataset_orchestrator.py` to manage multi-gigabyte datasets across categories.

### Available Datasets
- **MIDI**: Lakh MIDI, MAESTRO v3
- **Audio Emotion**: DEAM, EMO-Music
- **Vocal**: M4Singer
- **General Audio**: NSynth, FMA

### Workflow
```bash
# List available datasets
python scripts/training/dataset_orchestrator.py --list

# Download specific datasets
python scripts/training/dataset_orchestrator.py --download lakh_midi maestro_v3 deam

# Prepare datasets for training (runs KMidi-specific preprocessing)
python scripts/training/dataset_orchestrator.py --prepare lakh_midi
```

## 3. Data Augmentation

To maximize model robustness with scaled parameter counts, we use **Advanced Multi-Modal Augmentation**:

- **Audio**: Time-stretching, pitch-shifting, spectral tilt, noise injection.
- **MIDI**: Transposition, rhythmic jitter, velocity scaling.
- **Emotion**: Gaussian jitter in VAD space (robustness to noisy labels).
- **Coupled**: Synchronized stretching of audio and MIDI pairs.

See `scripts/training/advanced_augmentation.py` for implementation.

## 4. Run from Cursor Workspace (AI-Monitorable)

Cloud training can be started from the repo in Cursor and monitored via workspace files so the AI can report progress.

**Entrypoint**: `./scripts/cloud_training/run_from_workspace.sh`

**Status and logs** (for you and the AI):

- `cloud_training/status.json` – `state` (syncing | running | done | error), `message`, `remote_host`, `updated_at`, `log_file`
- `cloud_training/logs/training_latest.log` – live run log (tee’d from remote)

**Commands**:

```bash
# SSH to any GPU host (e.g. Lambda), optionally with a config
./scripts/cloud_training/run_from_workspace.sh ssh ubuntu@<IP> /workspace/KmiDi [--config config/cloud_training_lambda.yaml]

# Lambda shortcut: set LAMBDA_SSH=ubuntu@<IP>, then:
./scripts/cloud_training/run_from_workspace.sh lambda

# Re-attach to an already-running remote run (tail logs into workspace)
./scripts/cloud_training/run_from_workspace.sh attach ubuntu@<IP> /workspace/KmiDi

# Local Docker (if GPU + Docker on this machine)
./scripts/cloud_training/run_from_workspace.sh docker [--config CONFIG] [--gpus N]
```

When you ask “how’s training?”, the AI should read `cloud_training/status.json` and the last lines of `cloud_training/logs/training_latest.log`.

## 5. Lambda Labs (Max Quality)

For **as well trained as possible** on Lambda, use a **1x A100 (40GB)** instance and the Lambda-optimized config.

**Instance**: Lambda `gpu_1x_a100` (A100 40GB). Best quality/cost for Melody, Harmony, and MIDI Generator.

**Config**: `config/cloud_training_lambda.yaml`

- **Melody Transformer**: 200 epochs, bf16, AMP, cosine, warmup 10 (~30–60 min on A100).
- **Harmony Predictor**: 150 epochs, same precision (~15–30 min on A100).
- **MIDI Generator**: 15–30 epochs, batch 128 (~2–4 h on A100).
- **Data**: `data.root` is `/workspace/data`; manifests expected at `manifests/train_high_res.jsonl` and `manifests/val_high_res.jsonl` relative to that root.

### Quick start (copy-paste)

1. **Start a Lambda A100 instance** and note SSH (e.g. `ubuntu@<ip>` from the Lambda dashboard).

2. **Ensure data is at `/workspace/data` on the instance** (do this before running training):
   ```bash
   export LAMBDA_SSH=ubuntu@<ip>   # use your instance IP
   rsync -avz ~/kmidi_datasets/ "$LAMBDA_SSH:/workspace/data/"
   ```
   Or upload/mount so that manifests and datasets live under `/workspace/data`. The config uses `data.root: "/workspace/data"` and expects `manifests/train_high_res.jsonl` (and `val_high_res.jsonl`) under that root—create that layout or point the config at your layout.

3. **From the repo root** (e.g. in Cursor):
   ```bash
   export LAMBDA_SSH=ubuntu@<ip>
   ./scripts/cloud_training/run_from_workspace.sh lambda
   ```
   Or explicitly:
   ```bash
   ./scripts/cloud_training/run_from_workspace.sh ssh ubuntu@<ip> /workspace/KmiDi --config config/cloud_training_lambda.yaml
   ```

4. Logs stream into `cloud_training/logs/training_latest.log`; status is in `cloud_training/status.json`. If you disconnect, training continues; use `./scripts/cloud_training/run_from_workspace.sh attach ubuntu@<ip> /workspace/KmiDi` to tail again.

## 6. AWS SageMaker (managed training jobs)

For **managed GPU training** on AWS (no EC2/SSH), use SageMaker training jobs with the same entrypoint as the EC2 path (`scripts/aws_train_entrypoint.py`). You provide a custom container image (ECR) that runs that entrypoint; input and output are S3.

- **Setup:** [docs/CLOUD_SAGEMAKER.md](../CLOUD_SAGEMAKER.md) — IAM role, S3 buckets, container contract.
- **Env:** `cp config/env.sagemaker.example .env.sagemaker` and fill (role ARN, image URI, buckets).
- **Launch:** `./scripts/launch_sagemaker_training.py --image-uri <ECR-URI> --package-s3-uri ... --output-s3-uri ... --run-id run-$(date +%Y%m%d-%H%M)`.

Requires a training image in ECR; see the doc for an optional Dockerfile and build/push steps.

## 7. Other Cloud Execution Options

### Automated Remote Deployment (legacy)
```bash
# 1. Spin up a cloud instance (e.g. RunPod)
# 2. Sync and launch from local machine
./scripts/training/cloud_run.sh root@your-gpu-ip /workspace/KmiDi
```

### High-Performance Multi-GPU Launch
For A100/H100 clusters, use the distributed runner **on the cloud instance**:
```bash
./scripts/train_cloud.sh --config config/cloud_training.yaml --gpus 8
```

## 8. Configuration (config/cloud_training.yaml, config/cloud_training_lambda.yaml)

- **Distributed**: Uses PyTorch DDP (`nccl` backend).
- **Precision**: Supports `bf16` (bfloat16) for Ampere+ hardware.
- **AMP**: Automatic Mixed Precision enabled.
- **Scaling**: Architectures configured for the scaled parameter targets.

## 9. Output & Export

- **Checkpoints**: Saved to `checkpoints/cloud_run/`.
- **Exports**: ONNX and Core ML packages with `int8` quantization for local real-time inference.
- **Monitoring**: Weights & Biases (W&B) integration enabled by default.
