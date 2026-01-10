# AWS GPU Training Setup - Quick Start

## Architecture Overview

**Your M4 Mac (10 GPU cores, Metal)** → Development, testing, local inference  
**AWS NVIDIA GPUs (CUDA)** → Cloud training (rent by the hour)

Your M4 MacBook Pro is NOT used for training - it's your development machine. Training happens on rented NVIDIA GPUs in AWS, then you deploy the trained models back to your Mac for local inference.

---

## 1. Launch EC2 Instance

### Recommended Instance Types
| Instance | GPUs | VRAM | Spot $/hr | $150 Budget Gets | Recommendation |
|----------|------|------|-----------|------------------|----------------|
| g5.2xlarge | 1x A10G | 24GB | $1.21 | ~124 hrs | Slow but cheap |
| **g5.12xlarge** | **4x A10G** | **96GB** | **$4.86** | **~30 hrs** | **⭐ Best value** |
| p3.8xlarge | 4x V100 | 64GB | $3.06 | ~49 hrs | Good alternative |
| p4d.24xlarge | 8x A100 | 320GB | $10.00 | ~15 hrs | Overkill/expensive |

**Recommendation: g5.12xlarge with 4x A10G**
- Modern architecture (Ampere vs V100's Volta)
- 4 GPU distributed training = 4x speedup over single GPU
- fp16 precision is sufficient for this model (bf16 only marginal benefit)
- A100/H100 unnecessary - they excel at large language models, not this workload

### AMI Selection
Choose: **Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 22.04)**
- AMI ID: Search for "Deep Learning AMI GPU PyTorch" in your region
- Pre-installed: CUDA 12.1, cuDNN 8.9, PyTorch 2.1+

## 2. Instance Configuration

### Security Group
- SSH (port 22) from your IP
- Optional: TensorBoard (port 6006) if you want remote monitoring

### Storage
- Root volume: 100GB EBS (gp3) minimum
- Increase to 500GB+ if storing datasets on instance

### Key Pair
- Create or use existing SSH key pair
- Save the `.pem` file securely

## 3. SSH Connection

```bash
# Set correct permissions
chmod 400 ~/path/to/your-key.pem

# Connect to instance
ssh -i ~/path/to/your-key.pem ubuntu@<INSTANCE_PUBLIC_IP>
```

## 4. Setup KmiDi

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Clone repository
git clone https://github.com/sburdges-eng/KmiDi.git
cd KmiDi

# Verify GPU access
nvidia-smi

# Install any missing dependencies (usually pre-installed on DL AMI)
pip install einops pretty_midi librosa timm transformers

# Optional: Setup Weights & Biases for experiment tracking
pip install wandb
wandb login  # Enter your API key
```

## 5. Prepare Dataset

### Option A: Upload from Local
```bash
# From your local machine:
scp -i your-key.pem -r /local/audio/path ubuntu@<IP>:/home/ubuntu/data/audio
scp -i your-key.pem -r /local/midi/path ubuntu@<IP>:/home/ubuntu/data/midi
```

### Option B: Download from S3
```bash
# On EC2 instance:
aws s3 sync s3://your-bucket/audio /home/ubuntu/data/audio
aws s3 sync s3://your-bucket/midi /home/ubuntu/data/midi
```

### Option C: Use Public Datasets
```bash
# Example: Download from common sources
mkdir -p ~/data/audio ~/data/midi

# Download your training data here
# (Add specific dataset download commands as needed)
```

## 6. Run Training

### Test Setup (Dry Run)
```bash
./scripts/train_cuda_v2.sh \
    --audio-root ~/data/audio \
    --midi-root ~/data/midi \
    --dry-run
```

### Start Training - Single GPU
```bash
./scripts/train_cuda_v2.sh \
    --audio-root ~/data/audio \
    --midi-root ~/data/midi \
    --num-gpus 1
```

### Start Training - Multi-GPU (Recommended for g5.12xlarge)
```bash
./scripts/train_cuda_v2.sh \
    --audio-root ~/data/audio \
    --midi-root ~/data/midi \
    --num-gpus 4
```

### With W&B Logging
```bash
./scripts/train_cuda_v2.sh \
    --audio-root ~/data/audio \
    --midi-root ~/data/midi \
    --num-gpus 4 \
    --wandb-key YOUR_WANDB_API_KEY
```

### Resume from Checkpoint
```bash
./scripts/train_cuda_v2.sh \
    --audio-root ~/data/audio \
    --midi-root ~/data/midi \
    --num-gpus 4 \
    --resume ./outputs/spectocloud_20260101_120000/checkpoint_epoch_10.pt
```

## 7. Monitor Training

### Via TensorBoard (Local)
```bash
# On EC2 instance
tensorboard --logdir ./outputs --bind_all
```

Then access at `http://<INSTANCE_PUBLIC_IP>:6006` (if port is open in security group)

### Via SSH Tunnel
```bash
# On your local machine
ssh -i your-key.pem -L 6006:localhost:6006 ubuntu@<INSTANCE_PUBLIC_IP>

# Then access http://localhost:6006 in your browser
```

### Via tmux (Recommended for long runs)
```bash
# Start tmux session
tmux new -s training

# Run training inside tmux
./scripts/train_cuda_v2.sh --audio-root ~/data/audio --midi-root ~/data/midi --num-gpus 4

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t training
# List sessions: tmux ls
```

## 8. Retrieve Results

### Download to Your M4 Mac
```bash
# From your M4 MacBook Pro:
scp -i your-key.pem -r ubuntu@<IP>:/home/ubuntu/KmiDi/outputs ~/KmiDi/trained_models

# Or sync to S3, then download to Mac:
# On EC2:
aws s3 sync ./outputs s3://your-bucket/kmidi-training-runs/

# On your M4 Mac:
aws s3 sync s3://your-bucket/kmidi-training-runs/ ~/KmiDi/trained_models
```

### Deploy to M4 for Inference
After downloading trained models to your Mac:
```bash
# On your M4 Mac:
cd ~/KmiDi
python scripts/deploy_trained_models.py \
    --model-dir ~/KmiDi/trained_models/spectocloud_20260101_120000 \
    --target m4-inference

# Now you can run inference locally on your M4!
```

## 9. Cost Management

### Spot Instances
- Save 60-90% vs on-demand pricing
- Risk: Can be interrupted (rare for training workloads)
- Use for development and non-critical runs

### Stop vs Terminate
- **Stop**: Preserves data, still charges for EBS storage (~$0.10/GB/month)
- **Terminate**: Deletes everything, no charges
- Always sync results to S3 before terminating!

### Budget Alerts
```bash
# Example 12-hour training on g5.12xlarge spot
# Cost: 12 hours × $4.86/hr = ~$58.32
# With overhead + storage ≈ $65-75

# 18-hour training ≈ $87-97
# 30-hour training ≈ $146-150 (max budget)
```

## 10. Troubleshooting

### CUDA Out of Memory
```bash
# Edit config to reduce batch size
nano training/cuda_session/spectocloud_training_config.yaml

# Look for:
training:
  batch_size: 32  # Try 16 or 8
```

### Slow Data Loading
```bash
# Check I/O wait
iostat -x 1

# Solution: Use instance store (ephemeral) for data
# Or increase EBS IOPS
```

### Training Hangs
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# If 0% GPU usage, likely data pipeline issue
# Check training log for errors
tail -f outputs/spectocloud_*/training.log
```

### Connection Dropped
```bash
# Always use tmux or screen for long-running jobs!
# Re-attach after reconnecting:
tmux attach -t training
```

## Validation Protocol (Audio-Based Emotion Extraction)

1) **Ground truth & manifest**
    - Use a labeled dataset (e.g., DEAM/EMO-Music) plus any clinician-labeled clips as a fixed gold test split.
    - Build a manifest CSV: `id, path_audio, split(train/val/test), emotion_label, valence, arousal, intensity, genre, tempo, annotator`.
    - Balance classes; stratify by genre/tempo to reduce bias.

2) **Protocol & metrics**
    - Splits: k-fold on train/val; clinician gold set held out for test.
    - Metrics: per-class Precision/Recall/F1, macro-F1, confusion matrix; valence/arousal MAE/MSE/CCC; calibration (ECE) on probabilities; agreement vs clinician (Cohen/Fleiss kappa) with inter-rater as ceiling.
    - Preprocessing: consistent loudness (e.g., LUFS) and documented seed/config.

3) **Run validation**
    - Generate predictions on val/test; compute metrics, confusion matrix, valence–arousal scatter, calibration plot; slice metrics by genre/tempo/loudness.

4) **Error analysis**
    - Inspect top confusions (sad↔calm, excited↔happy), listen to misclassified clips, check correlation with tempo/loudness/instrumentation.

5) **Artifacts to save**
    - `metrics.json`, `confusion_matrix.png`, `calibration.png`, `valence_arousal_scatter.png`, manifest hash, model checkpoint ID, seed, and preprocessing settings.

This keeps validation reproducible and aligned with music-therapy-relevant labels.

## Quick Reference Commands

```bash
# Monitor GPUs
watch -n 1 nvidia-smi

# Monitor disk space
df -h

# Monitor training log
tail -f outputs/spectocloud_*/training.log

# Kill training
pkill -f train_spectocloud

# Check running processes
ps aux | grep train_spectocloud
```

## Example Full Workflow

```bash
# 1. Launch g5.12xlarge spot instance with Deep Learning AMI

# 2. SSH in
ssh -i kmidi-key.pem ubuntu@<IP>

# 3. Setup
git clone https://github.com/sburdges-eng/KmiDi.git
cd KmiDi
nvidia-smi  # Verify 4 GPUs

# 4. Prepare data (example with S3)
aws s3 sync s3://my-bucket/audio ~/data/audio
aws s3 sync s3://my-bucket/midi ~/data/midi

# 5. Start tmux
tmux new -s training

# 6. Run training
./scripts/train_cuda_v2.sh \
    --audio-root ~/data/audio \
    --midi-root ~/data/midi \
    --num-gpus 4 \
    --wandb-key abc123def456

# 7. Detach from tmux: Ctrl+B, then D

# 8. Monitor remotely via W&B dashboard

# 9. When complete, sync results
aws s3 sync ./outputs s3://my-bucket/kmidi-runs/

# 10. Terminate instance (after verifying S3 upload!)
```

---

**Happy Training! 🚀🎵**
