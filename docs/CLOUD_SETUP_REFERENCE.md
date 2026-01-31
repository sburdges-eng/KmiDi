# Cloud Setup — Quick Reference

**Full doc:** [CLOUD_TRAINING.md](CLOUD_TRAINING.md) | **AWS:** [CLOUD_AWS.md](CLOUD_AWS.md)

## From Mac (launch and monitor)

```bash
# Set your cloud SSH target
export CLOUD_SSH_HOST=ubuntu@<cloud-ip>

# AWS: add SSH key for .pem
export CLOUD_SSH_KEY=~/.ssh/your-key.pem

# One-shot: setup + train + attach to monitor
./scripts/cloud_launch_and_monitor.sh both

# Or stepwise:
./scripts/cloud_launch_and_monitor.sh setup    # one-time
./scripts/cloud_launch_and_monitor.sh train    # launch
./scripts/cloud_launch_and_monitor.sh monitor  # attach (Ctrl+B D to detach)
```

## On Cloud Instance (manual)

```bash
# 1. One-time setup
./scripts/cloud_setup.sh https://github.com/sburdges-eng/KmiDi

# 2. Launch training
./scripts/cloud_train.sh

# 3. Attach to session
tmux attach -t kmidi-train   # Ctrl+B D to detach
```

**Before setup:** SSH in, mount persistent volume at `/workspace`, run `nvidia-smi`.

**Artifacts:** Logs → `/workspace/logs/`, Checkpoints → `/workspace/checkpoints/`

**AWS:** See [CLOUD_AWS.md](CLOUD_AWS.md) for EC2 instance types, EBS mount, S3 sync.
