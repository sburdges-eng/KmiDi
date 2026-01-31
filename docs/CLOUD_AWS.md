# Cloud Training — AWS EC2

**Quick reference:** [CLOUD_SETUP_REFERENCE.md](CLOUD_SETUP_REFERENCE.md) | **Generic:** [CLOUD_TRAINING.md](CLOUD_TRAINING.md)

Use AWS EC2 GPU instances for KmiDi training. Same scripts as Lambda/RunPod; AWS adds instance provisioning and S3 sync.

## Instance types (GPU)

| Type | GPU | VRAM | Use case |
|------|-----|------|----------|
| **g4dn.xlarge** | T4 | 16 GB | JEPA, emotion; cheapest |
| **g5.xlarge** | A10G | 24 GB | Larger batch |
| **p3.2xlarge** | V100 | 16 GB | Production |
| **p4d.24xlarge** | A100 x8 | 320 GB | Distributed |

## One-time provisioning

### 1. Launch EC2 instance (Console or CLI)

```bash
# Example: g4dn.xlarge with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=kmidi-train}]'
```

**AMI:** Use *Deep Learning AMI GPU* (Ubuntu) or *Ubuntu 22.04* with [NVIDIA driver + Docker](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html).

**EBS:** Attach a 100+ GB gp3 volume. We'll mount it at `/workspace`.

### 2. Attach and mount EBS at /workspace

After instance boots, attach the volume (if not auto-attached) and on the instance:

```bash
# One-time: format and mount (replace nvme1n1 with your block device)
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir -p /workspace
sudo mount /dev/nvme1n1 /workspace
echo '/dev/nvme1n1 /workspace ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo chown $USER:$USER /workspace
```

**Deep Learning AMI:** May use different block device names; check `lsblk`.

### 3. SSH with key

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-public-ip>
# Or ec2-user@ for Amazon Linux AMI
```

## Training setup (on the instance)

Use the standard cloud scripts once you're SSH'd in.

```bash
cd /workspace
git clone https://github.com/sburdges-eng/KmiDi repos/kmidi
cd /workspace/repos/kmidi

# One-time: build docker image + prepare /workspace layout
./scripts/cloud_setup.sh

# Launch training in tmux (default config: /workspace/config.yaml)
./scripts/cloud_train.sh
```

Notes:
- `cloud_setup.sh` expects `/workspace` to be a persistent volume.
- `cloud_train.sh` mounts `/workspace/datasets` (read-only) and writes to `/workspace/checkpoints` and `/workspace/logs`.
- Edit `/workspace/config.yaml` to choose model/config (e.g. `configs/cloud_training.yaml`).

## Launch and monitor from Mac

```bash
# Use your .pem key for AWS
export CLOUD_SSH_HOST=ubuntu@<ec2-public-ip>
export CLOUD_SSH_KEY=~/.ssh/your-key.pem

./scripts/cloud_launch_and_monitor.sh both
```

## S3 checkpoint sync

Push checkpoints to S3 after training (or periodically). On the instance:

```bash
aws s3 sync /workspace/checkpoints s3://your-bucket/kmidi/checkpoints/ --exclude "*.log"
aws s3 sync /workspace/logs s3://your-bucket/kmidi/logs/
```

Or add a cron job or post-training hook. Ensure the instance has IAM role with `s3:PutObject` (or use `aws configure`).

## Security group

Allow:

- **22** (SSH) from your IP
- **80/443** if serving (optional)

## Cost notes

- **g4dn.xlarge:** ~$0.526/hr on-demand
- Use Spot for ~70% savings (interruption risk)
- Stop instance when not training; EBS persists
