# Cloud dev environment setup

One-time setup for cloud training (Lambda Labs, SageMaker, EC2/S3, SSH). Run from repo root.

**Start with local training:** If you have a 16 GB Mac or want a free option (Colab/Kaggle), run local training first: [docs/LOCAL_TRAINING_SETUP.md](LOCAL_TRAINING_SETUP.md). Then use this doc for cloud.

## Prerequisites

- **Base dev:** Run `./scripts/dev-setup.sh` first (bootstrap, npm, pip).
- **SSH:** For Lambda/SSH workflows, `ssh` and key-based auth to the GPU host.
- **AWS (optional):** For SageMaker or EC2/S3: AWS CLI configured (`aws configure`); IAM and S3/ECR as in [SAGEMAKER_SETUP.md](SAGEMAKER_SETUP.md) or [launch_aws_cloud_training.sh](../scripts/launch_aws_cloud_training.sh).
- **Docker (optional):** For local GPU runs via `run_from_workspace.sh docker`.

## One-time setup

```bash
./scripts/dev-setup-cloud.sh
```

Or with base dev included:

```bash
./scripts/dev-setup-cloud.sh --full
```

This creates `env/`, `cloud_training/logs`, and (if missing) `env/.env.training` from `env/.env.training.example`. Then load training env in your shell:

```bash
source scripts/load-env.sh training
```

## Env by scenario

### Lambda / SSH

- Set `LAMBDA_SSH=ubuntu@<ip>` (from Lambda dashboard) in your shell or in `env/.env.training`.
- Optional: `CHECKPOINT_PATH`, `RANK`, `WORLD_SIZE`, `WANDB_*` in `env/.env.training` (see [ENVIRONMENT.md](ENVIRONMENT.md)).

### SageMaker

- Use `config/env.sagemaker.example` and `.env.sagemaker` in repo root (not `env/.env.training`).
- Full steps: [SAGEMAKER_SETUP.md](SAGEMAKER_SETUP.md).

### EC2/S3

- Use `config/env.ec2-s3.example` and `.env.ec2-s3` in repo root.
- Launch: [scripts/launch_aws_cloud_training.sh](../scripts/launch_aws_cloud_training.sh).

## Run and monitor

- **Run:** [docs/ml/CLOUD_TRAINING_GUIDE.md](ml/CLOUD_TRAINING_GUIDE.md) — `./scripts/cloud_training/run_from_workspace.sh` (lambda | ssh | attach | docker).
- **Status and logs:** `cloud_training/status.json`, `cloud_training/logs/training_latest.log`. The AI uses these paths to report progress (see `.cursor/rules/cloud-training-monitoring.mdc`).
