# SageMaker AI Training Setup (JEPA)

One-time setup to run KmiDi JEPA (Audio-JEPA and Chord-JEPA) training on AWS SageMaker.

## Overview

- **Config:** `config/jepa_training.yaml` — training hyperparameters, SageMaker instance type, S3 paths.
- **Launch script:** `scripts/launch_jepa_sagemaker.py` — submits training jobs (requires custom training image and IAM role).
- **Entrypoint:** `scripts/sagemaker_train.py` — runs inside the SageMaker container; reads `/opt/ml/input/config/hyperparameters.json` and data from `/opt/ml/input/data/audio` and `/opt/ml/input/data/midi`, writes artifacts to `/opt/ml/model` (SageMaker uploads to S3).

## One-time setup

### 1. AWS prerequisites

- AWS CLI configured (`aws configure`) with credentials that can create SageMaker training jobs, ECR repos, and S3 buckets.
- Same region used for ECR, S3, and SageMaker (e.g. `us-east-1`).

### 2. IAM role for SageMaker

SageMaker needs an execution role that can read input data from S3 and write model output to S3.

**Option A — Use a custom role (recommended):**

1. In IAM, create a role for **SageMaker** (trust policy: `sagemaker.amazonaws.com`).
2. Attach a policy that allows:
   - `s3:GetObject` on your dataset buckets (e.g. `s3://kmidi-datasets/*`).
   - `s3:PutObject`, `s3:GetObject` on your output bucket (e.g. `s3://kmidi-runs/*`).
   - `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:BatchCheckLayerAvailability` on your ECR repo (or `*` for the account).
3. Note the role ARN (e.g. `arn:aws:iam::123456789012:role/SageMakerExecutionRole`).

**Option B — Use the SageMaker console:**

Create a role from the SageMaker console (Create role from template: “Amazon SageMaker – Execution”) and restrict the default policy to your buckets if desired.

Set the role ARN when launching:

```bash
export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/YourSageMakerRole
```

### 3. S3 buckets

Create (or reuse) two buckets:

| Purpose   | Example bucket     | Config key (in `config/jepa_training.yaml`) |
|----------|--------------------|---------------------------------------------|
| Datasets | `kmidi-datasets`   | `datasets.audio.s3_prefix`, `datasets.midi.s3_prefix` |
| Output   | `kmidi-runs`       | `checkpoints.s3_prefix`                     |

Upload your data:

- **Audio:** Put WAV/MP3/FLAC under e.g. `s3://kmidi-datasets/audio/` (any subfolder structure).
- **MIDI (Chord-JEPA):** Put MIDI files under e.g. `s3://kmidi-datasets/midi/`.

Config example:

```yaml
datasets:
  audio:
    s3_prefix: s3://kmidi-datasets/audio/
  midi:
    s3_prefix: s3://kmidi-datasets/midi/
checkpoints:
  s3_prefix: s3://kmidi-runs/jepa/
```

### 4. Build and push the training image

The launch script uses a **custom training image** (it does not use SageMaker script mode). Build the image from the repo root and push it to ECR.

**4.1 Create ECR repository (if needed):**

```bash
aws ecr create-repository --repository-name kmidi-jepa-train --region us-east-1
```

**4.2 Authenticate Docker to ECR:**

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

Replace `ACCOUNT_ID` with your AWS account ID.

**4.3 Build the image:**

From the **repo root**:

```bash
docker build -f Dockerfile.sagemaker -t kmidi-jepa-train .
```

**4.4 Tag and push:**

```bash
export ECR_URI=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/kmidi-jepa-train:latest
docker tag kmidi-jepa-train:latest $ECR_URI
docker push $ECR_URI
```

**4.5 Set the image URI for the launch script:**

```bash
export KMIDI_SAGEMAKER_IMAGE_URI=$ECR_URI
```

(Or pass `--image-uri $ECR_URI` each time you run the launch script.)

### 5. Launch a training job

From the repo root, with `SAGEMAKER_EXECUTION_ROLE_ARN` and `KMIDI_SAGEMAKER_IMAGE_URI` set (and optional `--config`):

```bash
# Audio-JEPA only
python scripts/launch_jepa_sagemaker.py --model audio_jepa

# Chord-JEPA only
python scripts/launch_jepa_sagemaker.py --model chord_jepa

# Both (two jobs)
python scripts/launch_jepa_sagemaker.py --model both

# Custom config and run ID
python scripts/launch_jepa_sagemaker.py --model audio_jepa --config config/jepa_training.yaml --run-id my-exp-001
```

**Dry run (print config, do not submit):**

```bash
python scripts/launch_jepa_sagemaker.py --model audio_jepa --dry-run
```

### 6. Monitor jobs

- **Console:** SageMaker → Training jobs.
- **CLI:**  
  `aws sagemaker describe-training-job --training-job-name kmidi-audio_jepa-RUN_ID`

Output artifacts are written to the S3 path derived from `checkpoints.s3_prefix` and `run_id` (e.g. `s3://kmidi-runs/jepa/RUN_ID/`).

## Config reference

| Section / key              | Purpose |
|----------------------------|--------|
| `config/jepa_training.yaml` | Main config file. |
| `sagemaker.instance_type`  | e.g. `ml.g5.xlarge` (GPU). |
| `sagemaker.max_run_seconds`| Job timeout (default 86400). |
| `sagemaker.volume_size_gb`| Instance volume size. |
| `training.epochs`          | Training epochs. |
| `training.batch_size`      | Batch size. |
| `training.learning_rate`   | Learning rate. |
| `datasets.audio.s3_prefix` | S3 prefix for audio data. |
| `datasets.midi.s3_prefix` | S3 prefix for MIDI data. |
| `checkpoints.s3_prefix`   | S3 prefix for model output. |

## Optional: local test of the entrypoint

To test the entrypoint script locally (without SageMaker), create a small data layout and point the script at it:

```bash
mkdir -p /opt/ml/input/data/audio /opt/ml/input/config /opt/ml/model
echo '{"model_type":"audio_jepa","epochs":"2","batch_size":"2"}' > /opt/ml/input/config/hyperparameters.json
# Copy a few WAV files into /opt/ml/input/data/audio/
python scripts/sagemaker_train.py
```

Or run inside the same Docker image:

```bash
docker run --rm -v /path/to/local/data:/opt/ml/input/data/audio -v /path/to/config.json:/opt/ml/input/config/hyperparameters.json -v /tmp/model:/opt/ml/model kmidi-jepa-train
```

## Troubleshooting

- **“No audio files found”:** Ensure the S3 channel `audio` (or `midi`) is populated and the role has `s3:GetObject` on that prefix. SageMaker downloads to `/opt/ml/input/data/audio` and `/opt/ml/input/data/midi`.
- **“Access Denied” on S3:** Check the SageMaker execution role policy and bucket permissions.
- **“Image not found” / pull errors:** Ensure the training image is in the same region as the SageMaker job and the role has ECR pull permissions.
- **Out of memory:** Reduce `training.batch_size` in config or use a larger instance (e.g. `ml.g5.2xlarge`).

## Reference

- `config/jepa_training.yaml` — default training and SageMaker settings.
- `scripts/launch_jepa_sagemaker.py` — job submission.
- `scripts/sagemaker_train.py` — container entrypoint.
- `Dockerfile.sagemaker` — training image definition.
- `music_brain.jepa.trainer` — training loops used by the entrypoint.
