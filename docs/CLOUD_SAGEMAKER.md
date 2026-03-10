# Connecting KmiDi to AWS SageMaker

This doc describes how to run KmiDi training as **SageMaker training jobs** (managed GPU, same entrypoint as EC2, S3 in/out). For one-off EC2 GPU launches see [config/env.ec2-s3.example](../config/env.ec2-s3.example) and `./scripts/launch_aws_cloud_training.sh`.

## Overview


| Aspect            | EC2 (launch_aws_cloud_training.sh)                      | SageMaker                                                   |
| ----------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| **Orchestration** | You launch an instance; user-data runs training in tmux | SageMaker runs a training job on managed instances          |
| **Input**         | S3 package URI (or pre-cached local on instance)        | S3 channel `package`                                        |
| **Output**        | S3 run bucket/prefix (upload from instance)             | S3 output path (uploaded by SageMaker from `/opt/ml/model`) |
| **Entrypoint**    | `scripts/aws_train_entrypoint.py`                       | Same; run inside a custom container                         |
| **Config**        | `config/run_contract.yaml`                              | Same; baked into image or passed via hyperparameters        |


Use SageMaker when you want managed lifecycle, built-in spot support, and no SSH/tmux. Use EC2 when you want full control and existing scripts (e.g. tmux, watchdog, cost script).

## Prerequisites

1. **IAM**
  - SageMaker **execution role** with: `AmazonSageMakerFullAccess` (or minimal: sagemaker:CreateTrainingJob, s3, ecr, logs).  
  - Trust: `sagemaker.amazonaws.com`.
2. **S3**
  - Same as EC2: a **package** location (packaged dataset) and a **run** location (artifacts).  
  - Can reuse buckets from [run_contract.yaml](../config/run_contract.yaml) (`s3.packageBucket`, `s3.runBucket`).
3. **Training image**
  - A Docker image that runs `scripts/aws_train_entrypoint.py` with inputs under `/opt/ml/input/data/package` and uploads outputs to the job’s S3 output path (or to a run bucket you pass via hyperparameters).  
  - Build from the repo (e.g. a `Dockerfile.train` that copies scripts + config and sets the entrypoint).  
  - Push to **ECR** in the same account/region as the SageMaker job.

## Console (SageMaker Studio)

**Studio domain (us-east-1):** [SageMaker Studio → Resources](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/studio/d-89omtqmujcxk?tab=resources) — use this to view runs, jobs, and resources in your domain. Training jobs submitted via `launch_sagemaker_training.py` appear under **Training** in the SageMaker console (same region).

## Env and config

1. Copy and fill the example env:
  ```bash
   cp config/env.sagemaker.example .env.sagemaker
   # edit .env.sagemaker: SAGEMAKER_EXECUTION_ROLE_ARN, S3 buckets, image URI
  ```
2. Ensure `config/run_contract.yaml` has the correct `s3.packageBucket`, `s3.runBucket`, and `training.activePackageId` (or pass package/run overrides when launching).

## Launching a training job

From repo root, with `.env.sagemaker` sourced (or the same vars in the environment):

```bash
source .env.sagemaker   # or export the vars manually
./scripts/launch_sagemaker_training.py \
  --image-uri "$KMIDI_SAGEMAKER_IMAGE_URI" \
  --package-s3-uri "s3://your-package-bucket/training/packages/your-package-id" \
  --output-s3-uri "s3://your-run-bucket/training/runs" \
  --run-id "run-$(date +%Y%m%d-%H%M)" \
  --instance-type "ml.g5.xlarge" \
  --max-runtime-seconds 21600
```

Optional: `--run-contract`, `--region`, `--profile`, `--dry-run`. The script uses **boto3** only (no SageMaker SDK dependency).

## Container contract

The image used for `--image-uri` must:

1. **Input data**
  SageMaker will mount the “package” channel at `/opt/ml/input/data/package`. The container must run the KmiDi entrypoint with a path to that directory (e.g. `--package-local-dir /opt/ml/input/data/package`).
2. **Output**
  The entrypoint should write artifacts to the S3 output location. Pass `--output-s3-uri` via hyperparameters (the launch script does this). Optionally write model artifacts under `/opt/ml/model` so SageMaker uploads them to the job’s output S3 path.
3. **Entrypoint**
  Run `python3 scripts/aws_train_entrypoint.py` (or equivalent) with at least:
  - `--package-local-dir /opt/ml/input/data/package`
  - `--output-s3-uri <from hyperparameters>`
  - `--run-id <from hyperparameters>`
  - `--workdir /opt/ml/output` (or similar)
  - `--run-contract` path if the contract is in the image.

The launch script passes these as SageMaker hyperparameters. The repo’s **scripts/sagemaker_entrypoint.py** reads `/opt/ml/input/config/hyperparameters.json` and invokes `aws_train_entrypoint.py` with the correct args; it is used by **Dockerfile.train**.

## Training image (Dockerfile.train)

The repo includes a **Dockerfile.train** and **scripts/sagemaker_entrypoint.py** so you can build the training image without extra wiring. The entrypoint reads SageMaker hyperparameters from `/opt/ml/input/config/hyperparameters.json` and runs `aws_train_entrypoint.py` with `--package-local-dir /opt/ml/input/data/package`, `--output-s3-uri`, `--run-id`, and `--workdir`.

**If you already have an ECR repo (your domain):** build, tag to your repo URI, and push. Then use that URI as `--image-uri` when calling `launch_sagemaker_training.py`.

From repo root:

```bash
# Build
docker build -f Dockerfile.train -t kmidi-train:latest .

# Log in to your ECR (replace <account>, <region>, <your-repo> with your domain/repo)
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

# Tag and push (use your existing ECR repo name)
docker tag kmidi-train:latest <account>.dkr.ecr.<region>.amazonaws.com/<your-repo>/kmidi-train:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/<your-repo>/kmidi-train:latest
```

Set `KMIDI_SAGEMAKER_IMAGE_URI` to the full ECR URI (e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-company/kmidi-train:latest`) in `.env.sagemaker` or pass `--image-uri` to the launch script.

## References

- [config/run_contract.yaml](../config/run_contract.yaml) — S3 buckets, training defaults, early stop.
- [config/env.ec2-s3.example](../config/env.ec2-s3.example) — EC2 + S3 env (same S3 concepts).
- [scripts/aws_train_entrypoint.py](../scripts/aws_train_entrypoint.py) — Training entrypoint (EC2 and SageMaker).
- [docs/ml/CLOUD_TRAINING_GUIDE.md](ml/CLOUD_TRAINING_GUIDE.md) — Lambda, RunPod, SSH, Docker options.

