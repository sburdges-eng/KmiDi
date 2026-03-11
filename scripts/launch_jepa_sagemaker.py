#!/usr/bin/env python3
"""
Launch JEPA training on AWS SageMaker.

Usage:
    python scripts/launch_jepa_sagemaker.py --model audio_jepa
    python scripts/launch_jepa_sagemaker.py --model chord_jepa
    python scripts/launch_jepa_sagemaker.py --model both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import yaml
except ImportError:
    yaml = None

try:
    import boto3
except ImportError:
    boto3 = None


def load_config(path: str) -> dict:
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


def launch_sagemaker_job(
    model_type: str,
    config: dict,
    image_uri: str,
    role_arn: str,
    run_id: str,
) -> str:
    """Submit a SageMaker training job for the specified JEPA model."""
    if boto3 is None:
        print("ERROR: boto3 not installed. Install with: pip install boto3")
        sys.exit(1)

    sm = boto3.client("sagemaker")
    training_cfg = config.get("training", {})
    sagemaker_cfg = config.get("sagemaker", {})

    job_name = f"kmidi-{model_type}-{run_id}"

    hyperparameters = {
        "model_type": model_type,
        "epochs": str(training_cfg.get("epochs", 120)),
        "batch_size": str(training_cfg.get("batch_size", 12)),
        "learning_rate": str(training_cfg.get("learning_rate", 3e-4)),
        "mixed_precision": str(training_cfg.get("mixed_precision", True)),
        "export_onnx": str(config.get("checkpoints", {}).get("export_onnx", True)),
    }

    model_cfg = config.get(model_type, config.get("audio_jepa", {}))
    for k, v in model_cfg.items():
        hyperparameters[f"model_{k}"] = str(v)

    datasets_cfg = config.get("datasets", {})
    input_channels = []
    for channel_name in ("audio", "midi"):
        channel = datasets_cfg.get(channel_name, {})
        s3_uri = channel.get("s3_prefix")
        if s3_uri:
            input_channels.append({
                "ChannelName": channel_name,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            })

    checkpoint_s3 = config.get("checkpoints", {}).get("s3_prefix", "")
    output_s3 = f"{checkpoint_s3}{run_id}/" if checkpoint_s3 else None

    request = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "RoleArn": role_arn,
        "HyperParameters": hyperparameters,
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": sagemaker_cfg.get("instance_type", "ml.g5.xlarge"),
            "VolumeSizeInGB": sagemaker_cfg.get("volume_size_gb", 100),
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": sagemaker_cfg.get("max_run_seconds", 86400),
        },
    }

    if input_channels:
        request["InputDataConfig"] = input_channels
    if output_s3:
        request["OutputDataConfig"] = {"S3OutputPath": output_s3}

    sm.create_training_job(**request)
    print(f"Launched SageMaker job: {job_name}")
    return job_name


def main():
    parser = argparse.ArgumentParser(
        description="Launch JEPA training on SageMaker"
    )
    parser.add_argument(
        "--model",
        choices=["audio_jepa", "chord_jepa", "both"],
        default="audio_jepa",
        help="Which JEPA model to train",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_REPO_ROOT, "config", "jepa_training.yaml"),
        help="Path to JEPA training config",
    )
    parser.add_argument(
        "--image-uri",
        default=os.environ.get("KMIDI_SAGEMAKER_IMAGE_URI", ""),
        help="ECR image URI for training container",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", ""),
        help="IAM role ARN for SageMaker",
    )
    parser.add_argument(
        "--run-id",
        default=f"jepa-{int(time.time())}",
        help="Unique run identifier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without launching",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    if not args.image_uri:
        print("ERROR: --image-uri or KMIDI_SAGEMAKER_IMAGE_URI required")
        sys.exit(1)
    if not args.role_arn:
        print("ERROR: --role-arn or SAGEMAKER_EXECUTION_ROLE_ARN required")
        sys.exit(1)

    models = (
        ["audio_jepa", "chord_jepa"]
        if args.model == "both"
        else [args.model]
    )

    for model_type in models:
        launch_sagemaker_job(
            model_type=model_type,
            config=config,
            image_uri=args.image_uri,
            role_arn=args.role_arn,
            run_id=args.run_id,
        )


if __name__ == "__main__":
    main()
