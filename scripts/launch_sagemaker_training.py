#!/usr/bin/env python3
"""Submit a KmiDi training job to AWS SageMaker (custom container, S3 in/out)."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Run from repo root so scripts.* and config are resolvable when needed
REPO_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_job_name(name: str) -> str:
    """SageMaker job name: alphanumeric and hyphen only, max 63 chars."""
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", name)
    return sanitized[:63].strip("-") or "kmidi-train"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit KmiDi training as a SageMaker training job (custom container)."
    )
    parser.add_argument(
        "--image-uri",
        default=os.environ.get("KMIDI_SAGEMAKER_IMAGE_URI", ""),
        help="ECR image URI for training container (or set KMIDI_SAGEMAKER_IMAGE_URI).",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", ""),
        help="SageMaker execution role ARN (or set SAGEMAKER_EXECUTION_ROLE_ARN).",
    )
    parser.add_argument(
        "--package-s3-uri",
        default=os.environ.get("KMIDI_PACKAGE_S3_URI", ""),
        help="S3 URI for packaged dataset (or set KMIDI_PACKAGE_S3_URI).",
    )
    parser.add_argument(
        "--output-s3-uri",
        default=os.environ.get("KMIDI_OUTPUT_S3_URI", ""),
        help="S3 prefix for job output (or set KMIDI_OUTPUT_S3_URI).",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("KMIDI_RUN_ID", ""),
        help="Run ID (used for job name and hyperparameters).",
    )
    parser.add_argument(
        "--instance-type",
        default=os.environ.get("KMIDI_SAGEMAKER_INSTANCE_TYPE", "ml.g5.xlarge"),
        help="SageMaker instance type (default: ml.g5.xlarge).",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=int(os.environ.get("KMIDI_SAGEMAKER_MAX_RUNTIME_SECONDS", "21600")),
        help="Max runtime in seconds (default: 21600 = 6h).",
    )
    parser.add_argument(
        "--volume-size-gb",
        type=int,
        default=100,
        help="EBS volume size in GB for the training instance (default: 100).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS region.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE", ""),
        help="AWS profile (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job payload and exit without submitting.",
    )
    return parser.parse_args()


def _resolve_uris(args: argparse.Namespace) -> tuple[str, str]:
    package_uri = (args.package_s3_uri or "").strip()
    output_uri = (args.output_s3_uri or "").strip()
    if not package_uri or not output_uri:
        bucket = os.environ.get("KMIDI_PACKAGE_BUCKET", "")
        run_bucket = os.environ.get("KMIDI_RUN_BUCKET", "")
        pkg_id = os.environ.get("KMIDI_PACKAGE_ID", "")
        if not package_uri and bucket and pkg_id:
            package_uri = f"s3://{bucket}/training/packages/{pkg_id}"
        if not output_uri and run_bucket:
            output_uri = f"s3://{run_bucket}/training/runs"
    return package_uri, output_uri


def main() -> int:
    args = _parse_args()
    package_uri, output_uri = _resolve_uris(args)

    if not args.image_uri:
        print("Missing --image-uri (or KMIDI_SAGEMAKER_IMAGE_URI).", file=sys.stderr)
        return 1
    if not args.role_arn:
        print("Missing --role-arn (or SAGEMAKER_EXECUTION_ROLE_ARN).", file=sys.stderr)
        return 1
    if not package_uri:
        print(
            "Missing --package-s3-uri (or KMIDI_PACKAGE_S3_URI / bucket+package id).",
            file=sys.stderr,
        )
        return 1
    if not output_uri:
        print(
            "Missing --output-s3-uri (or KMIDI_OUTPUT_S3_URI / KMIDI_RUN_BUCKET).",
            file=sys.stderr,
        )
        return 1

    run_id = (args.run_id or "run-sagemaker").strip()
    job_name = _sanitize_job_name(run_id)

    # HyperParameters: container reads these (e.g. from /opt/ml/input/config/hyperparameters.json)
    # and must run aws_train_entrypoint.py with --package-local-dir /opt/ml/input/data/package
    # and --output-s3-uri <output_s3_uri> --run-id <run_id> --workdir /opt/ml/output
    hyperparameters = {
        "run_id": run_id,
        "output_s3_uri": output_uri.rstrip("/"),
        "workdir": "/opt/ml/output",
    }

    payload = {
        "TrainingJobName": job_name,
        "RoleArn": args.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": args.image_uri,
            "TrainingInputMode": "File",
        },
        "InputDataConfig": [
            {
                "ChannelName": "package",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": package_uri.rstrip("/") + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/json",
                "CompressionType": "None",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": output_uri.rstrip("/") + "/",
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": args.volume_size_gb,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": args.max_runtime_seconds,
        },
        "HyperParameters": {k: str(v) for k, v in hyperparameters.items()},
    }

    if args.dry_run:
        import json

        print(json.dumps(payload, indent=2))
        return 0

    try:
        import boto3
    except ImportError:
        print("boto3 is required. Install with: pip install boto3", file=sys.stderr)
        return 1

    session = boto3.Session(region_name=args.region, profile_name=args.profile or None)
    client = session.client("sagemaker")
    client.create_training_job(**payload)
    print(f"Submitted SageMaker training job: {job_name}")
    print(f"  Package: {package_uri}")
    print(f"  Output:  {output_uri}")
    console_url = (
        f"https://{args.region}.console.aws.amazon.com/sagemaker/home"
        f"?region={args.region}#/jobs/{job_name}"
    )
    print(f"  Console: {console_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
