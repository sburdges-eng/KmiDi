#!/usr/bin/env python3
"""
Launch JEPA training on Google Cloud Vertex AI.

Usage:
    python scripts/launch_jepa_vertex.py --model audio_jepa
    python scripts/launch_jepa_vertex.py --model chord_jepa
    python scripts/launch_jepa_vertex.py --model emotion_probe
    python scripts/launch_jepa_vertex.py --model both
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
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None


def load_config(path: str) -> dict:
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


def launch_vertex_job(
    model_type: str,
    config: dict,
    image_uri: str,
    project: str,
    location: str,
    staging_bucket: str,
    run_id: str,
    args: argparse.Namespace = None,
) -> str:
    """Submit a Vertex AI training job for the specified JEPA model."""
    if aiplatform is None:
        print(
            "ERROR: google-cloud-aiplatform not installed. Install with: pip install "
            "google-cloud-aiplatform"
        )
        sys.exit(1)

    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

    training_cfg = config.get("training", {})
    vertex_cfg = config.get("vertex_ai", {})

    job_name = f"kmidi-{model_type}-{run_id}"

    # Flatten hyperparameters into command-line args for the container
    worker_args = [
        f"--model_type={model_type}",
        f"--epochs={training_cfg.get('epochs', 120)}",
        f"--batch_size={training_cfg.get('batch_size', 12)}",
        f"--learning_rate={training_cfg.get('learning_rate', 3e-4)}",
        f"--mixed_precision={training_cfg.get('mixed_precision', True)}",
        f"--export_onnx={config.get('checkpoints', {}).get('export_onnx', True)}",
    ]

    model_cfg = config.get(model_type, config.get("audio_jepa", {}))
    for k, v in model_cfg.items():
        worker_args.append(f"--model_{k}={v}")

    datasets_cfg = config.get("datasets", {})
    # For Vertex, we pass GCS paths. We assume the config has 'gcs_prefix' for each channel.
    if model_type == "audio_jepa":
        audio_gcs = datasets_cfg.get("audio", {}).get("gcs_prefix", "")
        if audio_gcs:
            worker_args.append(f"--audio_dir={audio_gcs}")
    elif model_type == "chord_jepa":
        midi_gcs = datasets_cfg.get("midi", {}).get("gcs_prefix", "")
        if midi_gcs:
            worker_args.append(f"--midi_dir={midi_gcs}")

    if hasattr(args, "embeddings_path") and args.embeddings_path and model_type == "emotion_probe":
        worker_args.append(f"--embeddings_path={args.embeddings_path}")

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=image_uri,
        staging_bucket=staging_bucket,
    )

    # Launch the job
    job.run(
        args=worker_args,
        replica_count=1,
        machine_type=vertex_cfg.get("machine_type", "n1-standard-8"),
        accelerator_type=vertex_cfg.get("accelerator_type", "NVIDIA_TESLA_V100"),
        accelerator_count=vertex_cfg.get("accelerator_count", 1),
        sync=False,  # Launch and return
    )

    print(f"Launched Vertex AI job: {job_name}")
    return job_name


def main():
    parser = argparse.ArgumentParser(description="Launch JEPA training on Vertex AI")
    parser.add_argument(
        "--model",
        choices=["audio_jepa", "chord_jepa", "emotion_probe", "both"],
        default="audio_jepa",
        help="Which JEPA model to train",
    )
    parser.add_argument(
        "--embeddings-path",
        default="",
        help="GCS paths to cached .pt embeddings (e.g., gs://bucket/jepa_emotion_embeddings.pt). "
        "Required for emotion_probe.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_REPO_ROOT, "config", "jepa_training.yaml"),
        help="Path to JEPA training config",
    )
    parser.add_argument(
        "--image-uri",
        default=os.environ.get("KMIDI_VERTEX_IMAGE_URI", ""),
        help="Artifact Registry image URI for training container",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT_ID", ""),
        help="Google Cloud Project ID",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Vertex AI location",
    )
    parser.add_argument(
        "--staging-bucket",
        default=os.environ.get("GCP_STAGING_BUCKET", ""),
        help="GCS bucket for staging artifacts",
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
        print("ERROR: --image-uri or KMIDI_VERTEX_IMAGE_URI required")
        sys.exit(1)
    if not args.project:
        print("ERROR: --project or GCP_PROJECT_ID required")
        sys.exit(1)
    if not args.staging_bucket:
        print("ERROR: --staging-bucket or GCP_STAGING_BUCKET required")
        sys.exit(1)

    models = ["audio_jepa", "chord_jepa"] if args.model == "both" else [args.model]

    for model_type in models:
        launch_vertex_job(
            model_type=model_type,
            config=config,
            image_uri=args.image_uri,
            project=args.project,
            location=args.location,
            staging_bucket=args.staging_bucket,
            run_id=args.run_id,
            args=args,
        )


if __name__ == "__main__":
    main()
