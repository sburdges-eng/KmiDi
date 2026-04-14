#!/usr/bin/env python3
"""
Launch NLM/LLM fine-tuning on Google Cloud Vertex AI.

Usage:
    python scripts/launch_nlm_vertex.py \
      --base-model meta-llama/Llama-3-8B-Instruct \
      --dataset-path gs://my-bucket/training_data/instruct.jsonl \
      --epochs 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None


def launch_nlm_vertex_job(args: argparse.Namespace) -> str:
    """Submit a Vertex AI training job for the NLM LLM."""
    if aiplatform is None:
        print("ERROR: google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(project=args.project, location=args.location, staging_bucket=args.staging_bucket)

    job_name = f"kmidi-nlm-{args.run_id}"

    # Flatten hyperparameters into command-line args for the container
    worker_args = [
        f"--base_model={args.base_model}",
        f"--dataset_path={args.dataset_path}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--learning_rate={args.learning_rate}",
        f"--lora_rank={args.lora_rank}",
        f"--lora_alpha={args.lora_alpha}",
        f"--max_seq_length={args.max_seq_length}",
    ]

    # Add HF Token if provided
    if args.hf_token:
        worker_args.append(f"--hf_token={args.hf_token}")

    # Set up the Base Output Directory where Vertex AI will write model artifacts
    base_output_dir = f"{args.staging_bucket.rstrip('/')}/{job_name}/model_output"

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=args.image_uri,
        staging_bucket=args.staging_bucket,
        model_serving_container_image_uri=None,  # We are just exporting the adapters, not serving directly here
    )

    # Launch the job - mapping output to AIP_MODEL_DIR via Vertex AI base_output_dir parameter
    job.run(
        args=worker_args,
        base_output_dir=base_output_dir,
        replica_count=1,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        sync=args.sync, 
    )

    print(f"Launched Vertex AI NLM job: {job_name}")
    print(f"Artifacts will be available at: {base_output_dir}")
    return job_name


def main():
    parser = argparse.ArgumentParser(description="Launch NLM fine-tuning on Vertex AI")

    # Required job args
    parser.add_argument("--base-model", required=True, help="HF model ID or GCS gs:// path to base weights")
    parser.add_argument("--dataset-path", required=True, help="HF dataset ID or GCS gs:// path to JSONL")
    
    # GCP Infrastructure args
    parser.add_argument("--image-uri", default=os.environ.get("KMIDI_NLM_VERTEX_IMAGE_URI", ""), help="Artifact Registry image URI for training container")
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID", ""), help="Google Cloud Project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--staging-bucket", default=os.environ.get("GCP_STAGING_BUCKET", ""), help="GCS bucket for staging artifacts (must start with gs://)")
    parser.add_argument("--run-id", default=f"finetune-{int(time.time())}", help="Unique run identifier")
    
    # Hardware args
    parser.add_argument("--machine-type", default="g2-standard-4", help="GCP Machine type (default: g2-standard-4 for L4)")
    parser.add_argument("--accelerator-type", default="NVIDIA_L4", help="GCP Accelerator type (default: NVIDIA_L4)")
    parser.add_argument("--accelerator-count", type=int, default=1, help="Number of GPUs")
    
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    
    # Credentials
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face Token")
    
    parser.add_argument("--dry-run", action="store_true", help="Print config without launching")
    parser.add_argument("--sync", action="store_true", help="Wait for job to finish before exiting script")

    args = parser.parse_args()

    if args.dry_run:
        args_dict = vars(args)
        for k, v in args_dict.items():
            print(f"{k}: {v}")
        return

    if not args.image_uri:
        print("ERROR: --image-uri or KMIDI_NLM_VERTEX_IMAGE_URI required")
        sys.exit(1)
    if not args.project:
        print("ERROR: --project or GCP_PROJECT_ID required")
        sys.exit(1)
    if not args.staging_bucket or not args.staging_bucket.startswith("gs://"):
        print("ERROR: --staging-bucket or GCP_STAGING_BUCKET required and must start with gs://")
        sys.exit(1)

    launch_nlm_vertex_job(args)


if __name__ == "__main__":
    main()
