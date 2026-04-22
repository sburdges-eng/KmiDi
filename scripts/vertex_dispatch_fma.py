#!/usr/bin/env python3
"""Dispatch FMA training jobs to Vertex AI (custom container jobs).

Uses a prebuilt PyTorch container (`torch-2-x-cu118.py310`) + a small
`pip_install` list for torchaudio/tensorboard. The repo is packaged as
a Python module via Vertex's `package_uris`, so we upload a source
tarball to GCS and point the job at it.

Usage::

    # Preflight — dry run, no job submitted
    python scripts/vertex_dispatch_fma.py --task genre --dry-run

    # Dispatch genre baseline
    python scripts/vertex_dispatch_fma.py --task genre

    # Dispatch subgenre / tags / jepa (once those trainers exist)
    python scripts/vertex_dispatch_fma.py --task subgenre
    python scripts/vertex_dispatch_fma.py --task tags
    python scripts/vertex_dispatch_fma.py --task jepa
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("vertex_dispatch_fma")

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_CONFIGS = {
    "genre": {
        "script": "training/scripts/train_fma_genre.py",
        "extra_args": ["--epochs", "30", "--batch-size", "32",
                       "--patience", "6", "--vertex"],
        "machine_type": "n1-standard-4",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    },
    # Sub / tags / jepa wired when their trainers land.
}


SETUP_PY = '''\
from setuptools import setup, find_packages

setup(
    name="kmidi_vertex_trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "tqdm",
        "tensorboard",
        "PyYAML",
        "librosa",
        "soundfile",
        "numpy",
    ],
)
'''


def build_source_tarball() -> Path:
    """Build a proper sdist with setup.py so Vertex `pip install`s the
    package + its deps inside the training container."""
    tmp = Path(tempfile.mkdtemp(prefix="kmidi-vertex-"))
    pkg = tmp / "kmidi_vertex_trainer"
    pkg.mkdir()
    (pkg / "setup.py").write_text(SETUP_PY)

    includes = [
        "training/__init__.py",
        "training/scripts/__init__.py",
        "training/scripts/train_fma_genre.py",
        "training/src/__init__.py",
        "training/src/models/__init__.py",
        "training/src/models/audio_classifier.py",
        "music_brain/__init__.py",
        "music_brain/penta_core/__init__.py",
        "music_brain/penta_core/ml/__init__.py",
        "music_brain/penta_core/ml/audio_dataset.py",
    ]
    for inc in includes:
        src = REPO_ROOT / inc
        if not src.exists():
            logger.warning("Missing include: %s", inc)
            continue
        dst = pkg / inc
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Build sdist via `python setup.py sdist`
    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(tmp)],
        cwd=str(pkg), check=True, capture_output=True)
    dist_files = list(tmp.glob("*.tar.gz"))
    if not dist_files:
        raise RuntimeError("setup.py sdist produced no .tar.gz")
    out = dist_files[0]
    logger.info("Built sdist: %s (%.1f KB)", out, out.stat().st_size / 1024)
    return out


def upload_tarball(tar_path: Path, bucket: str) -> str:
    """Upload tarball to gs://bucket/vertex-packages/ and return gs:// URI."""
    uri = f"gs://{bucket}/vertex-packages/{tar_path.name}"
    subprocess.run(
        ["gcloud", "storage", "cp", str(tar_path), uri],
        check=True, capture_output=True)
    logger.info("Uploaded package → %s", uri)
    return uri


def submit_job(args: argparse.Namespace, package_uri: str, run_name: str) -> None:
    from google.cloud import aiplatform

    cfg = TASK_CONFIGS[args.task]
    aiplatform.init(
        project=args.project, location=args.region,
        staging_bucket=f"gs://{args.bucket}",
    )

    # Remap /Volumes paths to /gcs/<bucket>/fma/audio/
    gcs_audio_prefix = f"/gcs/{args.bucket}/fma/audio/fma_medium"
    manifest_gcs = f"/gcs/{args.bucket}/fma/manifest/fma_medium_manifest.csv"

    script_args = [
        "--manifest", manifest_gcs,
        "--gcs-audio-prefix", gcs_audio_prefix,
        "--name", run_name,
        "--num-workers", "4",
    ] + cfg["extra_args"]

    # Output dir: AIP_MODEL_DIR is auto-set by Vertex when
    # base_output_dir is provided. Put it under gs://bucket/runs/<run>/model/
    base_output_dir = f"gs://{args.bucket}/runs/{run_name}"

    # "python_package_spec" expects a packaged .tar.gz uploaded to GCS.
    # module name is the dotted path to the train script without .py.
    module_name = cfg["script"].replace("/", ".").replace(".py", "")

    job = aiplatform.CustomJob(
        display_name=run_name,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": cfg["machine_type"],
                "accelerator_type": cfg["accelerator_type"],
                "accelerator_count": cfg["accelerator_count"],
            },
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri":
                    "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest",
                "package_uris": [package_uri],
                "python_module": module_name,
                "args": script_args,
            },
        }],
        base_output_dir=base_output_dir,
    )

    logger.info("Submitting %s | machine=%s+%s×%d | module=%s",
                run_name, cfg["machine_type"], cfg["accelerator_type"],
                cfg["accelerator_count"], module_name)
    logger.info("Output dir: %s", base_output_dir)

    if args.dry_run:
        logger.info("--dry-run: skipping submit.")
        return

    job.submit(
        service_account=args.service_account,
        enable_web_access=True,
    )
    logger.info("Submitted. Vertex console:")
    logger.info("  https://console.cloud.google.com/vertex-ai/locations/%s/training/%s?project=%s",
                args.region, job.resource_name.split("/")[-1], args.project)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=list(TASK_CONFIGS.keys()), required=True)
    ap.add_argument("--project", default="devvy-490312")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", default="kmidi-train-us-central1")
    ap.add_argument("--service-account", default=None,
                    help="Optional SA email. Default uses Compute Engine default SA.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--name", default=None,
                    help="Run name. Default: <task>-YYYYMMDD-HHMMSS")
    args = ap.parse_args()

    run_name = args.name or f"fma-{args.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if args.task not in TASK_CONFIGS:
        logger.error("Task %s not yet wired.", args.task)
        return 2

    # 1. Package the source tree
    tar = build_source_tarball()
    try:
        # 2. Upload
        pkg_uri = upload_tarball(tar, args.bucket)
        # 3. Submit
        submit_job(args, pkg_uri, run_name)
    finally:
        shutil.rmtree(tar.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
