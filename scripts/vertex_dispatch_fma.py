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
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        # Match the container's torch 2.3.x (pytorch-gpu.2-3.py310 image).
        "torchaudio==2.3.1",
        "librosa",      # mp3 decode via audioread fallbacks (broader than torchaudio.load)
        "soundfile",
        "audioread",
        "pandas",
        "tqdm",
        "tensorboard",
        "PyYAML",
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
    """Submit via `gcloud ai custom-jobs create` with a YAML config — the
    CLI accepts spot/preemptible GPUs (which the high-level aiplatform SDK
    doesn't expose cleanly) and supports baseOutputDirectory via --config."""
    import yaml
    cfg = TASK_CONFIGS[args.task]

    gcs_audio_prefix = f"/gcs/{args.bucket}/fma/audio/fma_medium"
    manifest_gcs = f"/gcs/{args.bucket}/fma/manifest/{args.manifest_name}"

    script_args = [
        "--manifest", manifest_gcs,
        "--gcs-audio-prefix", gcs_audio_prefix,
        "--name", run_name,
        "--num-workers", "4",
    ] + cfg["extra_args"]

    base_output_dir = f"gs://{args.bucket}/runs/{run_name}"
    module_name = cfg["script"].replace("/", ".").replace(".py", "")

    # Spec structure matches CustomJobSpec proto — camelCase keys.
    spec = {
        "workerPoolSpecs": [{
            "machineSpec": {
                "machineType": cfg["machine_type"],
                "acceleratorType": cfg["accelerator_type"],
                "acceleratorCount": cfg["accelerator_count"],
            },
            "replicaCount": 1,
            "pythonPackageSpec": {
                "executorImageUri":
                    "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest",
                "packageUris": [package_uri],
                "pythonModule": module_name,
                "args": script_args,
            },
        }],
        "baseOutputDirectory": {"outputUriPrefix": base_output_dir},
    }
    if args.spot:
        spec["scheduling"] = {"strategy": "SPOT"}

    cfg_path = Path(tempfile.mkdtemp(prefix="kmidi-vertex-cfg-")) / "custom_job.yaml"
    cfg_path.write_text(yaml.safe_dump(spec, sort_keys=False))

    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--project={args.project}",
        f"--region={args.region}",
        f"--display-name={run_name}",
        f"--config={cfg_path}",
    ]
    if args.service_account:
        cmd.append(f"--service-account={args.service_account}")

    logger.info("Submitting %s | machine=%s+%s×%d | spot=%s",
                run_name, cfg["machine_type"], cfg["accelerator_type"],
                cfg["accelerator_count"], args.spot)
    logger.info("Output dir: %s", base_output_dir)
    logger.info("Config: %s", cfg_path)

    if args.dry_run:
        logger.info("--dry-run: skipping submit.")
        logger.info("YAML config:\n%s", cfg_path.read_text())
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Submit failed:\n%s", result.stderr)
        raise SystemExit(result.returncode)
    logger.info("Submit output:\n%s", result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=list(TASK_CONFIGS.keys()), required=True)
    ap.add_argument("--project", default="devvy-490312")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", default="kmidi-train-us-central1")
    ap.add_argument("--service-account", default=None,
                    help="Optional SA email. Default uses Compute Engine default SA.")
    ap.add_argument("--manifest-name", default="fma_medium_manifest.csv",
                    help="Filename under gs://<bucket>/fma/manifest/")
    ap.add_argument("--spot", action="store_true", default=True,
                    help="Use spot/preemptible VMs (default, required for T4 on free quota)")
    ap.add_argument("--no-spot", dest="spot", action="store_false")
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
