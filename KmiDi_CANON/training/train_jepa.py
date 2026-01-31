"""
JEPA training entrypoint — GPU-validated, auto-resume, config-driven.

Usage:
  python -m KmiDi_CANON.training.train_jepa --config configs/jepa_audio.yaml --output_dir ~/Models/checkpoints/exp_001
  python -m KmiDi_CANON.training.train_jepa --config /workspace/config.yaml --output_dir /workspace/checkpoints  # cloud

Runs in tmux for long jobs. See docs/CLOUD_TRAINING.md, docs/ENV_AND_TMUX.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_repo_to_path():
    """Ensure repo root and brain are on path for imports."""
    root = Path(__file__).resolve().parent.parent.parent
    brain = root / "KmiDi_CANON" / "brain"
    for p in (str(root), str(brain)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return path to latest checkpoint for auto-resume, or None."""
    if not output_dir.exists():
        return None
    ckpts = list(output_dir.glob("checkpoint_*.pt")) + list(output_dir.glob("last.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="JEPA training (config-driven, auto-resume)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", default=None, help="Checkpoint dir (default from config)")
    parser.add_argument("--resume", action="store_true", help="Force resume from latest checkpoint")
    args = parser.parse_args()

    _add_repo_to_path()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    config = _load_config(str(config_path))
    output_dir = Path(args.output_dir or config.get("train", {}).get("output_dir", "./checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU validation
    device = config.get("device", "cuda")
    if device == "cuda" and not _check_gpu():
        logger.error("CUDA requested but torch.cuda.is_available() is False.")
        logger.error("  Check: nvidia-smi, CUDA drivers, PyTorch CUDA build.")
        return 1

    import torch
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Auto-resume
    auto_resume = config.get("train", {}).get("auto_resume", True) or args.resume
    ckpt = _find_latest_checkpoint(output_dir) if auto_resume else None
    if ckpt:
        logger.info("Resume from %s", ckpt)
    else:
        logger.info("No checkpoint found, starting from scratch.")

    # Placeholder: real JEPA training loop not yet implemented.
    # This validates the pipeline and exits. Replace with actual training.
    logger.info("JEPA training loop not yet implemented.")
    logger.info("  Config loaded. Pipeline ready for implementation.")
    logger.info("  Checkpoint dir: %s", output_dir.resolve())
    logger.info("  See configs/jepa_*.yaml and experiments/research for JEPA spec.")

    # Write minimal manifest for traceability (DATA_AND_TRAINING)
    manifest_path = output_dir / "manifest_run.json"
    manifest = {
        "experiment": output_dir.name or "jepa_train",
        "config": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": config.get("seed", 1337),
        "device": device,
        "resumed_from": str(ckpt) if ckpt else None,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("  Manifest: %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
