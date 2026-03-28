#!/usr/bin/env python3
"""
JEPA Training Script (Phase 5).

Thin wrapper around music_brain.jepa.trainer that:
- Loads config from config/jepa_training.yaml
- Trains Audio-JEPA and/or Chord-JEPA
- Saves checkpoints to ModelRegistry-compatible paths
- Auto-registers trained models in the registry

Usage:
    python3 training/scripts/train_jepa.py                    # train both
    python3 training/scripts/train_jepa.py --model audio      # audio only
    python3 training/scripts/train_jepa.py --model chord      # chord only
    python3 training/scripts/train_jepa.py --config custom.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_jepa")

try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
except ImportError:
    logger.error("PyTorch not installed. Run: pip install torch")
    sys.exit(1)


def load_config(path: str) -> dict:
    with open(path) as f:
        if yaml:
            return yaml.safe_load(f)
        return json.load(f)


def register_checkpoint(
    model_name: str,
    checkpoint_path: str,
    task: str,
    config: dict,
) -> None:
    """Register a trained checkpoint in the ModelRegistry."""
    try:
        from music_brain.penta_core.ml.model_registry import (
            ModelInfo,
            ModelBackend,
            ModelTask,
            get_registry,
        )

        task_map = {
            "audio_jepa": ModelTask.AUDIO_CLASSIFICATION,
            "chord_jepa": ModelTask.CHORD_PREDICTION,
            "emotion": ModelTask.EMOTION_CLASSIFICATION,
        }

        model_info = ModelInfo(
            name=model_name,
            task=task_map.get(task, ModelTask.CUSTOM),
            backend=ModelBackend.PYTORCH,
            path=checkpoint_path,
            version="1.0.0",
            sample_rate=config.get("audio_jepa", {}).get("sample_rate", 22050),
            description=f"JEPA {task} model trained {datetime.now().isoformat()}",
            tags=["jepa", task, "auto-registered"],
        )

        registry = get_registry()
        registry.register(model_info)

        # Save model_info.json next to checkpoint for future discovery
        info_path = Path(checkpoint_path).parent / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info.to_dict(), f, indent=2)

        logger.info("Registered model '%s' in registry", model_name)
    except Exception:
        logger.warning("Could not register model in registry (non-fatal)", exc_info=True)


def train_audio_jepa(config: dict, checkpoint_dir: Path) -> str:
    """Train Audio-JEPA and return checkpoint path."""
    from music_brain.jepa.config import AudioJEPAConfig, TrainingConfig
    from music_brain.jepa.trainer import train_audio_jepa as _train

    t = config.get("training", {})
    a = config.get("audio_jepa", {})

    training_cfg = TrainingConfig(
        epochs=t.get("epochs", 120),
        batch_size=t.get("batch_size", 12),
        learning_rate=t.get("learning_rate", 3e-4),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_epochs=t.get("warmup_epochs", 10),
        save_every=t.get("save_every", 5),
        early_stop_patience=t.get("early_stop_patience", 15),
        mixed_precision=t.get("mixed_precision", True),
        gradient_clip=t.get("gradient_clip", 1.0),
    )

    audio_cfg = AudioJEPAConfig(
        latent_dim=a.get("latent_dim", 256),
        n_mels=a.get("n_mels", 128),
        max_frames=a.get("max_frames", 512),
        sample_rate=a.get("sample_rate", 22050),
        hop_length=a.get("hop_length", 512),
        mask_ratio=a.get("mask_ratio", 0.6),
        mask_block_size=a.get("mask_block_size", 4),
        tier=a.get("tier", "medium"),
    )

    ckpt_dir = checkpoint_dir / "audio_jepa"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data directories
    data_dir = os.environ.get("JEPA_AUDIO_DIR", str(_REPO_ROOT / "data" / "audio"))

    logger.info("Training Audio-JEPA (data=%s, ckpt=%s)", data_dir, ckpt_dir)
    result = _train(
        audio_config=audio_cfg,
        training_config=training_cfg,
        data_dir=data_dir,
        checkpoint_dir=str(ckpt_dir),
    )

    best_path = str(ckpt_dir / "best.pt")
    if Path(best_path).exists():
        return best_path
    # Fallback to last checkpoint
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    return str(ckpts[-1]) if ckpts else ""


def train_chord_jepa(config: dict, checkpoint_dir: Path) -> str:
    """Train Chord-JEPA and return checkpoint path."""
    from music_brain.jepa.config import ChordJEPAConfig, TrainingConfig
    from music_brain.jepa.trainer import train_chord_jepa as _train

    t = config.get("training", {})
    c = config.get("chord_jepa", {})

    training_cfg = TrainingConfig(
        epochs=t.get("epochs", 120),
        batch_size=t.get("batch_size", 12),
        learning_rate=t.get("learning_rate", 3e-4),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_epochs=t.get("warmup_epochs", 10),
        save_every=t.get("save_every", 5),
        early_stop_patience=t.get("early_stop_patience", 15),
        mixed_precision=t.get("mixed_precision", True),
        gradient_clip=t.get("gradient_clip", 1.0),
    )

    chord_cfg = ChordJEPAConfig(
        d_model=c.get("d_model", 256),
        num_heads=c.get("num_heads", 8),
        num_layers=c.get("num_layers", 4),
        seq_len=c.get("seq_len", 64),
        num_chords=c.get("num_chords", 170),
        dropout=c.get("dropout", 0.1),
        mask_ratio=c.get("mask_ratio", 0.6),
    )

    ckpt_dir = checkpoint_dir / "chord_jepa"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir = os.environ.get("JEPA_MIDI_DIR", str(_REPO_ROOT / "data" / "midi"))

    logger.info("Training Chord-JEPA (data=%s, ckpt=%s)", data_dir, ckpt_dir)
    result = _train(
        chord_config=chord_cfg,
        training_config=training_cfg,
        data_dir=data_dir,
        checkpoint_dir=str(ckpt_dir),
    )

    best_path = str(ckpt_dir / "best.pt")
    if Path(best_path).exists():
        return best_path
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    return str(ckpts[-1]) if ckpts else ""


def main():
    parser = argparse.ArgumentParser(description="Train JEPA models")
    parser.add_argument(
        "--model", choices=["audio", "chord", "both"], default="both",
        help="Which model to train",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(_REPO_ROOT / "config" / "jepa_training.yaml"),
        help="Path to JEPA training config",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=str(_REPO_ROOT / "checkpoints"),
        help="Base checkpoint directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)

    logger.info("JEPA Training — model=%s, config=%s", args.model, args.config)

    if args.model in ("audio", "both"):
        ckpt = train_audio_jepa(config, checkpoint_dir)
        if ckpt:
            register_checkpoint("audio_jepa", ckpt, "audio_jepa", config)
            logger.info("Audio-JEPA checkpoint: %s", ckpt)

    if args.model in ("chord", "both"):
        ckpt = train_chord_jepa(config, checkpoint_dir)
        if ckpt:
            register_checkpoint("chord_jepa", ckpt, "chord_jepa", config)
            logger.info("Chord-JEPA checkpoint: %s", ckpt)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
