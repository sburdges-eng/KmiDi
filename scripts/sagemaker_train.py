#!/usr/bin/env python3
"""
SageMaker training entrypoint for JEPA (Audio-JEPA and Chord-JEPA).

Expects SageMaker contract:
  - Hyperparameters: /opt/ml/input/config/hyperparameters.json
  - Input data: /opt/ml/input/data/audio, /opt/ml/input/data/midi (channel names)
  - Output: /opt/ml/model (SageMaker uploads this to S3)

Usage inside container:
  python sagemaker_train.py

Can also be run locally with env vars or defaults for paths (e.g. for container debugging).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Repo root on host; in SageMaker container we rely on installed package
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sagemaker_train")

# SageMaker paths
SM_INPUT_CONFIG = "/opt/ml/input/config"
SM_HYPERPARAMETERS = os.path.join(SM_INPUT_CONFIG, "hyperparameters.json")
SM_INPUT_DATA = "/opt/ml/input/data"
SM_MODEL = "/opt/ml/model"


def load_hyperparameters() -> dict:
    """Load hyperparameters from SageMaker config or env."""
    if os.path.isfile(SM_HYPERPARAMETERS):
        with open(SM_HYPERPARAMETERS) as f:
            return json.load(f)
    # Local / test: from env or minimal defaults
    defaults = {
        "model_type": os.environ.get("model_type", "audio_jepa"),
        "epochs": os.environ.get("epochs", "120"),
        "batch_size": os.environ.get("batch_size", "12"),
        "learning_rate": os.environ.get("learning_rate", "3e-4"),
        "mixed_precision": os.environ.get("mixed_precision", "True"),
        "export_onnx": os.environ.get("export_onnx", "True"),
    }
    return defaults


def data_path(channel: str) -> str:
    """Path to a SageMaker input channel."""
    path = os.path.join(SM_INPUT_DATA, channel)
    if os.path.isdir(path):
        return path
    # Fallback for local runs
    return os.path.join(_REPO_ROOT, "data", channel)


def build_audio_dataloader(audio_dir: str, hyperparams: dict) -> "DataLoader":
    """Build DataLoader for Audio-JEPA from directory."""
    from torch.utils.data import DataLoader

    from music_brain.jepa.datasets import AudioMelDataset

    dataset = AudioMelDataset.from_directory(
        audio_dir,
        n_mels=int(hyperparams.get("model_n_mels", 128)),
        hop_length=int(hyperparams.get("model_hop_length", 512)),
        max_frames=int(hyperparams.get("model_max_frames", 512)),
        sr=int(hyperparams.get("model_sample_rate", 22050)),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No audio files found under {audio_dir}")
    batch_size = int(hyperparams.get("batch_size", 12))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )


def collect_midi_files(midi_dir: str) -> list:
    """Recursively collect .mid / .midi files under directory."""
    exts = {".mid", ".midi"}
    files = []
    for dirpath, _, filenames in os.walk(midi_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                files.append(os.path.join(dirpath, fname))
    return files


def build_chord_dataloader(midi_dir: str, hyperparams: dict) -> "DataLoader":
    """Build DataLoader for Chord-JEPA from MIDI directory."""
    from torch.utils.data import DataLoader

    from music_brain.jepa.datasets import ChordSequenceDataset

    files = collect_midi_files(midi_dir)
    seq_len = int(hyperparams.get("model_seq_len", 64))
    num_chords = int(hyperparams.get("model_num_chords", 170))
    dataset = ChordSequenceDataset(
        files=files if files else None,
        seq_len=seq_len,
        num_chords=num_chords,
        num_samples=256 if not files else None,
    )
    batch_size = int(hyperparams.get("batch_size", 12))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def hyperparams_to_configs(hyperparams: dict):
    """Build AudioJEPAConfig/ChordJEPAConfig and TrainingConfig from hyperparameters."""
    from music_brain.jepa.config import (
        AudioJEPAConfig,
        ChordJEPAConfig,
        TrainingConfig,
    )

    model_type = hyperparams.get("model_type", "audio_jepa")

    def get_model(key: str, default):
        val = hyperparams.get(f"model_{key}")
        return type(default)(val) if val is not None else default

    if model_type == "audio_jepa":
        model_config = AudioJEPAConfig(
            latent_dim=get_model("latent_dim", 256),
            n_mels=get_model("n_mels", 128),
            max_frames=get_model("max_frames", 512),
            sample_rate=get_model("sample_rate", 22050),
            hop_length=get_model("hop_length", 512),
            mask_ratio=get_model("mask_ratio", 0.6),
            mask_block_size=get_model("mask_block_size", 4),
            tier=str(hyperparams.get("model_tier", "medium")),
        )
    elif model_type == "chord_jepa":
        model_config = ChordJEPAConfig(
            d_model=get_model("d_model", 256),
            num_heads=get_model("num_heads", 8),
            num_layers=get_model("num_layers", 4),
            seq_len=get_model("seq_len", 64),
            num_chords=get_model("num_chords", 170),
            dropout=get_model("dropout", 0.1),
            mask_ratio=get_model("mask_ratio", 0.6),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    training = TrainingConfig(
        epochs=int(hyperparams.get("epochs", 120)),
        batch_size=int(hyperparams.get("batch_size", 12)),
        learning_rate=float(hyperparams.get("learning_rate", 3e-4)),
        weight_decay=0.01,
        warmup_epochs=10,
        save_every=5,
        early_stop_patience=15,
        mixed_precision=hyperparams.get("mixed_precision", "true").lower() == "true",
        gradient_clip=1.0,
    )

    return model_type, model_config, training


def main() -> None:
    hyperparams = load_hyperparameters()
    model_type = hyperparams.get("model_type", "audio_jepa")
    export_onnx = hyperparams.get("export_onnx", "true").lower() == "true"

    logger.info("SageMaker JEPA training: model_type=%s", model_type)

    _, model_config, training = hyperparams_to_configs(hyperparams)
    os.makedirs(SM_MODEL, exist_ok=True)
    checkpoint_dir = os.path.join(SM_MODEL, model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if model_type == "audio_jepa":
        audio_dir = data_path("audio")
        dataloader = build_audio_dataloader(audio_dir, hyperparams)
        from music_brain.jepa.trainer import train_audio_jepa

        train_audio_jepa(
            dataloader,
            model_config,
            training,
            checkpoint_dir=checkpoint_dir,
            export_path=os.path.join(SM_MODEL, "audio_jepa.onnx") if export_onnx else None,
        )
    elif model_type == "chord_jepa":
        midi_dir = data_path("midi")
        dataloader = build_chord_dataloader(midi_dir, hyperparams)
        from music_brain.jepa.trainer import train_chord_jepa

        train_chord_jepa(
            dataloader,
            model_config,
            training,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info("Training complete. Artifacts in %s", SM_MODEL)


if __name__ == "__main__":
    main()
