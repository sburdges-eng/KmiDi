#!/usr/bin/env python3
"""
Vertex AI training entrypoint for JEPA (Audio-JEPA and Chord-JEPA).

Expects Vertex AI contract:
  - Hyperparameters: Passed as command-line arguments.
  - Input data: GCS paths passed as arguments or via AIP_ env vars.
  - Output: AIP_MODEL_DIR (Vertex AI uses this to save the model to GCS).

Usage inside container:
  python vertex_train.py --model_type audio_jepa --epochs 120 ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Repo root on host; in Vertex container we rely on installed package
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vertex_train")

# Vertex AI paths
AIP_MODEL_DIR = os.environ.get(
    "AIP_MODEL_DIR", "/tmp/kmidi_model"
)  # Fallback to /tmp for local testing
AIP_DATA_FORMAT = os.environ.get("AIP_DATA_FORMAT", "")


def build_audio_dataloader(audio_dir: str, args: argparse.Namespace) -> "DataLoader":
    """Build DataLoader for Audio-JEPA from directory."""
    from torch.utils.data import DataLoader
    from music_brain.jepa.datasets import AudioMelDataset

    # Handle GCS Fuse path if audio_dir starts with gs://
    if audio_dir.startswith("gs://"):
        bucket_name = audio_dir.replace("gs://", "").split("/")[0]
        relative_path = "/".join(audio_dir.replace("gs://", "").split("/")[1:])
        audio_dir = f"/gcs/{bucket_name}/{relative_path}"
        logger.info("Using GCS Fuse path: %s", audio_dir)

    dataset = AudioMelDataset.from_directory(
        audio_dir,
        n_mels=args.model_n_mels,
        hop_length=args.model_hop_length,
        max_frames=args.model_max_frames,
        sr=args.model_sample_rate,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No audio files found under {audio_dir}")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )


def collect_midi_files(midi_dir: str) -> list:
    """Recursively collect .mid / .midi files under directory."""
    # Handle GCS Fuse
    if midi_dir.startswith("gs://"):
        bucket_name = midi_dir.replace("gs://", "").split("/")[0]
        relative_path = "/".join(midi_dir.replace("gs://", "").split("/")[1:])
        midi_dir = f"/gcs/{bucket_name}/{relative_path}"
        logger.info("Using GCS Fuse path for MIDI: %s", midi_dir)

    exts = {".mid", ".midi"}
    files = []
    for dirpath, _, filenames in os.walk(midi_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                files.append(os.path.join(dirpath, fname))
    return files


def build_chord_dataloader(midi_dir: str, args: argparse.Namespace) -> "DataLoader":
    """Build DataLoader for Chord-JEPA from MIDI directory."""
    from torch.utils.data import DataLoader
    from music_brain.jepa.datasets import ChordSequenceDataset

    files = collect_midi_files(midi_dir)
    dataset = ChordSequenceDataset(
        files=files if files else None,
        seq_len=args.model_seq_len,
        num_chords=args.model_num_chords,
        num_samples=256 if not files else None,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )


def args_to_configs(args: argparse.Namespace):
    """Build AudioJEPAConfig/ChordJEPAConfig and TrainingConfig from arguments."""
    from music_brain.jepa.config import (
        AudioJEPAConfig,
        ChordJEPAConfig,
        TrainingConfig,
    )

    if args.model_type == "audio_jepa":
        model_config = AudioJEPAConfig(
            latent_dim=args.model_latent_dim,
            n_mels=args.model_n_mels,
            max_frames=args.model_max_frames,
            sample_rate=args.model_sample_rate,
            hop_length=args.model_hop_length,
            mask_ratio=args.model_mask_ratio,
            mask_block_size=args.model_mask_block_size,
            tier=args.model_tier,
        )
    elif args.model_type == "chord_jepa":
        model_config = ChordJEPAConfig(
            d_model=args.model_d_model,
            num_heads=args.model_num_heads,
            num_layers=args.model_num_layers,
            seq_len=args.model_seq_len,
            num_chords=args.model_num_chords,
            dropout=args.model_dropout,
            mask_ratio=args.model_mask_ratio,
        )
    elif args.model_type == "emotion_probe":
        model_config = None
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_epochs=10,
        save_every=5,
        early_stop_patience=15,
        mixed_precision=args.mixed_precision,
        gradient_clip=1.0,
    )

    return args.model_type, model_config, training


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI JEPA training")

    # Generic training args
    parser.add_argument("--model_type", type=str, default="audio_jepa")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--mixed_precision", type=bool, default=True)
    parser.add_argument("--export_onnx", type=bool, default=True)

    # Data paths (can be GCS gs:// paths or local)
    parser.add_argument(
        "--audio_dir", type=str, default=os.environ.get("AIP_TRAINING_DATA_URI", "")
    )
    parser.add_argument("--midi_dir", type=str, default="")

    # Audio-JEPA specific
    parser.add_argument("--model_latent_dim", type=int, default=256)
    parser.add_argument("--model_n_mels", type=int, default=128)
    parser.add_argument("--model_max_frames", type=int, default=512)
    parser.add_argument("--model_sample_rate", type=int, default=22050)
    parser.add_argument("--model_hop_length", type=int, default=512)
    parser.add_argument("--model_mask_ratio", type=float, default=0.6)
    parser.add_argument("--model_mask_block_size", type=int, default=4)
    parser.add_argument("--model_tier", type=str, default="medium")

    # Chord-JEPA specific
    parser.add_argument("--model_d_model", type=int, default=256)
    parser.add_argument("--model_num_heads", type=int, default=8)
    parser.add_argument("--model_num_layers", type=int, default=4)
    parser.add_argument("--model_seq_len", type=int, default=64)
    parser.add_argument("--model_num_chords", type=int, default=170)
    parser.add_argument("--model_dropout", type=float, default=0.1)

    # Emotion Probe specific
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--probe_hidden_dim", type=int, default=128)

    args = parser.parse_args()

    logger.info("Vertex AI JEPA training: model_type=%s", args.model_type)

    _, model_config, training = args_to_configs(args)

    # AIP_MODEL_DIR is usually gs://bucket/path
    # Vertex AI mounts this at /gcs/bucket/path
    model_dir = AIP_MODEL_DIR
    if model_dir.startswith("gs://"):
        bucket_name = model_dir.replace("gs://", "").split("/")[0]
        relative_path = "/".join(model_dir.replace("gs://", "").split("/")[1:])
        model_dir = f"/gcs/{bucket_name}/{relative_path}"

    os.makedirs(model_dir, exist_ok=True)
    checkpoint_dir = os.path.join(model_dir, args.model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.model_type == "audio_jepa":
        audio_dir = args.audio_dir
        dataloader = build_audio_dataloader(audio_dir, args)
        from music_brain.jepa.trainer import train_audio_jepa

        train_audio_jepa(
            dataloader,
            model_config,
            training,
            checkpoint_dir=checkpoint_dir,
            export_path=os.path.join(model_dir, "audio_jepa.onnx") if args.export_onnx else None,
        )
    elif args.model_type == "chord_jepa":
        midi_dir = args.midi_dir
        dataloader = build_chord_dataloader(midi_dir, args)
        from music_brain.jepa.trainer import train_chord_jepa

        train_chord_jepa(
            dataloader,
            model_config,
            training,
            checkpoint_dir=checkpoint_dir,
        )
    elif args.model_type == "emotion_probe":
        import subprocess

        embeddings_path = args.embeddings_path
        if embeddings_path.startswith("gs://"):
            bucket = embeddings_path.replace("gs://", "").split("/")[0]
            rel = "/".join(embeddings_path.replace("gs://", "").split("/")[1:])
            embeddings_path = f"/gcs/{bucket}/{rel}"

        out_checkpoint = os.path.join(checkpoint_dir, "best_probe.pt")

        cmd = [
            sys.executable,
            os.path.join(_SCRIPT_DIR, "train_emotion_probe.py"),
            "--embeddings",
            embeddings_path,
            "--output",
            out_checkpoint,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.learning_rate),
            "--hidden-dim",
            str(args.probe_hidden_dim),
        ]

        logger.info("Delegating to train_emotion_probe.py with cmd: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    logger.info("Training complete. Artifacts in %s", model_dir)


if __name__ == "__main__":
    main()
