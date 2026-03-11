#!/usr/bin/env python3
"""
Run JEPA training locally (Audio-JEPA and Chord-JEPA) using config/jepa_training.yaml.

Uses local data dirs: data/audio, data/midi (or JEPA_AUDIO_DIR, JEPA_MIDI_DIR).
Checkpoints go to checkpoints/audio_jepa and checkpoints/chord_jepa (or JEPA_CHECKPOINT_DIR).

Usage:
    python3 scripts/train_jepa_local.py              # train both
    python3 scripts/train_jepa_local.py --model audio_jepa
    python3 scripts/train_jepa_local.py --model chord_jepa
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path: str) -> dict:
    with open(path) as f:
        if yaml:
            return yaml.safe_load(f)
        import json
        return json.load(f)


def build_training_config(raw: dict):
    from music_brain.jepa.config import (
        AudioJEPAConfig,
        ChordJEPAConfig,
        TrainingConfig,
    )
    t = raw.get("training", {})
    training = TrainingConfig(
        epochs=int(t.get("epochs", 120)),
        batch_size=int(t.get("batch_size", 12)),
        learning_rate=float(t.get("learning_rate", 3e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        warmup_epochs=int(t.get("warmup_epochs", 10)),
        save_every=int(t.get("save_every", 5)),
        early_stop_patience=int(t.get("early_stop_patience", 15)),
        mixed_precision=bool(t.get("mixed_precision", True)),
        gradient_clip=float(t.get("gradient_clip", 1.0)),
    )
    a = raw.get("audio_jepa", {})
    audio_config = AudioJEPAConfig(
        latent_dim=int(a.get("latent_dim", 256)),
        n_mels=int(a.get("n_mels", 128)),
        max_frames=int(a.get("max_frames", 512)),
        sample_rate=int(a.get("sample_rate", 22050)),
        hop_length=int(a.get("hop_length", 512)),
        mask_ratio=float(a.get("mask_ratio", 0.6)),
        mask_block_size=int(a.get("mask_block_size", 4)),
        tier=str(a.get("tier", "medium")),
    )
    c = raw.get("chord_jepa", {})
    chord_config = ChordJEPAConfig(
        d_model=int(c.get("d_model", 256)),
        num_heads=int(c.get("num_heads", 8)),
        num_layers=int(c.get("num_layers", 4)),
        seq_len=int(c.get("seq_len", 64)),
        num_chords=int(c.get("num_chords", 170)),
        dropout=float(c.get("dropout", 0.1)),
        mask_ratio=float(c.get("mask_ratio", 0.6)),
    )
    return training, audio_config, chord_config


def main():
    parser = argparse.ArgumentParser(description="Train JEPA models locally")
    parser.add_argument(
        "--model",
        choices=["audio_jepa", "chord_jepa", "both"],
        default="both",
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_REPO_ROOT, "config", "jepa_training.yaml"),
        help="Path to JEPA config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (default: from config)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    raw = load_config(config_path)
    training, audio_config, chord_config = build_training_config(raw)
    if args.epochs is not None:
        training.epochs = args.epochs

    audio_dir = os.environ.get("JEPA_AUDIO_DIR", os.path.join(_REPO_ROOT, "data", "audio"))
    midi_dir = os.environ.get("JEPA_MIDI_DIR", os.path.join(_REPO_ROOT, "data", "midi"))
    checkpoint_root = os.environ.get("JEPA_CHECKPOINT_DIR", os.path.join(_REPO_ROOT, "checkpoints"))

    from torch.utils.data import DataLoader
    from music_brain.jepa.datasets import AudioMelDataset, ChordSequenceDataset

    def _midi_files(d: str):
        exts = {".mid", ".midi"}
        out = []
        for dirpath, _, names in os.walk(d):
            for n in names:
                if Path(n).suffix.lower() in exts:
                    out.append(os.path.join(dirpath, n))
        return out

    models_to_run = ["audio_jepa", "chord_jepa"] if args.model == "both" else [args.model]

    for model_type in models_to_run:
        if model_type == "audio_jepa":
            if not os.path.isdir(audio_dir):
                print(f"WARNING: Audio dir not found: {audio_dir}. Create it and add WAV/MP3/FLAC, or set JEPA_AUDIO_DIR.", file=sys.stderr)
            try:
                dataset = AudioMelDataset.from_directory(
                    audio_dir,
                    n_mels=audio_config.n_mels,
                    hop_length=audio_config.hop_length,
                    max_frames=audio_config.max_frames,
                    sr=audio_config.sample_rate,
                )
            except Exception as e:
                print(f"ERROR: Audio-JEPA dataset failed: {e}", file=sys.stderr)
                if "both" in args.model:
                    continue
                sys.exit(1)
            if len(dataset) == 0:
                print(f"ERROR: No audio files in {audio_dir}. Add WAV/MP3/FLAC or set JEPA_AUDIO_DIR.", file=sys.stderr)
                if args.model != "both":
                    sys.exit(1)
                continue
            loader = DataLoader(dataset, batch_size=training.batch_size, shuffle=True, num_workers=0)
            ckpt_dir = os.path.join(checkpoint_root, "audio_jepa")
            export_path = os.path.join(checkpoint_root, "audio_jepa.onnx") if raw.get("checkpoints", {}).get("export_onnx", True) else None
            print("Training Audio-JEPA ...")
            from music_brain.jepa.trainer import train_audio_jepa
            train_audio_jepa(loader, audio_config, training, checkpoint_dir=ckpt_dir, export_path=export_path)
        else:
            midi_files = _midi_files(midi_dir) if os.path.isdir(midi_dir) else []
            dataset = ChordSequenceDataset(
                files=midi_files if midi_files else None,
                seq_len=chord_config.seq_len,
                num_chords=chord_config.num_chords,
                num_samples=256 if not midi_files else None,
            )
            loader = DataLoader(dataset, batch_size=training.batch_size, shuffle=True, num_workers=0)
            ckpt_dir = os.path.join(checkpoint_root, "chord_jepa")
            print("Training Chord-JEPA ...")
            from music_brain.jepa.trainer import train_chord_jepa
            train_chord_jepa(loader, chord_config, training, checkpoint_dir=ckpt_dir)
    print("Done.")


if __name__ == "__main__":
    main()
