#!/usr/bin/env python3
"""Laptop/MPS emotion classifier training for KmiDi.

A self-contained launcher that stitches two working pieces together:

- ``music_brain.penta_core.ml.audio_dataset`` — the Mac-optimized, lazy-
  loading AudioDataset + ``create_dataloaders`` (designed for <16GB RAM).
- ``training.src.models.audio_classifier.AudioClassifier`` — the existing
  small CNN (mel-spectrogram → logits).

The older ``training/scripts/train_emotion*.py`` scripts still reference a
``src.data`` / ``src.training`` layout that was gutted in a prior refactor;
this script intentionally avoids those so it actually runs. When/if
``src/data.py`` and ``src/training.py`` come back, migrate here.

Data layout (DATA LAW, docs/DATA_AND_TRAINING.md):
    <data-dir>/<emotion>/<files>.wav    # label = immediate parent dir name

Dataset path resolution order:
    1. --data-dir CLI flag
    2. $KMIDI_EMOTION_DIR env var
    3. $KELLY_AUDIO_DATA_ROOT/emotions/ravdess/processed (governance §1)
    4. ~/Datasets/kmidi_emotion/

The canonical convention (CLAUDE.md: "no hardcoded /Volumes/ paths in
source") is to symlink external-drive data into ~/Datasets — e.g.::

    ln -s "/Volumes/Sean's SSD/Datasets/processed/emotions/ravdess/processed" \\
          ~/Datasets/kmidi_emotion

Checkpoints go to ~/Models/checkpoints/<run-name>/ per governance §2, with
a ``run_manifest.yaml`` placed alongside per governance §3.

Usage::

    # Smoke test — one quick epoch on a small subset, verify plumbing:
    python3 training/scripts/train_emotion_local.py --smoke

    # Standard laptop run:
    python3 training/scripts/train_emotion_local.py --epochs 20

    # Point at a specific corpus:
    python3 training/scripts/train_emotion_local.py \\
        --data-dir "/Volumes/Sean's SSD/Datasets/processed/emotions/cremad"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

# Pull the two working modules.
from music_brain.penta_core.ml.audio_dataset import (  # noqa: E402
    AudioDataset,
    AudioDatasetTorch,
)
from src.models.audio_classifier import AudioClassifier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_emotion_local")

# ── dataset discovery ────────────────────────────────────────────

DEFAULT_CANDIDATES = [
    Path.home() / "Datasets" / "kmidi_emotion",
    _REPO_ROOT / "data" / "emotion",
]


def resolve_data_dir(cli_arg: str | None) -> Path:
    candidates: list[Path] = []
    if cli_arg:
        candidates.append(Path(cli_arg).expanduser())
    env = os.environ.get("KMIDI_EMOTION_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    audio_root = os.environ.get("KELLY_AUDIO_DATA_ROOT")
    if audio_root:
        candidates.append(Path(audio_root).expanduser() / "emotions" / "ravdess" / "processed")
    candidates.extend(DEFAULT_CANDIDATES)
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise SystemExit(
        "No emotion dataset found. Tried: "
        + "; ".join(str(c) for c in candidates)
        + "\nSymlink your dataset to ~/Datasets/kmidi_emotion "
        + "(per docs/DATA_AND_TRAINING.md §1), set KMIDI_EMOTION_DIR, "
        + "or pass --data-dir."
    )


def pick_device(force: str | None) -> torch.device:
    if force:
        return torch.device(force)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── dataloader ───────────────────────────────────────────────────


def collate_pad(batch):
    """Right-pad variable-length mel spectrograms in the time dimension
    to the batch max so default stacking works. Expects each item to be
    (features [1, n_mels, T], label)."""
    feats, labels = zip(*batch)
    max_t = max(f.shape[-1] for f in feats)
    padded = []
    for f in feats:
        pad = max_t - f.shape[-1]
        if pad > 0:
            padded.append(torch.nn.functional.pad(f, (0, pad)))
        else:
            padded.append(f)
    return torch.stack(padded, dim=0), torch.stack(labels, dim=0)


def build_loaders(
    torch_dataset: AudioDatasetTorch,
    batch_size: int,
    split_ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    n = len(torch_dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    tr = indices[:n_train].tolist()
    va = indices[n_train:n_train + n_val].tolist()
    te = indices[n_train + n_val:].tolist()

    common = dict(batch_size=batch_size, num_workers=0, pin_memory=False,
                  collate_fn=collate_pad)
    return (
        DataLoader(Subset(torch_dataset, tr), shuffle=True, **common),
        DataLoader(Subset(torch_dataset, va), shuffle=False, **common),
        DataLoader(Subset(torch_dataset, te), shuffle=False, **common),
    )


# ── training loop ────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device, desc: str,
                    spec_aug=None):
    model.train()
    running = 0.0
    n = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        feats, labels = batch[0].to(device), batch[1].to(device)
        if spec_aug is not None:
            feats = spec_aug(feats)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * feats.size(0)
        n += feats.size(0)
    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    correct = 0
    n = 0
    for batch in loader:
        feats, labels = batch[0].to(device), batch[1].to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        running += loss.item() * feats.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += feats.size(0)
    return running / max(n, 1), correct / max(n, 1)


# ── run manifest ────────────────────────────────────────────────


def write_manifest(path: Path, meta: dict) -> None:
    # YAML is the governance-preferred format, but PyYAML isn't in the
    # core deps. Fall back to JSON-with-.yaml-extension which is valid YAML.
    try:
        import yaml

        path.write_text(yaml.safe_dump(meta, sort_keys=False))
    except ImportError:
        path.write_text(json.dumps(meta, indent=2))


# ── main ────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-duration", type=float, default=3.0,
                    help="Truncate audio longer than this (seconds).")
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None, choices=[None, "mps", "cuda", "cpu"])
    ap.add_argument("--name", type=str, default=None,
                    help="Run slug. Default: emotion-local-YYYYMMDD-HHMMSS.")
    ap.add_argument("--models-root", type=str,
                    default=os.environ.get("KELLY_MODEL_ROOT", str(Path.home() / "Models")))
    ap.add_argument("--weight-decay", type=float, default=0.05,
                    help="AdamW weight decay (default 0.05).")
    ap.add_argument("--patience", type=int, default=5,
                    help="Early stop after N epochs without val_acc improvement. 0 disables.")
    ap.add_argument("--no-spec-aug", action="store_true",
                    help="Disable SpecAugment (frequency + time masking).")
    ap.add_argument("--freq-mask", type=int, default=12,
                    help="Max frequency-mask width (mel bins). Default 12 of n_mels.")
    ap.add_argument("--time-mask", type=int, default=20,
                    help="Max time-mask width (frames). Default 20.")
    ap.add_argument("--smoke", action="store_true",
                    help="Verify plumbing: 1 epoch, batch-size 4, stop after ~20 batches.")
    args = ap.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    device = pick_device(args.device)

    if args.smoke:
        args.epochs = 1
        args.batch_size = 4

    run_name = args.name or f"emotion-local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ckpt_dir = Path(args.models_root).expanduser() / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ckpt_dir / "tb"
    writer = SummaryWriter(log_dir=str(log_dir))

    logger.info("Run name      : %s", run_name)
    logger.info("Data dir      : %s", data_dir)
    logger.info("Checkpoint dir: %s", ckpt_dir)
    logger.info("Device        : %s", device)

    # Build the dataset (auto-labels from subdir names).
    raw_dataset = AudioDataset(
        data_dir=data_dir,
        sample_rate=args.sample_rate,
        mono=True,
        max_duration=args.max_duration,
        n_mels=args.n_mels,
    )
    if len(raw_dataset.samples) == 0:
        logger.error("Dataset is empty under %s", data_dir)
        return 2
    num_classes = len(raw_dataset.label_to_id)
    class_names = [raw_dataset.id_to_label[i] for i in range(num_classes)]
    logger.info("Found %d samples, %d classes: %s",
                len(raw_dataset.samples), num_classes, class_names)

    # Wrap in torch Dataset, then split. We build loaders directly (rather
    # than using create_dataloaders) so we can install a pad-to-batch-max
    # collate_fn for variable-length clips.
    torch_dataset = AudioDatasetTorch(raw_dataset, return_waveform=False)
    train_loader, val_loader, _test_loader = build_loaders(
        torch_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Model / optim / loss
    model = AudioClassifier(num_classes=num_classes, n_mels=args.n_mels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # SpecAugment: chained frequency + time masking applied per-batch on
    # the device. Disabled in smoke runs (no value, just adds variance).
    spec_aug = None
    if not args.no_spec_aug and not args.smoke:
        spec_aug = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=args.freq_mask),
            T.TimeMasking(time_mask_param=args.time_mask),
        ).to(device)

    # Governance-required run manifest (§4)
    manifest = {
        "experiment_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "device": str(device),
        "data_dir": str(data_dir),
        "num_samples": len(raw_dataset.samples),
        "num_classes": num_classes,
        "class_names": class_names,
        "seed": args.seed,
        "hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sample_rate": args.sample_rate,
            "max_duration": args.max_duration,
            "n_mels": args.n_mels,
        },
        "torch_version": str(torch.__version__),
        "smoke": args.smoke,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "spec_aug": spec_aug is not None,
        "freq_mask": args.freq_mask if spec_aug is not None else None,
        "time_mask": args.time_mask if spec_aug is not None else None,
    }
    write_manifest(ckpt_dir / "run_manifest.yaml", manifest)

    # Persist class mapping for downstream inference
    (ckpt_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

    # Train
    best_acc = 0.0
    best_epoch = -1
    epochs_since_best = 0
    for epoch in range(args.epochs):
        desc = f"epoch {epoch + 1}/{args.epochs}"
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, desc, spec_aug=spec_aug)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(
            "[%s] train_loss=%.4f  val_loss=%.4f  val_acc=%.2f%%",
            desc, train_loss, val_loss, val_acc * 100,
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "class_names": class_names,
                },
                ckpt_dir / "best.pt",
            )
            logger.info("  ↳ new best: %.2f%% → %s", val_acc * 100, ckpt_dir / "best.pt")
        else:
            epochs_since_best += 1
            if args.patience > 0 and epochs_since_best >= args.patience:
                logger.info("Early stop: no val_acc improvement in %d epochs (best=%.2f%% @ epoch %d).",
                            args.patience, best_acc * 100, best_epoch + 1)
                break

    writer.close()
    logger.info("Training complete. Best val acc: %.2f%% @ epoch %d  ckpt dir: %s",
                best_acc * 100, best_epoch + 1, ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
