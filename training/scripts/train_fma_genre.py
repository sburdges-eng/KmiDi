#!/usr/bin/env python3
"""FMA genre classifier — baseline for the FMA training sequence.

Reuses the proven AudioClassifier (mel CNN) from the emotion runs, fed
manifest-driven mp3s. Designed to run identically on Mac MPS (smoke /
local baseline) and on Vertex AI (custom job — same script, GCS Fuse
paths under /gcs/<bucket>, AIP_MODEL_DIR for output).

Manifest CSV columns: track_id, file_path, genre_top, subset.

Class imbalance (Rock 6103 vs Easy Listening 21) handled via
WeightedRandomSampler with inverse-frequency weights.

Usage::

    # Local smoke (200 files, tiny model, 1 epoch — under 5 min):
    python training/scripts/train_fma_genre.py --smoke

    # Local baseline:
    python training/scripts/train_fma_genre.py \\
        --manifest ~/Datasets/fma_metadata/fma_medium_manifest.csv \\
        --epochs 30 --batch-size 32

    # Vertex container (--manifest is GCS Fuse path)
    python training/scripts/train_fma_genre.py \\
        --manifest /gcs/kmidi-train-us-central1/fma/fma_medium_manifest.csv \\
        --models-root /gcs/kmidi-train-us-central1/runs \\
        --vertex
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

from src.models.audio_classifier import AudioClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_fma_genre")


# ── dataset ──────────────────────────────────────────────────────


class FMADataset(Dataset):
    """Loads mp3s, computes mel spectrogram (librosa, CPU), returns
    (features [1, n_mels, T], label)."""

    def __init__(self, manifest_df: pd.DataFrame, label_to_id: dict[str, int],
                 sample_rate: int = 16000, n_mels: int = 64,
                 max_duration: float = 6.0, gcs_prefix: str | None = None):
        self.df = manifest_df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = 1024
        self.hop_length = 512
        self.max_samples = int(sample_rate * max_duration)
        self.gcs_prefix = gcs_prefix

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, raw_path: str) -> str:
        if self.gcs_prefix and raw_path.startswith("/Volumes/"):
            tail = raw_path.split("/audio/fma_medium/", 1)
            if len(tail) == 2:
                return f"{self.gcs_prefix.rstrip('/')}/{tail[1]}"
        return raw_path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._resolve(row["file_path"])
        try:
            import librosa
            y, _ = librosa.load(path, sr=self.sample_rate, mono=True,
                                duration=self.max_samples / self.sample_rate)
            if len(y) < self.max_samples // 2:
                y = np.pad(y, (0, self.max_samples // 2 - len(y)))
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
            mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
            feat = torch.from_numpy(mel_db).float().unsqueeze(0)  # [1, n_mels, T]
            label = self.label_to_id[row["genre_top"]]
            return feat, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.debug("decode fail %s: %s", path, e)
            return (torch.zeros(1, self.n_mels, max(self.max_samples // self.hop_length, 8)),
                    torch.tensor(self.label_to_id[row["genre_top"]], dtype=torch.long))


def collate_pad(batch):
    feats, labels = zip(*batch)
    max_t = max(f.shape[-1] for f in feats)
    padded = []
    for f in feats:
        pad = max_t - f.shape[-1]
        if pad > 0:
            f = torch.nn.functional.pad(f, (0, pad))
        padded.append(f)
    return torch.stack(padded, dim=0), torch.stack(labels, dim=0)


# ── SpecAugment (pure torch) ─────────────────────────────────────


class _SpecAugment(nn.Module):
    """Minimal torchaudio-free SpecAugment: one frequency mask and one
    time mask per item, widths uniformly random in [0, max]."""

    def __init__(self, freq_mask: int = 12, time_mask: int = 24):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mels, T] or [B, n_mels, T]
        if x.dim() == 4:
            B, C, F, T = x.shape
        else:
            B, F, T = x.shape
            C = 1
            x = x.unsqueeze(1)
        if self.freq_mask > 0:
            f = torch.randint(0, self.freq_mask + 1, (B,), device=x.device)
            f_start = (torch.rand(B, device=x.device) * (F - f).clamp(min=1)).long()
            for i in range(B):
                if f[i] > 0:
                    x[i, :, f_start[i]:f_start[i] + f[i], :] = 0
        if self.time_mask > 0:
            t = torch.randint(0, self.time_mask + 1, (B,), device=x.device)
            t_start = (torch.rand(B, device=x.device) * (T - t).clamp(min=1)).long()
            for i in range(B):
                if t[i] > 0:
                    x[i, :, :, t_start[i]:t_start[i] + t[i]] = 0
        return x if C > 1 or x.dim() == 4 else x.squeeze(1)


# ── training ─────────────────────────────────────────────────────


def pick_device(force: str | None) -> torch.device:
    if force:
        return torch.device(force)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, desc, spec_aug=None):
    model.train()
    running = 0.0
    n = 0
    for feats, labels in tqdm(loader, desc=desc, leave=False):
        feats, labels = feats.to(device), labels.to(device)
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
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    running = 0.0
    correct = 0
    n = 0
    cm_correct = torch.zeros(num_classes, dtype=torch.long)
    cm_total = torch.zeros(num_classes, dtype=torch.long)
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        running += loss.item() * feats.size(0)
        correct += (pred == labels).sum().item()
        n += feats.size(0)
        for c in range(num_classes):
            mask = labels == c
            cm_total[c] += mask.sum().cpu()
            cm_correct[c] += ((pred == labels) & mask).sum().cpu()
    per_class_acc = (cm_correct.float() / cm_total.clamp(min=1)).tolist()
    bal_acc = float(sum(per_class_acc) / max(len([t for t in cm_total if t > 0]), 1))
    return running / max(n, 1), correct / max(n, 1), bal_acc, per_class_acc


# ── main ────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=str,
                    default=str(Path.home() / "Datasets" / "fma_metadata" / "fma_medium_manifest.csv"))
    ap.add_argument("--gcs-audio-prefix", type=str, default=None,
                    help="If set, rewrite manifest /Volumes/... paths to <prefix>/<sub>/<file>.mp3")
    ap.add_argument("--subset-filter", choices=["all", "small", "medium"], default="all",
                    help="all = small+medium, medium = medium-only, small = small-only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-duration", type=float, default=6.0)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--no-spec-aug", action="store_true")
    ap.add_argument("--freq-mask", type=int, default=12)
    ap.add_argument("--time-mask", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--models-root", type=str,
                    default=os.environ.get("KELLY_MODEL_ROOT", str(Path.home() / "Models")))
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--vertex", action="store_true",
                    help="Vertex AI mode: use AIP_MODEL_DIR for output, log to stderr")
    ap.add_argument("--smoke", action="store_true",
                    help="200 files, 1 epoch, batch 4 — pipeline plumbing check")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.manifest)
    if args.subset_filter == "small":
        df = df[df["subset"] == "small"]
    elif args.subset_filter == "medium":
        df = df[df["subset"] == "medium"]
    df = df.reset_index(drop=True)

    if args.smoke:
        # Stratified-ish sample across genres
        df = (df.groupby("genre_top").head(15).reset_index(drop=True))
        args.epochs = 1
        args.batch_size = 4

    classes = sorted(df["genre_top"].unique())
    label_to_id = {c: i for i, c in enumerate(classes)}
    id_to_label = {i: c for c, i in label_to_id.items()}
    num_classes = len(classes)

    device = pick_device(args.device)

    # 80/10/10 split, stratified per genre
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = df.index[df["genre_top"] == c].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * 0.8)
        n_va = int(n * 0.1)
        train_idx.extend(idx[:n_tr])
        val_idx.extend(idx[n_tr:n_tr + n_va])
        test_idx.extend(idx[n_tr + n_va:])
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Inverse-frequency sampler weights for the imbalanced training set
    cls_count = train_df["genre_top"].value_counts().to_dict()
    sample_weights = train_df["genre_top"].map(lambda c: 1.0 / cls_count[c]).to_numpy()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_df), replacement=True)

    ds_kwargs = dict(label_to_id=label_to_id, sample_rate=args.sample_rate,
                     n_mels=args.n_mels, max_duration=args.max_duration,
                     gcs_prefix=args.gcs_audio_prefix)
    train_set = FMADataset(train_df, **ds_kwargs)
    val_set = FMADataset(val_df, **ds_kwargs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              collate_fn=collate_pad, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_pad,
                            pin_memory=False)

    model = AudioClassifier(num_classes=num_classes, n_mels=args.n_mels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    spec_aug = None
    if not args.no_spec_aug and not args.smoke:
        spec_aug = _SpecAugment(
            freq_mask=args.freq_mask, time_mask=args.time_mask).to(device)

    run_name = args.name or f"fma-genre-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.vertex and "AIP_MODEL_DIR" in os.environ:
        ckpt_dir = Path(os.environ["AIP_MODEL_DIR"])
    else:
        ckpt_dir = Path(args.models_root).expanduser() / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb"))

    manifest_meta = {
        "experiment_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "device": str(device),
        "data_manifest": args.manifest,
        "gcs_audio_prefix": args.gcs_audio_prefix,
        "subset_filter": args.subset_filter,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "num_classes": num_classes,
        "class_names": classes,
        "class_counts_train": cls_count,
        "seed": args.seed,
        "hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "sample_rate": args.sample_rate,
            "max_duration": args.max_duration,
            "n_mels": args.n_mels,
            "patience": args.patience,
            "freq_mask": args.freq_mask if spec_aug else None,
            "time_mask": args.time_mask if spec_aug else None,
            "spec_aug": spec_aug is not None,
            "balanced_sampler": True,
        },
        "torch_version": str(torch.__version__),
        "smoke": args.smoke,
        "vertex": args.vertex,
    }
    try:
        import yaml
        (ckpt_dir / "run_manifest.yaml").write_text(
            yaml.safe_dump(manifest_meta, sort_keys=False))
    except ImportError:
        (ckpt_dir / "run_manifest.yaml").write_text(json.dumps(manifest_meta, indent=2))
    (ckpt_dir / "class_names.json").write_text(json.dumps(classes, indent=2))

    logger.info("Run %s | ckpt=%s | n_train=%d n_val=%d | classes=%d",
                run_name, ckpt_dir, len(train_df), len(val_df), num_classes)

    best_acc = 0.0
    best_bal = 0.0
    best_epoch = -1
    epochs_since_best = 0
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        desc = f"epoch {epoch + 1}/{args.epochs}"
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, desc, spec_aug=spec_aug)
        val_loss, val_acc, val_bal, per_class = validate(
            model, val_loader, criterion, device, num_classes)
        logger.info(
            "[%s] train_loss=%.4f  val_loss=%.4f  val_acc=%.2f%%  val_bal_acc=%.2f%%",
            desc, train_loss, val_loss, val_acc * 100, val_bal * 100,
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("bal_acc/val", val_bal, epoch)

        if val_bal > best_bal:  # Optimize for balanced accuracy on imbalanced data
            best_bal = val_bal
            best_acc = val_acc
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc, "val_bal_acc": val_bal,
                "num_classes": num_classes, "class_names": classes,
                "per_class_acc": per_class,
            }, ckpt_dir / "best.pt")
            logger.info("  ↳ new best bal_acc=%.2f%% (acc=%.2f%%) → %s",
                        val_bal * 100, val_acc * 100, ckpt_dir / "best.pt")
        else:
            epochs_since_best += 1
            if args.patience > 0 and epochs_since_best >= args.patience:
                logger.info("Early stop @ epoch %d, best bal_acc=%.2f%% @ epoch %d",
                            epoch + 1, best_bal * 100, best_epoch + 1)
                break

    writer.close()
    elapsed = time.perf_counter() - t_start
    logger.info("Done. best_bal_acc=%.2f%% best_acc=%.2f%% @ epoch %d wall=%.0fs ckpt=%s",
                best_bal * 100, best_acc * 100, best_epoch + 1, elapsed, ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
