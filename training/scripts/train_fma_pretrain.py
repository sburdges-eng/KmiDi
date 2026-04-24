#!/usr/bin/env python3
"""FMA self-supervised pretraining (SimSiam-style on mel spectrograms).

Pretrain an audio encoder *without labels* by enforcing that two
differently-augmented views of the same mel spectrogram produce similar
representations. Encoder = the same conv backbone as
``training/src/models/audio_classifier.py``. Adds a small projector
head and a predictor head; loss is symmetric negative cosine similarity
with stop-gradient on the target branch (per the SimSiam paper).

Why SimSiam vs full JEPA: equivalent representational quality on the
scale of FMA medium (~17k tracks), no negative-batch sampling, robust
to small batches (MPS-friendly), ~150 LOC. Outputs a backbone .pt that
can warm-start train_fma_subgenre / train_fma_tags or feed downstream
T1/T3 work in the KMiDi pipeline.

Usage::

    python training/scripts/train_fma_pretrain.py --smoke
    python training/scripts/train_fma_pretrain.py \\
        --manifest ~/Datasets/fma_metadata/fma_medium_tags_full_local.csv \\
        --epochs 30 --batch-size 64
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

from src.models.audio_classifier import AudioClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_fma_pretrain")


# ── dataset (label-free, returns mel only) ───────────────────────


class MelOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sample_rate: int = 16000,
                 n_mels: int = 64, max_duration: float = 6.0,
                 gcs_prefix: str | None = None):
        self.df = df.reset_index(drop=True)
        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = 1024
        self.hop_length = 512
        self.max_samples = int(sample_rate * max_duration)
        self.gcs_prefix = gcs_prefix

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, raw: str) -> str:
        if self.gcs_prefix and raw.startswith("/Volumes/"):
            tail = raw.split("/audio/fma_medium/", 1)
            if len(tail) == 2:
                return f"{self.gcs_prefix.rstrip('/')}/{tail[1]}"
        return raw

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._resolve(row["file_path"])
        try:
            import librosa
            y, _ = librosa.load(path, sr=self.sr, mono=True,
                                duration=self.max_samples / self.sr)
            if len(y) < self.max_samples // 2:
                y = np.pad(y, (0, self.max_samples // 2 - len(y)))
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
            mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
            return torch.from_numpy(mel_db).float().unsqueeze(0)
        except Exception as e:
            logger.debug("decode fail %s: %s", path, e)
            return torch.zeros(1, self.n_mels,
                               max(self.max_samples // self.hop_length, 8))


def collate_pad(batch):
    max_t = max(f.shape[-1] for f in batch)
    out = []
    for f in batch:
        pad = max_t - f.shape[-1]
        if pad > 0:
            f = F.pad(f, (0, pad))
        out.append(f)
    return torch.stack(out, dim=0)


# ── augmentation: two random views via SpecAugment + crop ────────


class TwoViewSpecAugment(nn.Module):
    """Returns two distinct augmented views of a batch [B, 1, F, T].
    Each view applies independent random freq-mask + time-mask + a
    random time-crop. Both views share the same input tensor so they're
    correlated views of the same underlying clip."""

    def __init__(self, freq_mask: int = 16, time_mask: int = 32,
                 crop_ratio: float = 0.7):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.crop_ratio = crop_ratio

    @staticmethod
    def _spec_aug(x, freq_mask, time_mask):
        B, C, F_, T = x.shape
        x = x.clone()
        if freq_mask > 0:
            f = torch.randint(0, freq_mask + 1, (B,), device=x.device)
            f0 = (torch.rand(B, device=x.device) * (F_ - f).clamp(min=1)).long()
            for i in range(B):
                if f[i] > 0:
                    x[i, :, f0[i]:f0[i] + f[i], :] = 0
        if time_mask > 0:
            t = torch.randint(0, time_mask + 1, (B,), device=x.device)
            t0 = (torch.rand(B, device=x.device) * (T - t).clamp(min=1)).long()
            for i in range(B):
                if t[i] > 0:
                    x[i, :, :, t0[i]:t0[i] + t[i]] = 0
        return x

    @staticmethod
    def _random_crop(x, crop_ratio):
        # Each item gets an independent random crop of crop_ratio*T frames,
        # then everything is right-padded back to max length so stacking works.
        B, C, F_, T = x.shape
        crop_len = max(int(T * crop_ratio), 16)
        crop_len = min(crop_len, T)
        starts = torch.randint(0, max(T - crop_len + 1, 1), (B,))
        out = []
        for i in range(B):
            out.append(x[i:i+1, :, :, starts[i]:starts[i] + crop_len])
        # Pad to longest (all crop_len, so same shape)
        return torch.cat(out, dim=0)

    def forward(self, x):
        v1 = self._spec_aug(self._random_crop(x, self.crop_ratio),
                            self.freq_mask, self.time_mask)
        v2 = self._spec_aug(self._random_crop(x, self.crop_ratio),
                            self.freq_mask, self.time_mask)
        return v1, v2


# ── SimSiam wrapper around AudioClassifier conv backbone ─────────


class SimSiam(nn.Module):
    """Encoder = AudioClassifier conv stack + GAP. Projection MLP +
    predictor MLP. Symmetric neg-cos loss with stop-gradient on target."""

    def __init__(self, n_mels: int = 64, proj_dim: int = 512,
                 hidden_dim: int = 512):
        super().__init__()
        # Borrow the conv stack from AudioClassifier and discard its head.
        cls = AudioClassifier(num_classes=2, n_mels=n_mels)
        self.encoder_conv = cls.conv_layers
        self.encoder_gap = cls.gap
        # Last conv channel count is channels[-1]; default 256.
        emb_dim = 256
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.proj_dim = proj_dim

    def encode(self, x):
        h = self.encoder_conv(x)
        h = self.encoder_gap(h).flatten(1)
        return h

    def forward(self, v1, v2):
        z1 = self.projector(self.encode(v1))
        z2 = self.projector(self.encode(v2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def neg_cos_sim(p, z):
    # p, z: [B, D]
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


# ── train ────────────────────────────────────────────────────────


def pick_device(force):
    if force:
        return torch.device(force)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=str, required=True,
                    help="CSV with at least a 'file_path' column. Labels ignored.")
    ap.add_argument("--gcs-audio-prefix", type=str, default=None)
    ap.add_argument("--subset-filter", choices=["all","small","medium"], default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--proj-dim", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-duration", type=float, default=6.0)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--freq-mask", type=int, default=16)
    ap.add_argument("--time-mask", type=int, default=32)
    ap.add_argument("--crop-ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--models-root", type=str,
                    default=os.environ.get("KELLY_MODEL_ROOT", str(Path.home()/"Models")))
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--vertex", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    df = pd.read_csv(args.manifest)
    if args.subset_filter == "small":
        df = df[df["subset"] == "small"]
    elif args.subset_filter == "medium":
        df = df[df["subset"] == "medium"]
    df = df.reset_index(drop=True)
    if args.smoke:
        df = df.head(200).reset_index(drop=True)
        args.epochs = 2
        args.batch_size = 8

    n_total = len(df)
    idx = np.arange(n_total)
    np.random.default_rng(args.seed).shuffle(idx)
    n_tr = int(n_total * 0.95)
    train_df = df.iloc[idx[:n_tr]].reset_index(drop=True)
    val_df = df.iloc[idx[n_tr:]].reset_index(drop=True)

    device = pick_device(args.device)

    ds_kwargs = dict(sample_rate=args.sample_rate, n_mels=args.n_mels,
                     max_duration=args.max_duration,
                     gcs_prefix=args.gcs_audio_prefix)
    train_set = MelOnlyDataset(train_df, **ds_kwargs)
    val_set = MelOnlyDataset(val_df, **ds_kwargs)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_pad,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_pad,
                            pin_memory=False, drop_last=True)

    model = SimSiam(n_mels=args.n_mels, proj_dim=args.proj_dim,
                    hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    aug = TwoViewSpecAugment(freq_mask=args.freq_mask, time_mask=args.time_mask,
                             crop_ratio=args.crop_ratio).to(device)

    total_steps = max(args.epochs * max(len(train_loader), 1), 1)
    warmup = min(args.warmup_steps, max(total_steps // 10, 1))
    import math as _math
    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + _math.cos(_math.pi * prog)))

    run_name = args.name or f"fma-pretrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.vertex and "AIP_MODEL_DIR" in os.environ:
        raw = os.environ["AIP_MODEL_DIR"]
        ckpt_dir = (Path("/gcs") / raw[len("gs://"):]) if raw.startswith("gs://") else Path(raw)
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
        "n_train": len(train_df), "n_val": len(val_df),
        "method": "SimSiam",
        "seed": args.seed,
        "hparams": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "proj_dim": args.proj_dim, "hidden_dim": args.hidden_dim,
            "n_mels": args.n_mels, "max_duration": args.max_duration,
            "freq_mask": args.freq_mask, "time_mask": args.time_mask,
            "crop_ratio": args.crop_ratio, "warmup_steps": warmup,
        },
        "torch_version": str(torch.__version__),
        "smoke": args.smoke, "vertex": args.vertex,
    }
    try:
        import yaml
        (ckpt_dir / "run_manifest.yaml").write_text(
            yaml.safe_dump(manifest_meta, sort_keys=False))
    except ImportError:
        (ckpt_dir / "run_manifest.yaml").write_text(json.dumps(manifest_meta, indent=2))

    logger.info("Run %s | ckpt=%s | n_train=%d n_val=%d",
                run_name, ckpt_dir, len(train_df), len(val_df))

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        s = 0.0; n = 0
        for x in loader:
            x = x.to(device)
            v1, v2 = aug(x)
            p1, p2, z1, z2 = model(v1, v2)
            loss = 0.5 * (neg_cos_sim(p1, z2) + neg_cos_sim(p2, z1))
            s += loss.item() * x.size(0); n += x.size(0)
        return s / max(n, 1)

    best_val = float("inf")
    global_step = 0
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0; n_seen = 0
        for x in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_step)
            x = x.to(device)
            v1, v2 = aug(x)
            p1, p2, z1, z2 = model(v1, v2)
            loss = 0.5 * (neg_cos_sim(p1, z2) + neg_cos_sim(p2, z1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * x.size(0); n_seen += x.size(0)
            global_step += 1
        train_loss = running / max(n_seen, 1)
        val_loss = evaluate(val_loader)
        logger.info("[epoch %d/%d] train_loss=%.4f val_loss=%.4f (more negative = better; floor=-1.0)",
                    epoch + 1, args.epochs, train_loss, val_loss)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                # Save just the conv backbone for downstream warm-start.
                "encoder_conv_state_dict": model.encoder_conv.state_dict(),
                "encoder_gap_state_dict": model.encoder_gap.state_dict(),
                "val_loss": val_loss,
                "config": manifest_meta["hparams"],
            }, ckpt_dir / "best.pt")
            logger.info("  ↳ new best val_loss=%.4f → %s", val_loss, ckpt_dir / "best.pt")

    writer.close()
    logger.info("Done. best_val=%.4f wall=%.0fs ckpt=%s",
                best_val, time.perf_counter() - t_start, ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
