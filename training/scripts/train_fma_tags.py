#!/usr/bin/env python3
"""FMA multi-label tags classifier over the same 106-subgenre vocabulary.

Target per track = multi-hot over the 106 subgenres. Positive labels
come from FMA's ``genres_all`` hierarchy expansion intersected with our
106-class vocabulary — e.g. a Thrash track's positives are {Thrash, Metal,
Rock} simultaneously.

Differences from train_fma_subgenre.py:
- Manifest column is ``target_ids`` (comma-separated genre_ids)
- Loss = BCEWithLogitsLoss with pos_weight for class imbalance
- No balanced-sampler option (multi-label has no single class per sample)
- Val metrics: macro-AUROC + macro-F1@thr + exact-match + hamming
- Best ckpt selected on macro-AUROC (threshold-free, robust)

Usage::

    python training/scripts/train_fma_tags.py --smoke
    python training/scripts/train_fma_tags.py \\
        --manifest ~/Datasets/fma_metadata/fma_medium_tags_uploaded.csv \\
        --init-from ~/Models/.../fma-genre-.../best.pt \\
        --epochs 40
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
logger = logging.getLogger("train_fma_tags")

LABEL_COL = "target_ids"  # comma-separated genre_ids in the manifest


# ── dataset ──────────────────────────────────────────────────────


class FMADataset(Dataset):
    """Loads mp3s, computes mel spectrogram (librosa, CPU), returns
    (features [1, n_mels, T], multi_hot_target [num_classes])."""

    def __init__(self, manifest_df: pd.DataFrame, label_to_id: dict[int, int],
                 sample_rate: int = 16000, n_mels: int = 64,
                 max_duration: float = 6.0, gcs_prefix: str | None = None,
                 mel_cache_root: str | None = None):
        self.df = manifest_df.reset_index(drop=True)
        self.label_to_id = label_to_id  # genre_id (int) → index
        self.num_classes = len(label_to_id)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = 1024
        self.hop_length = 512
        self.max_samples = int(sample_rate * max_duration)
        self.gcs_prefix = gcs_prefix
        self.mel_cache_root = mel_cache_root

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, raw_path: str) -> str:
        if self.gcs_prefix and raw_path.startswith("/Volumes/"):
            tail = raw_path.split("/audio/fma_medium/", 1)
            if len(tail) == 2:
                return f"{self.gcs_prefix.rstrip('/')}/{tail[1]}"
        return raw_path

    def _multi_hot(self, raw_ids: str) -> torch.Tensor:
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for tok in str(raw_ids).split(","):
            tok = tok.strip()
            if not tok:
                continue
            gid = int(tok)
            if gid in self.label_to_id:
                y[self.label_to_id[gid]] = 1.0
        return y


    def _try_cache(self, track_id):
        if not self.mel_cache_root:
            return None
        sub = f"{int(track_id):06d}"[:3]
        p = Path(self.mel_cache_root) / "fma_medium" / sub / f"{int(track_id):06d}.npy"
        if p.exists():
            try:
                arr = np.load(str(p)).astype(np.float32)
                return torch.from_numpy(arr).unsqueeze(0)
            except Exception:
                return None
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        target = self._multi_hot(row[LABEL_COL])
        cached = self._try_cache(row["track_id"]) if "track_id" in row else None
        if cached is not None:
            return cached, target
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
            feat = torch.from_numpy(mel_db).float().unsqueeze(0)
            return feat, target
        except Exception as e:
            logger.debug("decode fail %s: %s", path, e)
            return (torch.zeros(1, self.n_mels,
                                max(self.max_samples // self.hop_length, 8)),
                    target)


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


def train_one_epoch(model, loader, optimizer, criterion, device, desc, spec_aug=None,
                    lr_at=None, step_ref=None):
    model.train()
    running = 0.0
    n = 0
    for feats, labels in tqdm(loader, desc=desc, leave=False):
        if lr_at is not None and step_ref is not None:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step_ref[0])
            step_ref[0] += 1
        feats, labels = feats.to(device), labels.to(device)
        if spec_aug is not None:
            feats = spec_aug(feats)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * feats.size(0)
        n += feats.size(0)
    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, threshold: float = 0.5):
    """Multi-label eval: BCE loss + macro AUROC + macro F1@thr + exact-match
    + hamming. Returns dict."""
    model.eval()
    running = 0.0
    n = 0
    all_logits = []
    all_targets = []
    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        logits = model(feats)
        loss = criterion(logits, targets)
        running += loss.item() * feats.size(0)
        n += feats.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()  # [N, C]
    targets = torch.cat(all_targets, dim=0).numpy().astype(int)  # [N, C]
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Macro AUROC (per class, averaged) — skip classes with 0 or all positives
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    per_class_auroc = []
    per_class_ap = []
    for c in range(num_classes):
        if 0 < targets[:, c].sum() < targets.shape[0]:
            per_class_auroc.append(roc_auc_score(targets[:, c], probs[:, c]))
            per_class_ap.append(average_precision_score(targets[:, c], probs[:, c]))
        else:
            per_class_auroc.append(float("nan"))
            per_class_ap.append(float("nan"))
    valid_auroc = [a for a in per_class_auroc if not np.isnan(a)]
    valid_ap = [a for a in per_class_ap if not np.isnan(a)]
    macro_auroc = float(np.mean(valid_auroc)) if valid_auroc else 0.0
    macro_ap = float(np.mean(valid_ap)) if valid_ap else 0.0

    preds = (probs >= threshold).astype(int)
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    hamming = float(np.mean(preds != targets))
    exact_match = float(np.mean(np.all(preds == targets, axis=1)))

    return {
        "loss": running / max(n, 1),
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "hamming": hamming,
        "exact_match": exact_match,
        "per_class_auroc": per_class_auroc,
        "per_class_ap": per_class_ap,
    }


# ── main ────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=str,
                    default=str(Path.home() / "Datasets" / "fma_metadata" / "fma_medium_manifest.csv"))
    ap.add_argument("--gcs-audio-prefix", type=str, default=None,
                    help="If set, rewrite manifest /Volumes/... paths to <prefix>/<sub>/<file>.mp3")
    ap.add_argument("--mel-cache-root", type=str, default=None,
                    help="Pre-computed mel .npy cache root (produced by build_mel_cache.py).")
    ap.add_argument("--subset-filter", choices=["all", "small", "medium"], default="all",
                    help="all = small+medium, medium = medium-only, small = small-only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=200,
                    help="Linear LR warmup steps. 0 disables.")
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
    ap.add_argument("--init-from", type=str, default=None,
                    help="Path to a genre/subgenre best.pt; loads its conv "
                         "backbone into the tags model. The fc head is "
                         "reinitialized for the new num_classes.")
    ap.add_argument("--pos-weight-cap", type=float, default=20.0,
                    help="Max pos_weight per class for BCEWithLogitsLoss. "
                         "Caps extreme values from tiny tail classes.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for macro-F1 / hamming / "
                         "exact-match metrics. Macro AUROC ignores it.")
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
        df = df.head(300).reset_index(drop=True)
        args.epochs = 2
        args.batch_size = 8

    # Multi-label vocabulary: union of every target_ids entry in the manifest.
    def parse_ids(s):
        return [int(t) for t in str(s).split(",") if t.strip()]
    all_ids = set()
    for raw in df[LABEL_COL]:
        all_ids.update(parse_ids(raw))
    classes = sorted(all_ids)  # list[int] — genre IDs
    label_to_id = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    device = pick_device(args.device)

    # Multi-label random 80/10/10 split (no per-class stratification — a
    # track has multiple labels, so single-class strata don't apply).
    n_total = len(df)
    idx = np.arange(n_total)
    np.random.default_rng(args.seed).shuffle(idx)
    n_tr = int(n_total * 0.8)
    n_va = int(n_total * 0.1)
    train_df = df.iloc[idx[:n_tr]].reset_index(drop=True)
    val_df = df.iloc[idx[n_tr:n_tr + n_va]].reset_index(drop=True)

    # Per-class positive counts on TRAIN for pos_weight
    pos_counts = np.zeros(num_classes, dtype=np.int64)
    for raw in train_df[LABEL_COL]:
        for gid in parse_ids(raw):
            if gid in label_to_id:
                pos_counts[label_to_id[gid]] += 1
    neg_counts = max(len(train_df), 1) - pos_counts

    ds_kwargs = dict(label_to_id=label_to_id, sample_rate=args.sample_rate,
                     n_mels=args.n_mels, max_duration=args.max_duration,
                     gcs_prefix=args.gcs_audio_prefix,
                     mel_cache_root=args.mel_cache_root)
    train_set = FMADataset(train_df, **ds_kwargs)
    val_set = FMADataset(val_df, **ds_kwargs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_pad, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_pad,
                            pin_memory=False)

    model = AudioClassifier(num_classes=num_classes, n_mels=args.n_mels).to(device)

    # Optionally warm-start the conv backbone from a pretrained genre model.
    if args.init_from:
        from pathlib import Path as _Path
        src = _Path(args.init_from).expanduser()
        if src.exists():
            ck = torch.load(str(src), map_location="cpu", weights_only=False)
            sd_src = ck["model_state_dict"]
            sd_dst = model.state_dict()
            loaded, skipped = [], []
            for k, v in sd_src.items():
                # Skip the final fc head — it's sized for a different
                # num_classes and would either not match or contaminate.
                if k.startswith("fc.") or k.startswith("head.") or k.endswith(".fc.weight"):
                    skipped.append(k)
                    continue
                if k in sd_dst and sd_dst[k].shape == v.shape:
                    sd_dst[k] = v
                    loaded.append(k)
                else:
                    skipped.append(k)
            model.load_state_dict(sd_dst)
            logger.info("init-from %s: loaded %d tensors, skipped %d (head). Source val_acc=%.3f",
                        src, len(loaded), len(skipped), ck.get("val_acc", float("nan")))
        else:
            logger.warning("init-from path not found: %s — random init.", src)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Multi-label BCE. pos_weight compensates for the heavy negative skew
    # (avg 2.36 positives per 106-class vector → ~44 neg : 1 pos).
    pos_weight_vec = np.clip(neg_counts / np.maximum(pos_counts, 1),
                             a_min=1.0, a_max=args.pos_weight_cap)
    pos_weight_t = torch.tensor(pos_weight_vec, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    # LR schedule: linear warmup → cosine decay to 10% of peak.
    total_steps = max(args.epochs * len(train_loader), 1)
    warmup = min(args.warmup_steps, max(total_steps // 10, 1))
    import math as _math
    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + _math.cos(_math.pi * progress)))
    global_step = 0

    spec_aug = None
    if not args.no_spec_aug and not args.smoke:
        spec_aug = _SpecAugment(
            freq_mask=args.freq_mask, time_mask=args.time_mask).to(device)

    run_name = args.name or f"fma-subgenre-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.vertex and "AIP_MODEL_DIR" in os.environ:
        # Vertex sets this as a gs:// URI; rewrite to a /gcs/ Fuse path so
        # Path()-based mkdir + write actually persists to the bucket. Without
        # this rewrite, Path() collapses gs:// → gs:/ and the script writes
        # to a useless local directory that gets nuked on worker shutdown.
        raw = os.environ["AIP_MODEL_DIR"]
        if raw.startswith("gs://"):
            ckpt_dir = Path("/gcs") / raw[len("gs://"):]
        else:
            ckpt_dir = Path(raw)
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
        "label_column": LABEL_COL,
        "class_ids": [int(c) for c in classes],
        "class_id_to_name": (
            df.drop_duplicates(LABEL_COL).set_index(LABEL_COL)["subgenre_name"].astype(str).to_dict()
            if "subgenre_name" in df.columns else None
        ),
        "class_pos_counts_train": {int(c): int(pos_counts[i])
                                   for c, i in label_to_id.items()},
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
            "pos_weight_cap": args.pos_weight_cap,
            "threshold": args.threshold,
            "init_from": args.init_from,
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
    (ckpt_dir / "class_names.json").write_text(json.dumps(
        [int(c) for c in classes], indent=2))

    logger.info("Run %s | ckpt=%s | n_train=%d n_val=%d | classes=%d",
                run_name, ckpt_dir, len(train_df), len(val_df), num_classes)

    best_metric = 0.0
    best_epoch = -1
    epochs_since_best = 0
    t_start = time.perf_counter()

    step_ref = [global_step]
    for epoch in range(args.epochs):
        desc = f"epoch {epoch + 1}/{args.epochs}"
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, desc, spec_aug=spec_aug,
                                     lr_at=lr_at, step_ref=step_ref)
        m = validate(model, val_loader, criterion, device, num_classes,
                     threshold=args.threshold)
        logger.info(
            "[%s] train_loss=%.4f  val_loss=%.4f  AUROC=%.3f  mAP=%.3f  F1=%.3f  exact=%.3f",
            desc, train_loss, m["loss"], m["macro_auroc"], m["macro_ap"],
            m["macro_f1"], m["exact_match"],
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", m["loss"], epoch)
        writer.add_scalar("auroc/val", m["macro_auroc"], epoch)
        writer.add_scalar("ap/val", m["macro_ap"], epoch)
        writer.add_scalar("f1/val", m["macro_f1"], epoch)
        writer.add_scalar("exact_match/val", m["exact_match"], epoch)

        if m["macro_auroc"] > best_metric:  # AUROC is threshold-free
            best_metric = m["macro_auroc"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes, "class_ids": [int(c) for c in classes],
                "macro_auroc": m["macro_auroc"],
                "macro_ap": m["macro_ap"],
                "macro_f1": m["macro_f1"],
                "exact_match": m["exact_match"],
                "hamming": m["hamming"],
                "per_class_auroc": m["per_class_auroc"],
                "per_class_ap": m["per_class_ap"],
            }, ckpt_dir / "best.pt")
            logger.info("  ↳ new best AUROC=%.4f (AP=%.4f F1=%.3f exact=%.3f) → %s",
                        m["macro_auroc"], m["macro_ap"], m["macro_f1"],
                        m["exact_match"], ckpt_dir / "best.pt")
        else:
            epochs_since_best += 1
            if args.patience > 0 and epochs_since_best >= args.patience:
                logger.info("Early stop @ epoch %d, best AUROC=%.4f @ epoch %d",
                            epoch + 1, best_metric, best_epoch + 1)
                break

    writer.close()
    elapsed = time.perf_counter() - t_start
    logger.info("Done. best AUROC=%.4f @ epoch %d wall=%.0fs ckpt=%s",
                best_metric, best_epoch + 1, elapsed, ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
