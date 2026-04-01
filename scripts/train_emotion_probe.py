#!/usr/bin/env python3
"""
Train EmotionProbe on cached JEPA embeddings.

Loads pre-extracted embeddings (from extract_jepa_embeddings.py), trains a
small MLP probe to predict valence/arousal, and saves the best checkpoint.
At the end, evaluates on the held-out test set and reports MSE + Pearson r.

Usage:
    python scripts/train_emotion_probe.py
    python scripts/train_emotion_probe.py \\
        --embeddings data/jepa_emotion_embeddings.pt \\
        --output checkpoints/emotion_probe/best_probe.pt \\
        --epochs 50 --patience 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from music_brain.jepa.emotion_probe import EmotionProbe

logger = logging.getLogger(__name__)


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation coefficient between two 1-D tensors."""
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = (pred.norm() * target.norm()).clamp(min=1e-8)
    return float((pred * target).sum() / denom)


def evaluate(
    probe: EmotionProbe,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return MSE and Pearson r for valence/arousal on a DataLoader."""
    probe.eval()
    pred_v_list: list[torch.Tensor] = []
    pred_a_list: list[torch.Tensor] = []
    tgt_v_list: list[torch.Tensor] = []
    tgt_a_list: list[torch.Tensor] = []
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, targets in loader:
            x, targets = x.to(device), targets.to(device)
            preds = probe(x)
            loss = mse_loss(preds, targets)
            total_loss += loss.item()
            n_batches += 1
            pred_v_list.append(preds[:, 0].cpu())
            pred_a_list.append(preds[:, 1].cpu())
            tgt_v_list.append(targets[:, 0].cpu())
            tgt_a_list.append(targets[:, 1].cpu())

    pred_v = torch.cat(pred_v_list)
    pred_a = torch.cat(pred_a_list)
    tgt_v = torch.cat(tgt_v_list)
    tgt_a = torch.cat(tgt_a_list)

    return {
        "mse": total_loss / max(n_batches, 1),
        "valence_mse": float(nn.MSELoss()(pred_v, tgt_v)),
        "arousal_mse": float(nn.MSELoss()(pred_a, tgt_a)),
        "valence_r": pearson_r(pred_v, tgt_v),
        "arousal_r": pearson_r(pred_a, tgt_a),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train EmotionProbe on cached JEPA embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/jepa_emotion_embeddings.pt"),
        help="Cached embeddings .pt file (default: data/jepa_emotion_embeddings.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/emotion_probe/best_probe.pt"),
        help="Best checkpoint output path (default: checkpoints/emotion_probe/best_probe.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Probe hidden dim (default: 128)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve paths
    emb_path = (
        args.embeddings if args.embeddings.is_absolute() else Path(_PROJECT_ROOT) / args.embeddings
    )
    out_path = (
        args.output if args.output.is_absolute() else Path(_PROJECT_ROOT) / args.output
    )

    if not emb_path.exists():
        logger.error("Embeddings file not found: %s", emb_path)
        return 1

    # Load cached embeddings
    logger.info("Loading embeddings from %s", emb_path)
    cache = torch.load(str(emb_path), map_location="cpu", weights_only=False)
    embeddings: torch.Tensor = cache["embeddings"]    # (N, latent_dim)
    valence: torch.Tensor = cache["valence"]           # (N,)
    arousal: torch.Tensor = cache["arousal"]           # (N,)
    splits: list[str] = cache["splits"]

    latent_dim = embeddings.shape[1]
    targets = torch.stack([valence, arousal], dim=1)   # (N, 2)

    # Build split masks
    split_arr = [s for s in splits]
    train_mask = torch.tensor([s == "train" for s in split_arr])
    val_mask = torch.tensor([s == "val" for s in split_arr])
    test_mask = torch.tensor([s == "test" for s in split_arr])

    X_train, y_train = embeddings[train_mask], targets[train_mask]
    X_val, y_val = embeddings[val_mask], targets[val_mask]
    X_test, y_test = embeddings[test_mask], targets[test_mask]

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    probe = EmotionProbe(latent_dim=latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        probe.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(probe(x), y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        val_metrics = evaluate(probe, val_loader, device)
        val_loss = val_metrics["mse"]

        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
            epoch, args.epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "probe": probe.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "latent_dim": latent_dim,
                    "hidden_dim": args.hidden_dim,
                },
                str(out_path),
            )
            logger.info("  -> New best saved (val_loss=%.6f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d, best epoch=%d)",
                    epoch, args.patience, best_epoch,
                )
                break

    # Evaluate on test set using best checkpoint
    logger.info("Loading best checkpoint from %s (epoch %d)", out_path, best_epoch)
    best_ckpt = torch.load(str(out_path), map_location="cpu", weights_only=False)
    probe.load_state_dict(best_ckpt["probe"])
    probe.to(device)

    test_metrics = evaluate(probe, test_loader, device)

    print(f"\n=== Test Set Results (best epoch {best_epoch}) ===")
    print(f"  MSE (combined):  {test_metrics['mse']:.6f}")
    print(
        f"  Valence — MSE: {test_metrics['valence_mse']:.6f}"
        f"   r: {test_metrics['valence_r']:.4f}"
    )
    print(
        f"  Arousal — MSE: {test_metrics['arousal_mse']:.6f}"
        f"   r: {test_metrics['arousal_r']:.4f}"
    )
    print(f"  Checkpoint: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
