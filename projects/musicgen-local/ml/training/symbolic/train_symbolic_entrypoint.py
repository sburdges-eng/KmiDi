#!/usr/bin/env python3
"""First symbolic training entrypoint using approved manifests and shared Trainer."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.training.shared.bootstrap_from_ml_training_suite.training.trainer import Trainer


DEFAULT_MIDI_MANIFEST = ROOT_DIR / "ml/data/manifests/active/midi-approved.txt"
DEFAULT_MODEL_MANIFEST = ROOT_DIR / "ml/data/manifests/active/models-approved.txt"
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "models/checkpoints/symbolic"
DEFAULT_LOG_DIR = ROOT_DIR / "logs/symbolic"
DEFAULT_RUNS_DIR = ROOT_DIR / "ops/runs"


@dataclass
class PreparedItem:
    path: Path
    label_name: str
    label_id: int
    feature: torch.Tensor


class SymbolicBytesDataset(Dataset):
    """A minimal symbolic dataset built from raw MIDI byte distributions."""

    def __init__(self, items: List[PreparedItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        return item.feature, item.label_id


class SymbolicClassifier(nn.Module):
    """Small MLP for symbolic feature vectors."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def infer_label_name(path: Path) -> str:
    """Infer a label from path structure for first-pass symbolic classification."""
    parts = path.parts

    if "MIDI_kaggle" in parts:
        idx = parts.index("MIDI_kaggle")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if "examples" in parts and "midi" in parts:
        idx = parts.index("midi")
        if idx + 1 < len(parts):
            return f"examples_{parts[idx + 1]}"

    if "CODE" in parts and "mid" in parts:
        return "code_mid"

    parent = path.parent.name.strip()
    return parent if parent else "unknown"


def midi_feature_from_bytes(path: Path, max_bytes: int = 32768) -> torch.Tensor:
    """Build a deterministic symbolic feature vector from raw MIDI bytes."""
    raw = path.read_bytes()[:max_bytes]

    if not raw:
        hist = torch.zeros(256, dtype=torch.float32)
        size_ratio = 0.0
        zero_ratio = 0.0
        return torch.cat([hist, torch.tensor([size_ratio, zero_ratio], dtype=torch.float32)])

    byte_vals = torch.tensor(list(raw), dtype=torch.long)
    hist = torch.bincount(byte_vals, minlength=256).to(torch.float32)
    hist = hist / hist.sum().clamp(min=1.0)

    size_ratio = min(path.stat().st_size, max_bytes) / float(max_bytes)
    zero_ratio = float((byte_vals == 0).to(torch.float32).mean().item())

    return torch.cat([hist, torch.tensor([size_ratio, zero_ratio], dtype=torch.float32)])


def load_manifest_paths(manifest_path: Path) -> List[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    paths: List[Path] = []
    for line in manifest_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        p = Path(stripped)
        if p.exists() and p.is_file():
            paths.append(p)

    if not paths:
        raise RuntimeError(f"Manifest has no usable file paths: {manifest_path}")

    return paths


def prepare_items(midi_paths: List[Path]) -> Tuple[List[PreparedItem], Dict[str, int]]:
    label_to_id: Dict[str, int] = {}
    items: List[PreparedItem] = []

    for path in midi_paths:
        label_name = infer_label_name(path)
        if label_name not in label_to_id:
            label_to_id[label_name] = len(label_to_id)

        label_id = label_to_id[label_name]
        feature = midi_feature_from_bytes(path)
        items.append(PreparedItem(path=path, label_name=label_name, label_id=label_id, feature=feature))

    return items, label_to_id


def compute_split_sizes(total: int) -> Tuple[int, int, int]:
    if total < 3:
        raise RuntimeError("Need at least 3 samples for train/val/test splits.")

    train = max(1, int(total * 0.8))
    val = max(1, int(total * 0.1))
    test = total - train - val

    if test < 1:
        test = 1
        if train > val:
            train -= 1
        else:
            val -= 1

    return train, val, test


def write_run_log(
    run_path: Path,
    experiment_name: str,
    midi_manifest: Path,
    model_manifest: Path,
    midi_count: int,
    model_count: int,
    label_count: int,
    split_sizes: Tuple[int, int, int],
    history: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    checkpoint_dir: Path,
    log_dir: Path,
    duration_seconds: float,
) -> None:
    train_size, val_size, test_size = split_sizes
    run_path.parent.mkdir(parents=True, exist_ok=True)

    final_train_loss = history["train_loss"][-1] if history.get("train_loss") else None
    final_val_loss = history["val_loss"][-1] if history.get("val_loss") else None
    final_train_acc = history["train_acc"][-1] if history.get("train_acc") else None
    final_val_acc = history["val_acc"][-1] if history.get("val_acc") else None

    run_path.write_text(
        "\n".join(
            [
                "# Symbolic Training Run",
                f"Date: {datetime.now().strftime('%Y-%m-%d')}",
                f"Experiment: {experiment_name}",
                "",
                "## Inputs",
                f"- MIDI manifest: {midi_manifest}",
                f"- Model manifest: {model_manifest}",
                f"- MIDI entries consumed: {midi_count}",
                f"- Model artifact entries referenced: {model_count}",
                "",
                "## Dataset",
                f"- Labels/classes inferred: {label_count}",
                f"- Split sizes (train/val/test): {train_size}/{val_size}/{test_size}",
                "",
                "## Metrics",
                f"- Final train loss: {final_train_loss}",
                f"- Final val loss: {final_val_loss}",
                f"- Final train accuracy: {final_train_acc}",
                f"- Final val accuracy: {final_val_acc}",
                f"- Test loss: {test_metrics['loss']}",
                f"- Test accuracy: {test_metrics['accuracy']}",
                "",
                "## Artifacts",
                f"- Checkpoints: {checkpoint_dir / experiment_name}",
                f"- Logs: {log_dir / experiment_name}",
                f"- Duration seconds: {duration_seconds:.2f}",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first-pass symbolic model from approved MIDI manifest")
    parser.add_argument("--midi-manifest", type=Path, default=DEFAULT_MIDI_MANIFEST)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()
    torch.manual_seed(args.seed)

    midi_paths = load_manifest_paths(args.midi_manifest)
    model_paths = load_manifest_paths(args.model_manifest)

    items, label_to_id = prepare_items(midi_paths)
    dataset = SymbolicBytesDataset(items)

    train_size, val_size, test_size = compute_split_sizes(len(dataset))
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    input_dim = len(items[0].feature)
    num_classes = len(label_to_id)

    model = SymbolicClassifier(input_dim=input_dim, num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    experiment_name = args.experiment_name or datetime.now().strftime("symbolic_%Y%m%d_%H%M%S")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        scheduler=scheduler,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        log_dir=DEFAULT_LOG_DIR,
        experiment_name=experiment_name,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=5,
        save_best=True,
    )

    best_checkpoint = DEFAULT_CHECKPOINT_DIR / experiment_name / "best.pt"
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)

    test_metrics = trainer.validate(test_loader)

    run_name = f"{datetime.now().strftime('%Y-%m-%d')}-symbolic-entrypoint-{experiment_name}.md"
    run_path = DEFAULT_RUNS_DIR / run_name

    write_run_log(
        run_path=run_path,
        experiment_name=experiment_name,
        midi_manifest=args.midi_manifest,
        model_manifest=args.model_manifest,
        midi_count=len(midi_paths),
        model_count=len(model_paths),
        label_count=len(label_to_id),
        split_sizes=(train_size, val_size, test_size),
        history=history,
        test_metrics=test_metrics,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        log_dir=DEFAULT_LOG_DIR,
        duration_seconds=time.time() - start,
    )

    print(json.dumps({
        "experiment": experiment_name,
        "midi_entries": len(midi_paths),
        "model_entries": len(model_paths),
        "labels": len(label_to_id),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "run_log": str(run_path),
    }, indent=2))


if __name__ == "__main__":
    main()
