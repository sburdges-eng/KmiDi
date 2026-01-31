#!/usr/bin/env python3
"""
Optimized emotion classifier training for MacBook Pro M2/M4 (16GB).

Uses Metal (MPS) on Apple Silicon or CUDA on NVIDIA. Config-driven from
configs/exp_m2_m4_emotion.yaml. Low-latency compact models.

Usage:
  # From repo root, with tmux for long runs:
  tmux new -s kmidi_train
  micromamba activate kmidi-training   # or your env
  python -m KmiDi_CANON.training.train_emotion_optimized --config configs/exp_m2_m4_emotion.yaml
  python -m KmiDi_CANON.training.train_emotion_optimized --config configs/exp_m2_m4_emotion.yaml --data-dir ~/Datasets/Emotion_Instrument_Library/sub

  # Override device:
  python -m KmiDi_CANON.training.train_emotion_optimized --config configs/exp_m2_m4_emotion.yaml --device cuda  # NVIDIA

Governance: DATA LAW — checkpoints/logs to ~/Models. Run in tmux for critical jobs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure repo root and ml-training-suite on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ML_SUITE = REPO_ROOT / "ML_TRAINED_MODELS" / "ml" / "ml-training-suite"
for p in (str(REPO_ROOT), str(ML_SUITE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).expanduser().resolve()


def _get_device(cfg: dict, override: str | None) -> tuple:
    """Return (device, use_amp, num_workers)."""
    import torch
    dev = override or cfg.get("device", "auto")
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    device = torch.device(dev)
    use_amp = dev in ("cuda", "mps") and cfg.get("precision", "fp32") in ("fp16", "bf16")
    num_workers = cfg.get("data", {}).get("num_workers", 0 if dev == "mps" else 4)
    return device, use_amp, num_workers


def _find_audio_files(root: Path, exts: list[str]) -> tuple[list, list, dict]:
    """Recursive find; parent dir = label."""
    root = Path(root)
    audio_files, labels, class_to_idx = [], [], {}
    for ext in exts:
        for f in root.rglob(f"*{ext}"):
            if "venv" in str(f) or "site-packages" in str(f):
                continue
            rel = f.relative_to(root)
            emotion = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            if emotion not in class_to_idx:
                class_to_idx[emotion] = len(class_to_idx)
            audio_files.append(f)
            labels.append(class_to_idx[emotion])
    return audio_files, labels, class_to_idx


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    ckpts = list(output_dir.glob("checkpoint_*.pt")) + list(output_dir.glob("best.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimized emotion training (MPS/CUDA)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data-dir", default=None, help="Override dataset path")
    parser.add_argument("--device", default=None, help="Override device: auto|mps|cuda|cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    cfg = _load_config(str(config_path))
    paths = cfg.get("paths", {})
    checkpoint_dir = _resolve_path(paths.get("checkpoint_dir", "~/Models/checkpoints/exp_m2_m4_emotion"))
    log_dir = _resolve_path(paths.get("log_dir", "~/Models/logs/exp_m2_m4_emotion"))
    data_dir = _resolve_path(args.data_dir or paths.get("dataset", "~/Datasets/kmidi_emotion"))
    exp_name = cfg.get("experiment", "exp_m2_m4_emotion")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error("Data dir not found: %s", data_dir)
        logger.error("  Set paths.dataset in config or --data-dir. Example: ~/Datasets/Emotion_Instrument_Library/sub")
        return 1

    # Device
    import torch
    device, use_amp, num_workers = _get_device(cfg, args.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        logger.info("  Apple Metal (MPS)")
    logger.info(f"Mixed precision (AMP): {use_amp}")
    logger.info(f"Data workers: {num_workers}")

    # Imports (after path setup)
    from src.data import AudioDataset
    from src.models import AudioClassifier
    from src.training import Trainer

    lora_cfg = cfg.get("model", {}).get("lora", {})
    use_lora = lora_cfg.get("enabled", False)
    if use_lora:
        from src.models.lora import apply_lora_to_model, get_lora_params, save_lora_weights, count_lora_params

    # Data
    exts = cfg.get("data", {}).get("extensions", [".mp3", ".wav", ".aif", ".aiff", ".flac"])
    audio_files, labels, class_to_idx = _find_audio_files(data_dir, exts)
    num_classes = len(class_to_idx)

    if not audio_files:
        logger.error("No audio files in %s", data_dir)
        return 1

    logger.info(f"\nData: {len(audio_files)} files, {num_classes} classes")
    for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        count = sum(1 for l in labels if l == idx)
        logger.info(f"  {name}: {count}")

    data_cfg = cfg.get("data", {})
    dataset = AudioDataset(
        audio_paths=audio_files,
        labels=labels,
        sample_rate=data_cfg.get("sample_rate", 22050),
        duration=data_cfg.get("duration", 3.0),
        n_mels=data_cfg.get("n_mels", 96),
    )
    dataset.class_to_idx = class_to_idx
    dataset.idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 16)
    grad_accum = train_cfg.get("grad_accum_steps", 1)

    # Use pin_memory only for CUDA (MPS can have issues)
    torch.manual_seed(cfg.get("seed", 1337))
    total = len(dataset)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    pin_mem = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
    )

    # Model
    model_cfg = cfg.get("model", {})
    channels = model_cfg.get("channels", [24, 48, 96, 192])
    model = AudioClassifier(
        num_classes=num_classes,
        n_mels=data_cfg.get("n_mels", 96),
        channels=channels,
        dropout=model_cfg.get("dropout", 0.3),
    )

    if use_lora:
        model = apply_lora_to_model(
            model,
            rank=lora_cfg.get("rank", 4),
            alpha=lora_cfg.get("alpha", 8.0),
            dropout=0.1,
            target_modules=lora_cfg.get("target_modules", ["fc"]),
        )
        info = count_lora_params(model)
        logger.info(f"LoRA: {info['trainable']:,} trainable params ({info['trainable_pct']:.1f}%)")
    else:
        logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)

    # Optimizer
    opt_cfg = cfg.get("optimizer", {})
    lr = opt_cfg.get("lr", 8e-4)
    if use_lora:
        params = get_lora_params(model)
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=opt_cfg.get("weight_decay", 0.01))

    sch_cfg = cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sch_cfg.get("mode", "min"),
        factor=sch_cfg.get("factor", 0.5),
        patience=sch_cfg.get("patience", 5),
        min_lr=sch_cfg.get("min_lr", 1e-6),
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Save class names
    (checkpoint_dir / exp_name).mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / exp_name / "class_names.json", "w") as f:
        json.dump(list(class_to_idx.keys()), f, indent=2)

    # Trainer with gradient accumulation (custom loop for AMP + grad accum)
    # Use standard Trainer but we'll wrap train_epoch for AMP + grad_accum
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        scheduler=scheduler,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        experiment_name=exp_name,
    )

    # Custom train_epoch for AMP + gradient accumulation (when needed)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    def train_epoch_amp(loader):
        from tqdm import tqdm
        trainer.model.train()
        total_loss, correct, total = 0.0, 0, 0
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and use_amp else None
        pbar = tqdm(loader, desc=f"Epoch {trainer.current_epoch + 1} [Train]")
        optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            with torch.amp.autocast(device.type if device.type in ("cuda", "mps") else "cpu", enabled=use_amp):
                out = trainer.model(data)
                loss = trainer.criterion(out, target) / grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            pbar.set_postfix(loss=total_loss / (batch_idx + 1), acc=100.0 * correct / total)

        return {"loss": total_loss / len(loader), "accuracy": 100.0 * correct / total}

    if use_amp or grad_accum > 1:
        trainer.train_epoch = train_epoch_amp

    # Resume
    ckpt = None
    if train_cfg.get("auto_resume", True) or args.resume:
        ckpt = _find_latest_checkpoint(checkpoint_dir / exp_name)
        if ckpt:
            logger.info("Resume from %s", ckpt)
            trainer.load_checkpoint(str(ckpt))

    # Train
    logger.info("\n" + "=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.get("epochs", 50),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 12),
    )

    if use_lora:
        lora_path = checkpoint_dir / exp_name / "lora_weights.pt"
        save_lora_weights(model, str(lora_path))
        logger.info(f"LoRA weights: {lora_path}")

    # Test
    logger.info("\n" + "=" * 60)
    logger.info("Test evaluation")
    logger.info("=" * 60)
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%")

    # Manifest
    manifest = {
        "experiment": exp_name,
        "config": str(config_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir),
        "log_dir": str(log_dir),
        "data_dir": str(data_dir),
        "device": str(device),
        "seed": cfg.get("seed", 1337),
        "num_classes": num_classes,
        "test_accuracy": test_metrics["accuracy"],
        "timestamp": datetime.now().isoformat(),
    }
    manifest_path = checkpoint_dir / exp_name / "manifest_run.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nManifest: {manifest_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
