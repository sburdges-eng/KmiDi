#!/usr/bin/env python3
"""
Optimized Emotion Classifier Training (Phase 5).

Builds on existing training/scripts/train_emotion.py with:
- ModelRegistry integration for checkpoint management
- Support for ~/Datasets/kmidi_emotion convention
- Auto-registration of trained models
- ONNX export for C++ inference

Usage:
    python3 training/scripts/train_emotion_optimized.py
    python3 training/scripts/train_emotion_optimized.py --data-dir ~/Datasets/kmidi_emotion
    python3 training/scripts/train_emotion_optimized.py --export-onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_emotion")

try:
    import torch
    import torch.nn as nn
except ImportError:
    logger.error("PyTorch not installed")
    sys.exit(1)


def find_data_dir() -> Path:
    """Find the emotion dataset directory using project conventions."""
    candidates = [
        Path.home() / "Datasets" / "kmidi_emotion",
        Path("/Volumes/sbdrive/Emotion_Instrument_Library/sub"),
        _REPO_ROOT / "data" / "emotion",
        _REPO_ROOT / "datasets" / "emotion",
    ]
    # Also check env var
    env_dir = os.environ.get("KMIDI_EMOTION_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]  # Return first convention even if missing


def find_audio_files(root_dir: Path):
    """Find audio files with emotion labels from directory structure."""
    audio_files = []
    labels = []
    class_to_idx = {}

    extensions = {".mp3", ".wav", ".aif", ".aiff", ".flac", ".ogg"}

    for audio_file in root_dir.rglob("*"):
        if audio_file.suffix.lower() not in extensions:
            continue
        if "venv" in str(audio_file) or "site-packages" in str(audio_file):
            continue

        relative = audio_file.relative_to(root_dir)
        if len(relative.parts) > 1:
            emotion = relative.parts[0]
        else:
            emotion = "unknown"

        if emotion not in class_to_idx:
            class_to_idx[emotion] = len(class_to_idx)

        audio_files.append(audio_file)
        labels.append(class_to_idx[emotion])

    return audio_files, labels, class_to_idx


def register_model(
    name: str,
    checkpoint_path: str,
    num_classes: int,
    class_names: list[str],
) -> None:
    """Register trained emotion model in ModelRegistry."""
    try:
        from music_brain.penta_core.ml.model_registry import (
            ModelInfo,
            ModelBackend,
            ModelTask,
            get_registry,
        )

        model_info = ModelInfo(
            name=name,
            task=ModelTask.EMOTION_CLASSIFICATION,
            backend=ModelBackend.PYTORCH,
            path=checkpoint_path,
            version="1.0.0",
            sample_rate=22050,
            output_shape=[num_classes],
            description=f"Emotion classifier ({num_classes} classes), trained {datetime.now().isoformat()}",
            tags=["emotion", "audio", "auto-registered"],
        )

        registry = get_registry()
        registry.register(model_info)

        # Save model_info.json for discovery
        info_path = Path(checkpoint_path).parent / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info.to_dict(), f, indent=2)

        # Save class names alongside
        classes_path = Path(checkpoint_path).parent / "class_names.json"
        with open(classes_path, "w") as f:
            json.dump(class_names, f, indent=2)

        logger.info("Registered model '%s' (%d classes)", name, num_classes)
    except Exception:
        logger.warning("Could not register in ModelRegistry (non-fatal)", exc_info=True)


def export_onnx(model: nn.Module, output_path: str, n_mels: int = 128) -> bool:
    """Export model to ONNX for C++ inference."""
    try:
        model.eval()
        dummy = torch.randn(1, 1, n_mels, 128)
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["mel_spectrogram"],
            output_names=["emotion_logits"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch", 3: "time"},
                "emotion_logits": {0: "batch"},
            },
            opset_version=17,
        )
        logger.info("ONNX exported to %s", output_path)
        return True
    except Exception:
        logger.warning("ONNX export failed", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Train optimized emotion classifier")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Emotion dataset directory (auto-detected if omitted)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Audio clip duration in seconds")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export best model to ONNX")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(_REPO_ROOT / "checkpoints" / "emotion"))
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    # Find data
    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    if not data_dir.exists():
        logger.error("Dataset not found at %s", data_dir)
        logger.info("Expected: ~/Datasets/kmidi_emotion/ with emotion-labeled subdirectories")
        sys.exit(1)

    exp_name = args.name or f"emotion_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = Path(args.checkpoint_dir) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Emotion Training — data=%s, ckpt=%s", data_dir, ckpt_dir)

    # Load data
    audio_files, labels, class_to_idx = find_audio_files(data_dir)
    num_classes = len(class_to_idx)
    logger.info("Found %d files, %d classes: %s", len(audio_files), num_classes, list(class_to_idx.keys()))

    if not audio_files:
        logger.error("No audio files found in %s", data_dir)
        sys.exit(1)

    # Import training modules
    from src.data import AudioDataset, create_dataloaders
    from src.models import AudioClassifier
    from src.training import Trainer

    # Device
    device = torch.device(
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    # Dataset
    dataset = AudioDataset(
        audio_paths=audio_files,
        labels=labels,
        sample_rate=22050,
        duration=args.duration,
        n_mels=128,
    )

    train_loader, val_loader, _ = create_dataloaders(
        dataset, batch_size=args.batch_size, train_split=0.8, val_split=0.1, num_workers=0,
    )

    # Model
    model = AudioClassifier(num_classes=num_classes, n_mels=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(ckpt_dir),
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.2f%%",
            epoch + 1, args.epochs, train_loss, val_loss, val_acc * 100,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = str(ckpt_dir / "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
            }, best_path)
            logger.info("New best: %.2f%% → %s", val_acc * 100, best_path)

    # Register
    best_path = str(ckpt_dir / "best.pt")
    if Path(best_path).exists():
        register_model(
            name=exp_name,
            checkpoint_path=best_path,
            num_classes=num_classes,
            class_names=list(class_to_idx.keys()),
        )

        # ONNX export
        if args.export_onnx:
            onnx_path = str(ckpt_dir / "model.onnx")
            checkpoint = torch.load(best_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            export_onnx(model, onnx_path)

    logger.info("Training complete. Best accuracy: %.2f%%", best_acc * 100)


if __name__ == "__main__":
    main()
