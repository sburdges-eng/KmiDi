#!/usr/bin/env python3
"""Training script for emotion classification using labeled directory structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import json

from src.data import AudioDataset, create_dataloaders
from src.models import AudioClassifier
from src.models.lora import (
    apply_lora_to_model,
    get_lora_params,
    save_lora_weights,
    count_lora_params,
)
from src.training import Trainer


def find_audio_files_recursive(root_dir, extensions=[".mp3", ".wav", ".aif", ".aiff", ".flac"]):
    """Find all audio files recursively, using parent directory as label."""
    root_dir = Path(root_dir)
    audio_files = []
    labels = []
    class_to_idx = {}

    # Find all subdirectories that contain audio files
    for ext in extensions:
        for audio_file in root_dir.rglob(f"*{ext}"):
            # Skip venv/test files
            if "venv" in str(audio_file) or "site-packages" in str(audio_file):
                continue

            # Get the emotion label (first subdirectory under root)
            relative = audio_file.relative_to(root_dir)
            if len(relative.parts) > 1:
                emotion = relative.parts[0]  # e.g., "ANXIETY", "JOY", etc.
            else:
                emotion = "unknown"

            if emotion not in class_to_idx:
                class_to_idx[emotion] = len(class_to_idx)

            audio_files.append(audio_file)
            labels.append(class_to_idx[emotion])

    return audio_files, labels, class_to_idx


def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--data-dir", type=str,
                       default="/Volumes/sbdrive/Emotion_Instrument_Library/sub",
                       help="Directory with emotion-labeled audio")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--duration", type=float, default=3.0, help="Audio duration (seconds)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--lora-only", action="store_true", help="Train only LoRA params")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()

    # Experiment name
    exp_name = args.name or f"emotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print("Emotion Classifier Training")
    print("=" * 60)
    print(f"\nExperiment: {exp_name}")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Find audio files with emotion labels
    print(f"\nLoading data from: {args.data_dir}")
    audio_files, labels, class_to_idx = find_audio_files_recursive(args.data_dir)

    num_classes = len(class_to_idx)
    print(f"Found {len(audio_files)} audio files")
    print(f"Classes ({num_classes}): {list(class_to_idx.keys())}")

    # Count per class
    from collections import Counter
    label_counts = Counter(labels)
    for emotion, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {emotion}: {label_counts[idx]} files")

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {args.data_dir}")

    # Create dataset
    dataset = AudioDataset(
        audio_paths=audio_files,
        labels=labels,
        sample_rate=22050,
        duration=args.duration,
        n_mels=128,
    )
    dataset.class_to_idx = class_to_idx
    dataset.idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Save class names
    class_names_path = Path("models/checkpoints") / exp_name
    class_names_path.mkdir(parents=True, exist_ok=True)
    with open(class_names_path / "class_names.json", "w") as f:
        json.dump(list(class_to_idx.keys()), f)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        num_workers=0,
    )
    print(f"\nTrain: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Create model
    model = AudioClassifier(
        num_classes=num_classes,
        n_mels=128,
        channels=[32, 64, 128, 256],
        dropout=0.3,
    )

    # Load pretrained weights if provided
    if args.pretrained and Path(args.pretrained).exists():
        print(f"\nLoading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        if "model_state_dict" in checkpoint:
            # Load all except final classifier layer (different num_classes)
            state_dict = checkpoint["model_state_dict"]
            # Filter out final fc layer
            state_dict = {k: v for k, v in state_dict.items()
                         if not k.startswith("fc.4") and "lora_" not in k}
            model.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained conv layers (excluding classifier head)")

    # Apply LoRA
    print(f"\nApplying LoRA (rank={args.lora_rank})")
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=16.0,
        dropout=0.1,
        target_modules=["fc"],
    )

    param_info = count_lora_params(model)
    print(f"Total params: {param_info['total']:,}")
    print(f"Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']:.1f}%)")

    model = model.to(device)

    # Optimizer
    if args.lora_only:
        lora_params = get_lora_params(model)
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
        print("Training LoRA parameters only")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        scheduler=scheduler,
        checkpoint_dir="models/checkpoints",
        log_dir="logs",
        experiment_name=exp_name,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=10,
    )

    # Save LoRA weights
    lora_path = Path("models/checkpoints") / exp_name / "lora_weights.pt"
    save_lora_weights(model, str(lora_path))
    print(f"\nLoRA weights saved to: {lora_path}")

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)
    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Checkpoints: models/checkpoints/{exp_name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
