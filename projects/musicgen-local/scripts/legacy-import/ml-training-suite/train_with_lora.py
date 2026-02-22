#!/usr/bin/env python3
"""Training script with LoRA fine-tuning support."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm

from src.data import AudioDataset, create_dataloaders
from src.models import AudioClassifier
from src.models.lora import (
    apply_lora_to_model,
    get_lora_params,
    save_lora_weights,
    count_lora_params,
)
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train audio classifier with LoRA")
    parser.add_argument("--data-dir", type=str, default="/Volumes/sbdrive",
                       help="Directory with audio files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--duration", type=float, default=5.0, help="Audio duration (seconds)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--lora-only", action="store_true", help="Train only LoRA params")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()

    # Experiment name
    exp_name = args.name or f"lora_r{args.lora_rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print("ML Training Suite - Audio Classifier with LoRA")
    print("=" * 60)
    print(f"\nExperiment: {exp_name}")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Load dataset
    print(f"\nLoading data from: {args.data_dir}")

    # Find all audio files
    from pathlib import Path
    data_dir = Path(args.data_dir)
    audio_files = []
    for ext in [".mp3", ".wav", ".aif", ".aiff", ".flac"]:
        # Only search root level to avoid venv/site-packages test files
        audio_files.extend(list(data_dir.glob(f"*{ext}")))

    # Filter out any test/venv files just in case
    audio_files = [f for f in set(audio_files) if "venv" not in str(f) and "site-packages" not in str(f) and "test" not in str(f).lower()]
    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {args.data_dir}")

    # Create labels based on filename patterns or use dummy labels for unsupervised
    # Simple heuristic: group by first word in filename
    labels = []
    class_to_idx = {}
    for f in audio_files:
        # Extract category from filename (first part before underscore or number)
        name = f.stem.lower()
        # Simple categorization by keywords
        if any(k in name for k in ["piano", "keys", "rhodes"]):
            cat = "piano"
        elif any(k in name for k in ["drum", "beat", "kick", "snare", "hat"]):
            cat = "drums"
        elif any(k in name for k in ["guitar", "guit"]):
            cat = "guitar"
        elif any(k in name for k in ["vocal", "voice", "sing", "scream", "cry", "laugh"]):
            cat = "vocals"
        elif any(k in name for k in ["bass"]):
            cat = "bass"
        elif any(k in name for k in ["synth", "pad", "ambient"]):
            cat = "synth"
        elif any(k in name for k in ["fx", "effect", "noise", "glitch"]):
            cat = "fx"
        elif any(k in name for k in ["horn", "brass"]):
            cat = "brass"
        elif any(k in name for k in ["string", "violin", "cello"]):
            cat = "strings"
        else:
            cat = "other"

        if cat not in class_to_idx:
            class_to_idx[cat] = len(class_to_idx)
        labels.append(class_to_idx[cat])

    num_classes = len(class_to_idx)
    print(f"Auto-categorized into {num_classes} classes: {list(class_to_idx.keys())}")

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
    print(f"Total samples: {len(dataset)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        num_workers=0,  # MPS works better with 0 workers
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Create model
    model = AudioClassifier(
        num_classes=num_classes,
        n_mels=128,
        channels=[32, 64, 128, 256],
        dropout=0.3,
    )

    # Load pretrained weights if provided
    if args.pretrained:
        print(f"\nLoading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Apply LoRA
    print(f"\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.1,
        target_modules=["fc"],  # Apply LoRA to fully connected layers
    )

    # Count parameters
    param_info = count_lora_params(model)
    print(f"Total params: {param_info['total']:,}")
    print(f"LoRA params: {param_info['lora']:,}")
    print(f"Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']:.2f}%)")

    model = model.to(device)

    # Setup optimizer - train only LoRA params if specified
    if args.lora_only:
        lora_params = get_lora_params(model)
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
        print("\nTraining LoRA parameters only")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("\nTraining all parameters")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Create trainer
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
        early_stopping_patience=15,
    )

    # Save LoRA weights separately
    lora_path = Path("models/checkpoints") / exp_name / "lora_weights.pt"
    save_lora_weights(model, str(lora_path))
    print(f"\nLoRA weights saved to: {lora_path}")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Checkpoints: models/checkpoints/{exp_name}/")
    print(f"LoRA weights: {lora_path}")
    print(f"Logs: logs/{exp_name}/")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    main()
