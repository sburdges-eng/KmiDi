#!/usr/bin/env python3
"""Main training script for audio classification."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import torch.nn as nn
from datetime import datetime

from src.data import AudioDataset, create_dataloaders
from src.models import AudioClassifier
from src.models.audio_classifier import AudioClassifierResNet
from src.training import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train audio classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.data_dir:
        config["data"]["root_dir"] = args.data_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device
    if args.name:
        config["experiment"]["name"] = args.name

    # Experiment name
    exp_name = config["experiment"]["name"]
    if exp_name is None:
        exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    print("=" * 60)
    print("ML Training Suite - Audio Classifier")
    print("=" * 60)
    print(f"\nExperiment: {exp_name}")
    print(f"Config: {args.config}")

    # Create dataset
    print(f"\nLoading data from: {config['data']['root_dir']}")
    dataset = AudioDataset.from_directory(
        root_dir=config["data"]["root_dir"],
        extensions=config["data"]["extensions"],
        sample_rate=config["data"]["sample_rate"],
        duration=config["data"]["duration"],
        n_mels=config["data"]["n_mels"],
        n_fft=config["data"]["n_fft"],
        hop_length=config["data"]["hop_length"],
    )

    print(f"Total samples: {len(dataset)}")

    # Determine number of classes
    if hasattr(dataset, "class_to_idx") and dataset.class_to_idx:
        num_classes = len(dataset.class_to_idx)
        print(f"Classes: {list(dataset.class_to_idx.keys())}")
    else:
        num_classes = config["model"]["num_classes"]
        print(f"Using default num_classes: {num_classes}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config["training"]["batch_size"],
        train_split=config["splits"]["train"],
        val_split=config["splits"]["val"],
        num_workers=config["training"]["num_workers"],
        seed=config["splits"]["seed"],
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    model_type = config["model"]["type"]
    if model_type == "AudioClassifierResNet":
        model = AudioClassifierResNet(
            num_classes=num_classes,
            n_mels=config["data"]["n_mels"],
            dropout=config["model"]["dropout"],
        )
    else:
        model = AudioClassifier(
            num_classes=num_classes,
            n_mels=config["data"]["n_mels"],
            channels=config["model"]["channels"],
            fc_dim=config["model"]["fc_dim"],
            dropout=config["model"]["dropout"],
        )

    print(f"\nModel: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer_type = config["optimizer"]["type"]
    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            betas=tuple(config["optimizer"]["betas"]),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

    # Create scheduler
    scheduler_type = config["scheduler"]["type"]
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["scheduler"]["mode"],
            factor=config["scheduler"]["factor"],
            patience=config["scheduler"]["patience"],
            min_lr=config["scheduler"]["min_lr"],
        )
    else:
        scheduler = None

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config["device"],
        scheduler=scheduler,
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        log_dir=config["paths"]["log_dir"],
        experiment_name=exp_name,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final evaluation on test set...")
    print("=" * 60)

    # Load best model
    best_checkpoint = Path(config["paths"]["checkpoint_dir"]) / exp_name / "best.pt"
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)

    test_metrics = trainer.validate(test_loader)
    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints: {config['paths']['checkpoint_dir']}/{exp_name}/")
    print(f"Logs: {config['paths']['log_dir']}/{exp_name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
