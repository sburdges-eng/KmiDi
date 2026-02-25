"""Training loop and utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Callable
from tqdm import tqdm
import json
from datetime import datetime


class Trainer:
    """
    Trainer class for PyTorch models.

    Features:
    - Training and validation loops
    - Learning rate scheduling
    - Checkpoint saving/loading
    - TensorBoard logging
    - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "auto",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str | Path = "models/checkpoints",
        log_dir: str | Path = "logs",
        experiment_name: Optional[str] = None,
    ):
        """
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on ("auto", "cpu", "cuda", "mps")
            scheduler: Optional learning rate scheduler
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
            experiment_name: Name for this experiment
        """
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup experiment name and logging
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Update progress bar
            pbar.set_postfix(
                loss=total_loss / (batch_idx + 1),
                acc=100.0 * correct / total,
            )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 10,
        save_best: bool = True,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            save_best: Save best model checkpoint

        Returns:
            Training history dictionary
        """
        patience_counter = 0

        print(f"\nTraining on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Experiment: {self.experiment_name}\n")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])

            # Print epoch summary
            print(
                f"\nEpoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                patience_counter = 0
                if save_best:
                    self.save_checkpoint("best.pt")
                    print(f"  [Saved best model]")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Save final checkpoint
        self.save_checkpoint("final.pt")

        # Save training history
        history_path = self.log_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        self.writer.close()
        print(f"\nTraining complete. Logs saved to {self.log_dir}")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / self.experiment_name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path / filename)

    def load_checkpoint(self, path: str | Path):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
