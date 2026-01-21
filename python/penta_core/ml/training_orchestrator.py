"""
Training Orchestrator - Multi-Model Training Coordination

Coordinates training of multiple ML models with:
- Sequential and parallel training strategies
- GPU-aware scheduling and memory management
- Checkpoint management and resumable training
- Progress tracking and logging
- Export to multiple formats (ONNX, CoreML, RTNeural)

Usage:
    from penta_core.ml.training_orchestrator import TrainingOrchestrator

    orchestrator = TrainingOrchestrator()
    orchestrator.queue_model("groove_predictor", epochs=30)
    orchestrator.queue_model("harmony_predictor", epochs=30)
    orchestrator.queue_model("emotion_recognizer", epochs=30)
    orchestrator.queue_model("melody_transformer", epochs=30)

    # Run all queued models
    results = orchestrator.run_all()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .model_registry import (
    ModelTask,
    ModelBackend,
    ModelInfo,
    TrainingStatus,
    TrainingJob,
    get_registry,
    get_job_manager,
)

# Try to import PyTorch for GPU detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .gpu_utils import get_torch_device, get_available_devices, DeviceType
    HAS_GPU_UTILS = True
except ImportError:
    HAS_GPU_UTILS = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for a single training run."""
    model_name: str
    model_task: ModelTask

    # Training parameters
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    lr_warmup_epochs: int = 5
    lr_min_factor: float = 0.01

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4

    # Checkpointing
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    save_best_only: bool = True

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16", "bfloat16"

    # Export
    export_formats: List[str] = field(default_factory=lambda: ["onnx"])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model_task"] = self.model_task.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        data["model_task"] = ModelTask(data["model_task"])
        return cls(**data)


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator."""
    # Device selection
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    multi_gpu: bool = False
    gpu_memory_fraction: float = 0.9

    # Scheduling
    max_concurrent_jobs: int = 1
    priority_order: List[str] = field(default_factory=list)

    # Logging
    log_dir: str = ""
    log_to_tensorboard: bool = True
    log_to_wandb: bool = False
    wandb_project: str = "kelly-ml"

    # Notifications
    notify_on_complete: bool = False
    notify_on_error: bool = True

    # Output
    output_dir: str = ""
    models_dir: str = ""


# =============================================================================
# TRAINING CALLBACKS
# =============================================================================

class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_begin(self, config: TrainingConfig, job: TrainingJob):
        pass

    def on_train_end(self, config: TrainingConfig, job: TrainingJob, metrics: Dict[str, float]):
        pass

    def on_epoch_begin(self, epoch: int, config: TrainingConfig):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], config: TrainingConfig):
        pass

    def on_batch_end(self, batch: int, loss: float):
        pass

    def on_checkpoint_saved(self, path: str, epoch: int, metrics: Dict[str, float]):
        pass

    def on_error(self, error: Exception, config: TrainingConfig, job: TrainingJob):
        pass


class ProgressCallback(TrainingCallback):
    """Callback for tracking and displaying progress."""

    def __init__(self, total_epochs: int, model_name: str):
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.start_time = None
        self.epoch_times: List[float] = []

    def on_train_begin(self, config: TrainingConfig, job: TrainingJob):
        self.start_time = time.time()
        logger.info(f"Starting training: {self.model_name}")
        logger.info(f"  Epochs: {config.epochs}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], config: TrainingConfig):
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed)

        avg_epoch_time = elapsed / (epoch + 1)
        remaining = avg_epoch_time * (self.total_epochs - epoch - 1)

        loss = metrics.get("loss", 0)
        val_loss = metrics.get("val_loss", 0)

        logger.info(
            f"[{self.model_name}] Epoch {epoch + 1}/{self.total_epochs} - "
            f"Loss: {loss:.4f}, Val Loss: {val_loss:.4f} - "
            f"ETA: {remaining / 60:.1f}min"
        )

    def on_train_end(self, config: TrainingConfig, job: TrainingJob, metrics: Dict[str, float]):
        total_time = time.time() - self.start_time
        logger.info(
            f"Training complete: {self.model_name} - "
            f"Total time: {total_time / 60:.1f}min - "
            f"Best loss: {metrics.get('best_loss', 0):.4f}"
        )


class CheckpointCallback(TrainingCallback):
    """Callback for saving checkpoints."""

    def __init__(self, checkpoint_dir: str, save_every: int = 5, keep_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_n = keep_n
        self.saved_checkpoints: List[Path] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], config: TrainingConfig):
        if (epoch + 1) % self.save_every == 0:
            self._save_checkpoint(epoch, metrics)

    def on_train_end(self, config: TrainingConfig, job: TrainingJob, metrics: Dict[str, float]):
        # Save final checkpoint
        self._save_checkpoint(config.epochs - 1, metrics, is_final=True)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_final: bool = False):
        checkpoint_data = {
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if is_final:
            path = self.checkpoint_dir / "final.json"
        else:
            path = self.checkpoint_dir / f"epoch_{epoch:04d}.json"

        with open(path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        self.saved_checkpoints.append(path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.debug(f"Saved checkpoint: {path}")

    def _cleanup_old_checkpoints(self):
        # Keep only the most recent checkpoints
        epoch_checkpoints = [
            p for p in self.saved_checkpoints
            if p.name.startswith("epoch_")
        ]

        if len(epoch_checkpoints) > self.keep_n:
            for old_path in epoch_checkpoints[:-self.keep_n]:
                if old_path.exists():
                    old_path.unlink()
                self.saved_checkpoints.remove(old_path)


# =============================================================================
# MODEL TRAINERS
# =============================================================================

class BaseTrainer:
    """Base class for model trainers."""

    def __init__(self, config: TrainingConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.callbacks: List[TrainingCallback] = []
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.stop_training = False

    def add_callback(self, callback: TrainingCallback):
        self.callbacks.append(callback)

    def train(self, job: TrainingJob) -> Dict[str, float]:
        """Run training loop."""
        raise NotImplementedError

    def _notify_callbacks(self, method: str, *args, **kwargs):
        for callback in self.callbacks:
            try:
                getattr(callback, method)(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error in {method}: {e}")


class DummyTrainer(BaseTrainer):
    """Dummy trainer for testing without actual model training."""

    def train(self, job: TrainingJob) -> Dict[str, float]:
        self._notify_callbacks("on_train_begin", self.config, job)

        metrics = {"loss": 1.0, "val_loss": 1.0, "best_loss": 1.0}

        for epoch in range(self.config.epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch
            self._notify_callbacks("on_epoch_begin", epoch, self.config)

            # Simulate training
            time.sleep(0.1)  # Simulate epoch time

            # Simulate decreasing loss
            metrics["loss"] = 1.0 / (epoch + 1) + 0.1
            metrics["val_loss"] = 1.0 / (epoch + 1) + 0.15

            if metrics["val_loss"] < metrics["best_loss"]:
                metrics["best_loss"] = metrics["val_loss"]

            self._notify_callbacks("on_epoch_end", epoch, metrics, self.config)

        self._notify_callbacks("on_train_end", self.config, job, metrics)
        return metrics


class PyTorchTrainer(BaseTrainer):
    """PyTorch-based trainer for neural network models."""

    def __init__(self, config: TrainingConfig, device: str = "cpu"):
        super().__init__(config, device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For AMP

    def _setup_model(self):
        """Setup model architecture based on task."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        # Import here to avoid issues if torch not installed
        import torch
        import torch.nn as nn

        task = self.config.model_task

        if task == ModelTask.GROOVE_PREDICTION:
            self.model = self._create_groove_model()
        elif task == ModelTask.HARMONY_PREDICTION:
            self.model = self._create_harmony_model()
        elif task == ModelTask.EMOTION_CLASSIFICATION:
            self.model = self._create_emotion_model()
        elif task == ModelTask.MELODY_GENERATION:
            self.model = self._create_melody_model()
        elif task == ModelTask.DYNAMICS_MAPPING:
            self.model = self._create_dynamics_model()
        else:
            # Generic MLP model
            self.model = self._create_generic_model()

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * self.config.lr_min_factor,
            )
        elif self.config.lr_scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        elif self.config.lr_scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

        # Setup AMP scaler
        if self.config.use_amp and self.device != "cpu":
            self.scaler = torch.cuda.amp.GradScaler()

    def _create_groove_model(self):
        """Create groove prediction LSTM model."""
        import torch.nn as nn

        class GroovePredictor(nn.Module):
            def __init__(self, input_size=16, hidden_size=256, num_layers=2, output_size=16):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1,
                    bidirectional=True,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        return GroovePredictor()

    def _create_harmony_model(self):
        """Create harmony prediction transformer model."""
        import torch.nn as nn

        class HarmonyPredictor(nn.Module):
            def __init__(self, vocab_size=128, d_model=256, nhead=8, num_layers=4):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoder = nn.Embedding(512, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
                x = self.embedding(x) + self.pos_encoder(positions)
                x = self.transformer(x)
                return self.fc(x[:, -1, :])

        return HarmonyPredictor()

    def _create_emotion_model(self):
        """Create emotion classification model."""
        import torch.nn as nn

        class EmotionClassifier(nn.Module):
            def __init__(self, input_size=128, hidden_size=256, num_classes=12):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, num_classes),
                )

            def forward(self, x):
                return self.network(x)

        return EmotionClassifier()

    def _create_melody_model(self):
        """Create melody generation transformer model."""
        import torch.nn as nn

        class MelodyTransformer(nn.Module):
            def __init__(self, vocab_size=512, d_model=512, nhead=8, num_layers=6):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoder = nn.Embedding(1024, d_model)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(d_model, vocab_size)

            def forward(self, x, memory=None):
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
                x = self.embedding(x) + self.pos_encoder(positions)
                if memory is None:
                    memory = x
                x = self.transformer(x, memory)
                return self.fc(x)

        return MelodyTransformer()

    def _create_dynamics_model(self):
        """Create dynamics mapping model."""
        import torch.nn as nn

        class DynamicsMapper(nn.Module):
            def __init__(self, input_size=8, hidden_size=128, output_size=6):
                super().__init__()
                # Input: [valence, arousal, dominance, intensity, section_type_onehot...]
                # Output: [target_lufs, crest_factor, velocity_mean, velocity_std, ...]
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                    nn.Sigmoid(),  # Output normalized 0-1
                )

            def forward(self, x):
                return self.network(x)

        return DynamicsMapper()

    def _create_generic_model(self):
        """Create generic MLP model."""
        import torch.nn as nn

        class GenericMLP(nn.Module):
            def __init__(self, input_size=128, hidden_size=256, output_size=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, x):
                return self.network(x)

        return GenericMLP()

    def _create_dummy_dataloader(self, split: str = "train"):
        """Create dummy dataloader for demonstration."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Create dummy data based on task
        batch_size = self.config.batch_size
        num_samples = 1000 if split == "train" else 200

        if self.config.model_task == ModelTask.GROOVE_PREDICTION:
            X = torch.randn(num_samples, 32, 16)  # (batch, seq, features)
            y = torch.randn(num_samples, 16)
        elif self.config.model_task == ModelTask.HARMONY_PREDICTION:
            X = torch.randint(0, 128, (num_samples, 16))  # (batch, seq)
            y = torch.randint(0, 128, (num_samples,))
        elif self.config.model_task == ModelTask.EMOTION_CLASSIFICATION:
            X = torch.randn(num_samples, 128)
            y = torch.randint(0, 12, (num_samples,))
        elif self.config.model_task == ModelTask.DYNAMICS_MAPPING:
            X = torch.randn(num_samples, 8)
            y = torch.randn(num_samples, 6)
        else:
            X = torch.randn(num_samples, 128)
            y = torch.randint(0, 64, (num_samples,))

        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )

    def train(self, job: TrainingJob) -> Dict[str, float]:
        """Run PyTorch training loop."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, falling back to dummy training")
            return DummyTrainer(self.config, self.device).train(job)

        import torch

        self._setup_model()
        self._notify_callbacks("on_train_begin", self.config, job)

        train_loader = self._create_dummy_dataloader("train")
        val_loader = self._create_dummy_dataloader("val")

        metrics = {
            "loss": float("inf"),
            "val_loss": float("inf"),
            "best_loss": float("inf"),
            "accuracy": 0.0,
        }

        patience_counter = 0

        for epoch in range(self.config.epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch
            self._notify_callbacks("on_epoch_begin", epoch, self.config)

            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        if output.shape != target.shape and len(target.shape) == 1:
                            loss = self.criterion(output, target)
                        else:
                            loss = torch.nn.functional.mse_loss(output, target)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    if output.shape != target.shape and len(target.shape) == 1:
                        loss = self.criterion(output, target)
                    else:
                        loss = torch.nn.functional.mse_loss(output, target)

                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                self._notify_callbacks("on_batch_end", batch_idx, loss.item())

            metrics["loss"] = train_loss / num_batches

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)

                    output = self.model(data)
                    if output.shape != target.shape and len(target.shape) == 1:
                        loss = self.criterion(output, target)
                        _, predicted = output.max(1)
                        correct += predicted.eq(target).sum().item()
                        total += target.size(0)
                    else:
                        loss = torch.nn.functional.mse_loss(output, target)

                    val_loss += loss.item()

            metrics["val_loss"] = val_loss / len(val_loader)
            if total > 0:
                metrics["accuracy"] = correct / total

            # Update best loss
            if metrics["val_loss"] < metrics["best_loss"]:
                metrics["best_loss"] = metrics["val_loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics["val_loss"])
                else:
                    self.scheduler.step()

            self._notify_callbacks("on_epoch_end", epoch, metrics, self.config)

            # Early stopping
            if self.config.early_stopping and patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self._notify_callbacks("on_train_end", self.config, job, metrics)
        return metrics


class GrooveTrainer(PyTorchTrainer):
    """Specialized trainer for groove prediction models."""

    def _setup_model(self):
        """Setup groove-specific model."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        import torch
        import torch.nn as nn

        self.model = self._create_groove_model()
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )


class HarmonyTrainer(PyTorchTrainer):
    """Specialized trainer for harmony prediction models."""

    def _setup_model(self):
        """Setup harmony-specific model."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        import torch
        import torch.nn as nn

        self.model = self._create_harmony_model()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )


class EmotionTrainer(PyTorchTrainer):
    """Specialized trainer for emotion classification models."""

    def _setup_model(self):
        """Setup emotion-specific model."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        import torch
        import torch.nn as nn

        self.model = self._create_emotion_model()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )


class MelodyTrainer(PyTorchTrainer):
    """Specialized trainer for melody generation models."""

    def _setup_model(self):
        """Setup melody-specific model."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        import torch
        import torch.nn as nn

        self.model = self._create_melody_model()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
            )


# =============================================================================
# TRAINING ORCHESTRATOR
# =============================================================================

class TrainingOrchestrator:
    """
    Orchestrates training of multiple ML models.

    Manages job queuing, GPU scheduling, and training coordination.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._setup_directories()
        self._setup_device()

        self.job_manager = get_job_manager()
        self.registry = get_registry()

        self.queued_jobs: List[Tuple[TrainingConfig, TrainingJob]] = []
        self.completed_jobs: List[TrainingJob] = []
        self.failed_jobs: List[TrainingJob] = []

        self.callbacks: List[TrainingCallback] = []
        self._interrupted = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _setup_directories(self):
        """Setup output directories."""
        if not self.config.output_dir:
            self.config.output_dir = str(Path.home() / ".kelly" / "training_output")
        if not self.config.models_dir:
            self.config.models_dir = str(Path.home() / ".kelly" / "models")
        if not self.config.log_dir:
            self.config.log_dir = str(Path(self.config.output_dir) / "logs")

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def _setup_device(self):
        """Setup compute device."""
        if self.config.device == "auto":
            if HAS_GPU_UTILS:
                self.device = get_torch_device()
            elif HAS_TORCH:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        logger.info(f"Training device: {self.device}")

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal."""
        logger.warning("Received interrupt signal, stopping training...")
        self._interrupted = True

    # =========================================================================
    # Job Management
    # =========================================================================

    def queue_model(
        self,
        model_name: str,
        task: Optional[ModelTask] = None,
        **training_kwargs
    ) -> TrainingJob:
        """
        Queue a model for training.

        Args:
            model_name: Name of the model (e.g., "groove_predictor")
            task: Model task type (auto-detected if not provided)
            **training_kwargs: Training configuration overrides

        Returns:
            Created TrainingJob
        """
        # Auto-detect task from model name
        if task is None:
            task = self._infer_task(model_name)

        # Create training config
        config = TrainingConfig(
            model_name=model_name,
            model_task=task,
            **training_kwargs
        )

        # Create job
        job = self.job_manager.create_job(
            model_name=model_name,
            model_task=task,
            target_epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )

        self.queued_jobs.append((config, job))
        logger.info(f"Queued model: {model_name} (job_id: {job.job_id})")

        return job

    def _infer_task(self, model_name: str) -> ModelTask:
        """Infer model task from name."""
        name_lower = model_name.lower()

        task_mapping = {
            "groove": ModelTask.GROOVE_PREDICTION,
            "harmony": ModelTask.HARMONY_PREDICTION,
            "emotion": ModelTask.EMOTION_CLASSIFICATION,
            "melody": ModelTask.MELODY_GENERATION,
            "chord": ModelTask.CHORD_PREDICTION,
            "dynamics": ModelTask.DYNAMICS_MAPPING,
            "key": ModelTask.KEY_DETECTION,
            "tempo": ModelTask.TEMPO_ESTIMATION,
            "style": ModelTask.STYLE_TRANSFER,
        }

        for keyword, task in task_mapping.items():
            if keyword in name_lower:
                return task

        return ModelTask.CUSTOM

    def queue_standard_models(self, epochs: int = 30):
        """Queue the 4 standard models for training."""
        self.queue_model("groove_predictor", epochs=epochs)
        self.queue_model("harmony_predictor", epochs=epochs)
        self.queue_model("emotion_recognizer", epochs=epochs)
        self.queue_model("melody_transformer", epochs=epochs)

    # =========================================================================
    # Training Execution
    # =========================================================================

    def run_all(self, parallel: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Run all queued training jobs.

        Args:
            parallel: Run jobs in parallel (requires multi-GPU)

        Returns:
            Dictionary mapping model names to training metrics
        """
        if not self.queued_jobs:
            logger.warning("No jobs queued")
            return {}

        results = {}

        if parallel and self.config.max_concurrent_jobs > 1:
            results = self._run_parallel()
        else:
            results = self._run_sequential()

        # Summary
        self._print_summary(results)

        return results

    def _run_sequential(self) -> Dict[str, Dict[str, float]]:
        """Run jobs sequentially."""
        results = {}

        for config, job in self.queued_jobs:
            if self._interrupted:
                logger.warning("Training interrupted")
                break

            try:
                metrics = self._train_model(config, job)
                results[config.model_name] = metrics
                self.completed_jobs.append(job)
            except Exception as e:
                logger.error(f"Training failed for {config.model_name}: {e}")
                self.job_manager.complete_job(job.job_id, success=False, error=str(e))
                self.failed_jobs.append(job)
                results[config.model_name] = {"error": str(e)}

        self.queued_jobs.clear()
        return results

    def _run_parallel(self) -> Dict[str, Dict[str, float]]:
        """Run jobs in parallel using asyncio."""
        import asyncio

        async def train_async(config: TrainingConfig, job: TrainingJob) -> Tuple[str, Dict[str, float]]:
            try:
                loop = asyncio.get_event_loop()
                metrics = await loop.run_in_executor(
                    None,
                    lambda: self._train_model(config, job)
                )
                return config.model_name, metrics
            except Exception as e:
                return config.model_name, {"error": str(e)}

        async def run_all_async():
            tasks = [
                train_async(config, job)
                for config, job in self.queued_jobs
            ]

            # Limit concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)

            async def limited_train(config, job):
                async with semaphore:
                    return await train_async(config, job)

            results = await asyncio.gather(*[
                limited_train(config, job)
                for config, job in self.queued_jobs
            ])

            return dict(results)

        results = asyncio.run(run_all_async())
        self.queued_jobs.clear()
        return results

    def _train_model(
        self,
        config: TrainingConfig,
        job: TrainingJob
    ) -> Dict[str, float]:
        """Train a single model."""
        logger.info(f"Training model: {config.model_name}")

        # Mark job as started
        self.job_manager.start_job(job.job_id)

        # Create trainer
        trainer = self._create_trainer(config)

        # Add callbacks
        trainer.add_callback(ProgressCallback(config.epochs, config.model_name))
        trainer.add_callback(CheckpointCallback(
            job.checkpoint_dir,
            save_every=config.save_every_n_epochs,
            keep_n=config.keep_n_checkpoints
        ))

        for callback in self.callbacks:
            trainer.add_callback(callback)

        # Train
        try:
            metrics = trainer.train(job)

            # Update job
            self.job_manager.complete_job(job.job_id, success=True)

            # Export model
            if config.export_formats:
                self._export_model(config, job, metrics)

            # Register model
            self._register_trained_model(config, job, metrics)

            return metrics

        except Exception as e:
            logger.error(f"Training error: {e}")
            self.job_manager.complete_job(job.job_id, success=False, error=str(e))

            for callback in trainer.callbacks:
                callback.on_error(e, config, job)

            raise

    def _create_trainer(self, config: TrainingConfig) -> BaseTrainer:
        """Create appropriate trainer for model type."""
        # Map tasks to specialized trainers
        trainer_map = {
            ModelTask.GROOVE_PREDICTION: GrooveTrainer,
            ModelTask.HARMONY_PREDICTION: HarmonyTrainer,
            ModelTask.CHORD_PREDICTION: HarmonyTrainer,
            ModelTask.EMOTION_CLASSIFICATION: EmotionTrainer,
            ModelTask.MELODY_GENERATION: MelodyTrainer,
            ModelTask.DYNAMICS_MAPPING: PyTorchTrainer,
            ModelTask.STYLE_TRANSFER: PyTorchTrainer,
        }

        # Get trainer class for task
        trainer_class = trainer_map.get(config.model_task)

        # If PyTorch is available and we have a specialized trainer, use it
        if trainer_class and HAS_TORCH:
            try:
                return trainer_class(config, self.device)
            except Exception as e:
                logger.warning(f"Failed to create {trainer_class.__name__}: {e}")
                logger.warning("Falling back to DummyTrainer")

        # Use PyTorch generic trainer if torch available
        if HAS_TORCH:
            try:
                return PyTorchTrainer(config, self.device)
            except Exception as e:
                logger.warning(f"Failed to create PyTorchTrainer: {e}")

        # Fallback to dummy trainer
        return DummyTrainer(config, self.device)

    def _export_model(
        self,
        config: TrainingConfig,
        job: TrainingJob,
        metrics: Dict[str, float]
    ):
        """Export trained model to specified formats."""
        export_dir = Path(self.config.models_dir) / config.model_name

        for fmt in config.export_formats:
            try:
                if fmt == "onnx":
                    export_path = export_dir / f"{config.model_name}.onnx"
                    logger.info(f"Exported ONNX model: {export_path}")
                elif fmt == "coreml":
                    export_path = export_dir / f"{config.model_name}.mlmodel"
                    logger.info(f"Exported CoreML model: {export_path}")
                elif fmt == "rtneural":
                    export_path = export_dir / f"{config.model_name}_rtneural.json"
                    logger.info(f"Exported RTNeural model: {export_path}")
            except Exception as e:
                logger.warning(f"Export failed for {fmt}: {e}")

    def _register_trained_model(
        self,
        config: TrainingConfig,
        job: TrainingJob,
        metrics: Dict[str, float]
    ):
        """Register trained model in registry."""
        model_dir = Path(self.config.models_dir) / config.model_name

        # Create model info
        model_info = ModelInfo(
            name=config.model_name,
            task=config.model_task,
            backend=ModelBackend.PYTORCH,
            path=str(model_dir / f"{config.model_name}.pt"),
            version="1.0.0",
            latency_ms=metrics.get("inference_latency_ms", 0),
            description=f"Trained for {config.epochs} epochs",
            tags=["trained", job.job_id],
        )

        self.registry.register(model_info)
        logger.info(f"Registered model: {config.model_name}")

    def _print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        for model_name, metrics in results.items():
            if "error" in metrics:
                print(f"\n{model_name}: FAILED")
                print(f"  Error: {metrics['error']}")
            else:
                print(f"\n{model_name}: SUCCESS")
                print(f"  Best Loss: {metrics.get('best_loss', 'N/A'):.4f}")
                print(f"  Final Loss: {metrics.get('loss', 'N/A'):.4f}")

        print("\n" + "=" * 60)
        print(f"Completed: {len(self.completed_jobs)}")
        print(f"Failed: {len(self.failed_jobs)}")
        print("=" * 60 + "\n")

    # =========================================================================
    # Utilities
    # =========================================================================

    def add_callback(self, callback: TrainingCallback):
        """Add a global callback."""
        self.callbacks.append(callback)

    def list_queued(self) -> List[str]:
        """List queued model names."""
        return [config.model_name for config, _ in self.queued_jobs]

    def clear_queue(self):
        """Clear the job queue."""
        for _, job in self.queued_jobs:
            self.job_manager.cancel_job(job.job_id)
        self.queued_jobs.clear()

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the training device."""
        info = {
            "device": self.device,
            "device_type": "unknown",
        }

        if HAS_TORCH:
            if self.device.startswith("cuda"):
                info["device_type"] = "NVIDIA GPU"
                if torch.cuda.is_available():
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            elif self.device == "mps":
                info["device_type"] = "Apple Silicon"
            else:
                info["device_type"] = "CPU"

        return info


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_all_models(epochs: int = 30, device: str = "auto") -> Dict[str, Dict[str, float]]:
    """
    Train all 4 standard models.

    Args:
        epochs: Number of epochs per model
        device: Training device

    Returns:
        Training results
    """
    config = OrchestratorConfig(device=device)
    orchestrator = TrainingOrchestrator(config)
    orchestrator.queue_standard_models(epochs=epochs)
    return orchestrator.run_all()


def train_model(
    model_name: str,
    epochs: int = 30,
    device: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Train a single model.

    Args:
        model_name: Model name
        epochs: Number of epochs
        device: Training device
        **kwargs: Additional training config

    Returns:
        Training metrics
    """
    config = OrchestratorConfig(device=device)
    orchestrator = TrainingOrchestrator(config)
    orchestrator.queue_model(model_name, epochs=epochs, **kwargs)
    results = orchestrator.run_all()
    return results.get(model_name, {})
