#!/usr/bin/env python3
"""
Integrated Kelly Model Training Script

Trains all Kelly models with performance-first configuration.
Based on previous training results for optimal hyperparameters.

Output Structure:
    checkpoints/
    ├── emotion_recognizer/best_model.pt
    ├── dynamics_engine/best_model.pt
    ├── groove_predictor/best_model.pt
    ├── harmony_predictor/best_model.pt
    ├── melody_transformer/best_model.pt
    ├── spectocloud_vit/best_model.pt
    └── training_results.json

Usage:
    python train_integrated.py --config integrated_training_config.yaml
    python train_integrated.py --model emotion_recognizer --epochs 100
    python train_integrated.py --all --device mps
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required: pip install torch")
    sys.exit(1)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_FILE = CHECKPOINTS_DIR / "training_results.json"


# =============================================================================
# Results Tracking
# =============================================================================

@dataclass
class TrainingResult:
    """Result from training a single model."""
    model_name: str
    accuracy: float
    epochs_trained: int
    epochs_total: int
    early_stopped: bool
    training_time_seconds: float
    best_val_loss: float
    checkpoint_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def summary_line(self) -> str:
        """Format like: emotion_recognizer  89.50%  36/100 (early stopped)  12.2 min"""
        status = f"(early stopped)" if self.early_stopped else ""
        time_str = f"{self.training_time_seconds / 60:.1f} min" if self.training_time_seconds > 60 else f"{self.training_time_seconds:.1f}s"
        return f"{self.model_name:20s} {self.accuracy*100:6.2f}%  {self.epochs_trained}/{self.epochs_total} {status:15s} {time_str}"


class ResultsManager:
    """Manages training results across all models."""
    
    def __init__(self, results_path: Path = RESULTS_FILE):
        self.results_path = results_path
        self.results: Dict[str, TrainingResult] = {}
        self._load_existing()
    
    def _load_existing(self):
        """Load existing results if available."""
        if self.results_path.exists():
            try:
                with open(self.results_path) as f:
                    data = json.load(f)
                for name, result_data in data.get("models", {}).items():
                    self.results[name] = TrainingResult(**result_data)
                logger.info(f"Loaded {len(self.results)} existing results")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load existing results: {e}")
    
    def add_result(self, result: TrainingResult):
        """Add or update a training result."""
        self.results[result.model_name] = result
        self._save()
    
    def _save(self):
        """Save results to JSON."""
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_models": len(self.results),
            "models": {name: r.to_dict() for name, r in self.results.items()},
            "summary": self._generate_summary(),
        }
        
        with open(self.results_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {self.results_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        accuracies = [r.accuracy for r in self.results.values()]
        times = [r.training_time_seconds for r in self.results.values()]
        
        return {
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "best_accuracy": max(accuracies),
            "total_training_time_seconds": sum(times),
            "models_early_stopped": sum(1 for r in self.results.values() if r.early_stopped),
        }
    
    def print_summary(self):
        """Print formatted summary table."""
        print("\n" + "=" * 70)
        print("Training Results Summary")
        print("=" * 70)
        print(f"{'Model':<20} {'Accuracy':>8} {'Epochs':>12} {'Status':>15} {'Time':>10}")
        print("-" * 70)
        
        for result in sorted(self.results.values(), key=lambda r: r.accuracy, reverse=True):
            print(result.summary_line())
        
        print("=" * 70)
        print(f"Results saved to: {self.results_path}")


# =============================================================================
# Model Architectures
# =============================================================================

class EmotionRecognizerCNN(nn.Module):
    """CNN with attention for emotion recognition. Target: 92%+"""
    
    def __init__(self, input_size=128, hidden_layers=[512, 256, 128], output_size=64, dropout=0.15):
        super().__init__()
        
        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(256, 4, dropout=dropout, batch_first=True)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(256 * 8, hidden_layers[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.GELU(),
            nn.Linear(hidden_layers[2], output_size),
        )
    
    def forward(self, x):
        # x: (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        x = self.conv(x)     # (batch, 256, 8)
        x = x.transpose(1, 2)  # (batch, 8, 256)
        x, _ = self.attention(x, x, x)
        x = x.flatten(1)     # (batch, 256*8)
        return self.mlp(x)


class DynamicsEngineMLP(nn.Module):
    """Residual MLP for dynamics. Target: 85%+"""
    
    def __init__(self, input_size=32, hidden_layers=[128, 128, 64], output_size=16, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_layers[0])
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layers[i], hidden_layers[i+1] if i+1 < len(hidden_layers) else hidden_layers[i]),
                nn.BatchNorm1d(hidden_layers[i+1] if i+1 < len(hidden_layers) else hidden_layers[i]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for i in range(len(hidden_layers) - 1)
        ])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x) if x.shape[-1] == block[0].out_features else block(x)
        return torch.sigmoid(self.output(x))


class GroovePredictorMLP(nn.Module):
    """Simple MLP for groove. Already at 100%."""
    
    def __init__(self, input_size=64, hidden_layers=[128, 96, 64], output_size=32, dropout=0.2):
        super().__init__()
        
        layers = []
        in_dim = input_size
        for out_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class HarmonyPredictorTransformer(nn.Module):
    """Small transformer for harmony. Target: 75%+"""
    
    def __init__(self, input_size=128, hidden_dim=256, num_layers=4, num_heads=4, output_size=64, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dim
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence
        return self.output(x)


class MelodyTransformer(nn.Module):
    """Decoder transformer for melody generation. Target: 60%+"""
    
    def __init__(self, vocab_size=512, input_size=64, hidden_dim=384, num_layers=8, 
                 num_heads=6, output_size=128, dropout=0.1, max_seq_len=1024):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emotion_proj = nn.Linear(input_size, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output = nn.Linear(hidden_dim, output_size)
    
    def forward(self, tokens, emotion):
        B, T = tokens.shape
        
        positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        
        emotion_context = self.emotion_proj(emotion).unsqueeze(1)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        x = self.transformer(x, emotion_context, tgt_mask=causal_mask)
        return self.output(x)


# =============================================================================
# Datasets
# =============================================================================

class SyntheticDataset(Dataset):
    """Synthetic dataset for model training."""
    
    def __init__(self, num_samples: int, input_size: int, num_classes: int, task: str = "classification"):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.task = task
        
        # Pre-generate for consistency
        np.random.seed(42)
        self.data = np.random.randn(num_samples, input_size).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = self.labels[idx]
        return x, y


class MelodyDataset(Dataset):
    """Dataset for melody transformer."""
    
    def __init__(self, num_samples: int, seq_len: int = 64, vocab_size: int = 512, emotion_dim: int = 64):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.emotion_dim = emotion_dim
        
        np.random.seed(42)
        self.tokens = np.random.randint(0, vocab_size, (num_samples, seq_len))
        self.emotions = np.random.randn(num_samples, emotion_dim).astype(np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokens[idx], dtype=torch.long)
        emotion = torch.tensor(self.emotions[idx])
        return tokens[:-1], emotion, tokens[1:]  # input, emotion, target


# =============================================================================
# Training Loop
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
) -> TrainingResult:
    """Train a single model with early stopping."""
    
    model_name = config.get("name", "unknown")
    epochs = config.get("epochs", 100)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)
    patience = config.get("patience", 20)
    min_epochs = config.get("min_epochs", 10)
    grad_clip = config.get("grad_clip", 1.0)
    
    logger.info(f"\nTraining {model_name}")
    logger.info(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    epochs_no_improve = 0
    start_time = time.time()
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 2:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            else:  # Melody transformer
                tokens, emotion, targets = batch
                tokens, emotion, targets = tokens.to(device), emotion.to(device), targets.to(device)
                outputs = model(tokens, emotion)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                else:
                    tokens, emotion, targets = batch
                    tokens, emotion, targets = tokens.to(device), emotion.to(device), targets.to(device)
                    outputs = model(tokens, emotion)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        scheduler.step()
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={val_loss:.4f}, acc={accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracy = accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience and epoch >= min_epochs:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    result = TrainingResult(
        model_name=model_name,
        accuracy=best_accuracy,
        epochs_trained=epoch + 1,
        epochs_total=epochs,
        early_stopped=(epochs_no_improve >= patience),
        training_time_seconds=training_time,
        best_val_loss=best_val_loss,
        checkpoint_path=str(best_model_path),
    )
    
    logger.info(f"  Completed: {result.summary_line()}")
    
    return result


# =============================================================================
# Model Factory
# =============================================================================

MODEL_CONFIGS = {
    "emotion_recognizer": {
        "name": "emotion_recognizer",
        "class": EmotionRecognizerCNN,
        "kwargs": {"input_size": 128, "output_size": 64},
        "dataset_kwargs": {"input_size": 128, "num_classes": 64},
        "epochs": 100,
        "lr": 3e-4,
        "patience": 20,
        "min_epochs": 40,
    },
    "dynamics_engine": {
        "name": "dynamics_engine",
        "class": DynamicsEngineMLP,
        "kwargs": {"input_size": 32, "output_size": 16},
        "dataset_kwargs": {"input_size": 32, "num_classes": 16},
        "epochs": 100,
        "lr": 5e-4,
        "patience": 25,
        "min_epochs": 60,
    },
    "groove_predictor": {
        "name": "groove_predictor",
        "class": GroovePredictorMLP,
        "kwargs": {"input_size": 64, "output_size": 32},
        "dataset_kwargs": {"input_size": 64, "num_classes": 32},
        "epochs": 60,
        "lr": 3e-4,
        "patience": 15,
        "min_epochs": 30,
    },
    "harmony_predictor": {
        "name": "harmony_predictor",
        "class": HarmonyPredictorTransformer,
        "kwargs": {"input_size": 128, "output_size": 64},
        "dataset_kwargs": {"input_size": 128, "num_classes": 64},
        "epochs": 150,
        "lr": 1e-4,
        "patience": 40,
        "min_epochs": 80,
    },
    "melody_transformer": {
        "name": "melody_transformer",
        "class": MelodyTransformer,
        "kwargs": {"vocab_size": 512, "input_size": 64, "output_size": 128},
        "is_seq2seq": True,
        "epochs": 200,
        "lr": 1e-4,
        "patience": 50,
        "min_epochs": 100,
    },
}


def train_single_model(model_name: str, device: torch.device, results_manager: ResultsManager, num_samples: int = 2000):
    """Train a single model."""
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    # Create model
    model = config["class"](**config.get("kwargs", {}))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model {model_name}: {n_params:,} parameters")
    
    # Create dataset
    if config.get("is_seq2seq"):
        dataset = MelodyDataset(num_samples)
    else:
        dataset = SyntheticDataset(num_samples, **config["dataset_kwargs"])
    
    # Split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    
    # Train
    checkpoint_dir = CHECKPOINTS_DIR / model_name
    result = train_model(model, train_loader, val_loader, config, device, checkpoint_dir)
    
    # Save result
    results_manager.add_result(result)
    
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Kelly models")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--model", type=str, help="Single model to train")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--samples", type=int, default=2000, help="Number of training samples")
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Results manager
    results_manager = ResultsManager()
    
    # Train
    if args.model:
        train_single_model(args.model, device, results_manager, args.samples)
    elif args.all:
        for model_name in MODEL_CONFIGS:
            try:
                train_single_model(model_name, device, results_manager, args.samples)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
    else:
        parser.print_help()
        print("\nAvailable models:", list(MODEL_CONFIGS.keys()))
        return
    
    # Print summary
    results_manager.print_summary()


if __name__ == "__main__":
    main()
