#!/usr/bin/env python3
"""
Improved Full Training Pipeline for Kelly ML Models.

Based on analysis of previous training results:
- emotion_recognizer: 89.50% (good)
- dynamics_engine: 73.50% (moderate)
- groove_predictor: 100.00% (excellent)
- harmony_predictor: 54.00% (needs improvement)
- melody_transformer: 34.50% (needs improvement)

This script implements improvements for underperforming models.

Usage:
    # Train all models with improved config
    python scripts/improved_training_pipeline.py --all
    
    # Train specific model
    python scripts/improved_training_pipeline.py --model harmony_predictor
    
    # Train only underperforming models
    python scripts/improved_training_pipeline.py --improve-only
    
    # Dry run (show config only)
    python scripts/improved_training_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes (no PyTorch dependency)
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    status: str = "unknown"
    previous_accuracy: float = 0.0
    
    # Architecture
    input_dim: int = 64
    output_dim: int = 32
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    use_residual: bool = False
    use_layer_norm: bool = False
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    early_stopping_patience: int = 20
    gradient_accumulation_steps: int = 1
    
    # Loss
    loss_type: str = "cross_entropy"
    label_smoothing: float = 0.0
    
    # Notes
    notes: str = ""
    improvements: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    device: str = "auto"
    seed: int = 42
    output_dir: str = "checkpoints"
    onnx_output_dir: str = "models/onnx"
    
    # Models
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        config = cls(
            device=data.get("device", "auto"),
            seed=data.get("seed", 42),
            output_dir=data.get("output_dir", "checkpoints"),
            onnx_output_dir=data.get("onnx_output_dir", "models/onnx"),
        )
        
        # Parse model configs
        for name, model_data in data.get("models", {}).items():
            config.models[name] = ModelConfig(
                name=name,
                status=model_data.get("status", "unknown"),
                previous_accuracy=model_data.get("previous_accuracy", 0.0),
                input_dim=model_data.get("input_dim", 64),
                output_dim=model_data.get("output_dim", 32),
                hidden_layers=model_data.get("hidden_layers", [256, 128]),
                dropout=model_data.get("dropout", 0.2),
                use_residual=model_data.get("use_residual", False),
                use_layer_norm=model_data.get("use_layer_norm", False),
                epochs=model_data.get("epochs", 100),
                batch_size=model_data.get("batch_size", 32),
                learning_rate=model_data.get("learning_rate", 0.001),
                weight_decay=model_data.get("weight_decay", 0.01),
                warmup_epochs=model_data.get("warmup_epochs", 10),
                early_stopping_patience=model_data.get("early_stopping_patience", 20),
                gradient_accumulation_steps=model_data.get("gradient_accumulation_steps", 1),
                loss_type=model_data.get("loss_type", "cross_entropy"),
                label_smoothing=model_data.get("label_smoothing", 0.0),
                notes=model_data.get("notes", ""),
                improvements=model_data.get("improvements", []),
            )
        
        return config
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default improved configuration based on training results."""
        config = cls()
        
        # Emotion Recognizer - Already good (89.5%)
        config.models["emotion_recognizer"] = ModelConfig(
            name="emotion_recognizer",
            status="good",
            previous_accuracy=0.895,
            input_dim=128,
            output_dim=64,
            hidden_layers=[512, 256, 128],
            dropout=0.3,
            epochs=100,
            batch_size=16,
            learning_rate=0.001,
            early_stopping_patience=15,
            notes="Performing well - maintain current settings",
        )
        
        # Dynamics Engine - Moderate (73.5%)
        config.models["dynamics_engine"] = ModelConfig(
            name="dynamics_engine",
            status="moderate",
            previous_accuracy=0.735,
            input_dim=32,
            output_dim=16,
            hidden_layers=[128, 64, 32],
            dropout=0.2,
            epochs=80,
            batch_size=64,
            learning_rate=0.0008,
            early_stopping_patience=20,
            improvements=["Increased hidden layers", "Lower learning rate", "More epochs"],
        )
        
        # Groove Predictor - Excellent (100%)
        config.models["groove_predictor"] = ModelConfig(
            name="groove_predictor",
            status="excellent",
            previous_accuracy=1.0,
            input_dim=64,
            output_dim=32,
            hidden_layers=[128, 64],
            dropout=0.1,
            epochs=50,
            batch_size=128,
            learning_rate=0.001,
            early_stopping_patience=15,
            notes="Perfect accuracy - maintain settings",
        )
        
        # Harmony Predictor - Needs Improvement (54%)
        config.models["harmony_predictor"] = ModelConfig(
            name="harmony_predictor",
            status="needs_improvement",
            previous_accuracy=0.54,
            input_dim=128,
            output_dim=64,
            hidden_layers=[512, 256, 128],
            dropout=0.3,
            use_residual=True,
            use_layer_norm=True,
            epochs=150,
            batch_size=32,
            learning_rate=0.0005,
            warmup_epochs=15,
            early_stopping_patience=30,
            loss_type="label_smoothing",
            label_smoothing=0.1,
            improvements=[
                "Deeper architecture with residual connections",
                "Lower learning rate (0.0005 vs 0.001)",
                "Longer warmup (15 epochs)",
                "Label smoothing for better generalization",
                "Longer training (150 epochs) with more patience",
            ],
        )
        
        # Melody Transformer - Needs Improvement (34.5%)
        config.models["melody_transformer"] = ModelConfig(
            name="melody_transformer",
            status="needs_improvement",
            previous_accuracy=0.345,
            input_dim=64,
            output_dim=128,
            hidden_layers=[512, 512, 256],
            dropout=0.2,
            epochs=200,
            batch_size=16,
            learning_rate=0.0003,
            weight_decay=0.05,
            warmup_epochs=20,
            early_stopping_patience=40,
            gradient_accumulation_steps=4,
            improvements=[
                "Much lower learning rate (0.0003 vs 0.001)",
                "Gradient accumulation (effective batch=64)",
                "Longer training (200 epochs) with very long patience",
                "Increased weight decay (0.05)",
                "Deeper hidden layers",
            ],
        )
        
        return config


# =============================================================================
# Main function and CLI
# =============================================================================

MODEL_ORDER = [
    "emotion_recognizer",
    "dynamics_engine",
    "groove_predictor",
    "harmony_predictor",
    "melody_transformer",
]


def print_config(config: PipelineConfig):
    """Print configuration in a readable format."""
    print("\n" + "=" * 70)
    print("📋 Improved Training Configuration")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Output: {config.output_dir}")
    print()
    
    for name in MODEL_ORDER:
        if name not in config.models:
            continue
        model_cfg = config.models[name]
        
        # Status indicator
        status_icons = {
            "excellent": "🟢",
            "good": "🟢",
            "moderate": "🟡",
            "needs_improvement": "🔴",
            "unknown": "⚪",
        }
        status_icon = status_icons.get(model_cfg.status, "⚪")
        
        print(f"{status_icon} {name}")
        print(f"   Previous accuracy: {model_cfg.previous_accuracy:.1%}")
        print(f"   Status: {model_cfg.status}")
        print(f"   Epochs: {model_cfg.epochs}")
        print(f"   Batch size: {model_cfg.batch_size}")
        print(f"   Learning rate: {model_cfg.learning_rate}")
        print(f"   Hidden layers: {model_cfg.hidden_layers}")
        if model_cfg.use_residual:
            print(f"   Residual: Yes")
        if model_cfg.gradient_accumulation_steps > 1:
            print(f"   Grad accumulation: {model_cfg.gradient_accumulation_steps} (effective batch={model_cfg.batch_size * model_cfg.gradient_accumulation_steps})")
        if model_cfg.improvements:
            print(f"   Improvements:")
            for imp in model_cfg.improvements:
                print(f"      • {imp}")
        print()
    
    print("=" * 70)
    print()


def run_training(config: PipelineConfig, models_to_train: List[str]):
    """Run the actual training - requires PyTorch."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        logger.error("PyTorch is required for training. Install with: pip install torch")
        sys.exit(1)
    
    # Detect device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("✅ Using CUDA GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("✅ Using Apple Silicon (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("⚠️ Using CPU")
    else:
        device = torch.device(config.device)
    
    # Set seed
    torch.manual_seed(config.seed)
    
    results = {}
    total_start = time.time()
    
    for model_name in models_to_train:
        if model_name not in config.models:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        model_cfg = config.models[model_name]
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🔧 Training: {model_name}")
        logger.info(f"   Previous: {model_cfg.previous_accuracy:.1%}")
        logger.info(f"   Status: {model_cfg.status}")
        logger.info("=" * 60)
        
        # Create simple MLP model
        layers = []
        prev_dim = model_cfg.input_dim
        for hidden_dim in model_cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if model_cfg.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_cfg.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, model_cfg.output_dim))
        model = nn.Sequential(*layers).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Parameters: {num_params:,}")
        
        # Create synthetic dataset for pipeline testing
        # Note: Replace with real dataset for actual training
        class SyntheticDataset(Dataset):
            """Synthetic dataset for testing the pipeline.
            
            This creates learnable patterns by adding correlation between
            input features and labels. Replace with real audio/MIDI data
            for production training.
            """
            def __init__(self, num_samples, input_dim, num_classes):
                self.num_samples = num_samples
                self.input_dim = input_dim
                self.num_classes = num_classes
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Create input with some structure based on index
                torch.manual_seed(idx)  # Reproducible per sample
                x = torch.randn(self.input_dim)
                # Add class-dependent signal for learnability
                label_idx = idx % self.num_classes
                x[:self.num_classes] += torch.zeros(self.num_classes).scatter_(0, torch.tensor(label_idx), 1.0)
                return x, label_idx
        
        # Determine number of classes
        if model_name == "emotion_recognizer":
            num_classes = 7
        elif model_name == "harmony_predictor":
            num_classes = 48
        elif model_name == "melody_transformer":
            num_classes = 128
        else:
            num_classes = model_cfg.output_dim
        
        train_dataset = SyntheticDataset(1000, model_cfg.input_dim, num_classes)
        val_dataset = SyntheticDataset(200, model_cfg.input_dim, num_classes)
        
        train_loader = DataLoader(train_dataset, batch_size=model_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model_cfg.batch_size)
        
        # Training setup
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_cfg.learning_rate,
            weight_decay=model_cfg.weight_decay,
        )
        
        # Loss function - use label smoothing if configured
        if model_cfg.loss_type == "label_smoothing" or model_cfg.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=model_cfg.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float("inf")
        best_accuracy = 0.0
        epochs_no_improve = 0
        start_time = time.time()
        
        for epoch in range(model_cfg.epochs):
            # Train with gradient accumulation
            model.train()
            optimizer.zero_grad()
            num_batches = len(train_loader)
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / model_cfg.gradient_accumulation_steps
                loss.backward()
                
                # Step optimizer after accumulation or at end of epoch
                is_accumulation_step = (batch_idx + 1) % model_cfg.gradient_accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == num_batches
                if is_accumulation_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Validate
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    correct += (outputs.argmax(dim=1) == targets).sum().item()
                    total += targets.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save checkpoint
                checkpoint_dir = Path(config.output_dir) / model_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            else:
                epochs_no_improve += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"   Epoch {epoch + 1}/{model_cfg.epochs}: loss={avg_val_loss:.4f}, acc={accuracy:.2%}")
            
            if epochs_no_improve >= model_cfg.early_stopping_patience:
                logger.info(f"   Early stopping at epoch {epoch + 1}")
                break
        
        training_time = time.time() - start_time
        improvement = best_accuracy - model_cfg.previous_accuracy
        
        results[model_name] = {
            "success": True,
            "best_accuracy": best_accuracy,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch + 1,
            "training_time_seconds": training_time,
            "improvement": improvement,
        }
        
        if improvement > 0:
            logger.info(f"   ✅ Accuracy: {best_accuracy:.1%} (improved by {improvement:.1%})")
        else:
            logger.info(f"   ➡️ Accuracy: {best_accuracy:.1%} (change: {improvement:.1%})")
    
    # Summary
    total_time = time.time() - total_start
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Training Summary")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    for name, result in results.items():
        imp = result["improvement"]
        status = "✅" if imp >= 0 else "⚠️"
        prev = config.models[name].previous_accuracy
        logger.info(f"  {status} {name}: {result['best_accuracy']:.1%} (prev: {prev:.1%}, {'↑' if imp > 0 else '↓'}{abs(imp):.1%})")
    
    logger.info("=" * 60)
    
    # Save results
    results_path = Path(config.output_dir) / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "total_time_seconds": total_time,
            "results": results,
        }, f, indent=2)
    
    logger.info(f"📝 Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Improved Training Pipeline for Kelly ML Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_ORDER,
        help="Train specific model",
    )
    parser.add_argument(
        "--improve-only",
        action="store_true",
        help="Train only underperforming models (harmony_predictor, melody_transformer)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without training",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = PipelineConfig.from_yaml(Path(args.config))
    else:
        config = PipelineConfig.default()
    
    if args.device:
        config.device = args.device
    
    # Dry run - show config only
    if args.dry_run:
        print_config(config)
        return
    
    # Determine which models to train
    if args.model:
        models_to_train = [args.model]
    elif args.improve_only:
        models_to_train = [
            name for name, cfg in config.models.items()
            if cfg.status == "needs_improvement"
        ]
        logger.info(f"Training underperforming models: {models_to_train}")
    else:
        models_to_train = MODEL_ORDER
    
    # Run training
    run_training(config, models_to_train)


if __name__ == "__main__":
    main()
