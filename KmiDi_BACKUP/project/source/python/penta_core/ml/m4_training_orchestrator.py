#!/usr/bin/env python3
"""
M4 Training Orchestrator - Unified training script for Apple Silicon.

Trains all 5 models in the emotion pipeline on M4 Pro/Max:
1. EmotionRecognizer - Audio → emotion embedding
2. MelodyTransformer - Emotion → note probabilities
3. HarmonyPredictor - Context → chord probabilities
4. DynamicsEngine - Emotion → expression parameters
5. GroovePredictor - Arousal → timing/humanization

Usage:
    # Train all models
    python -m penta_core.ml.m4_training_orchestrator --all
    
    # Train specific model
    python -m penta_core.ml.m4_training_orchestrator --model emotion_recognizer
    
    # Export to ONNX
    python -m penta_core.ml.m4_training_orchestrator --export
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class TrainingResult:
    """Result from a training run."""
    model_name: str
    success: bool
    accuracy: float = 0.0
    loss: float = 0.0
    epochs_completed: int = 0
    training_time_seconds: float = 0.0
    checkpoint_path: str = ""
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "success": self.success,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "epochs_completed": self.epochs_completed,
            "training_time_seconds": self.training_time_seconds,
            "checkpoint_path": self.checkpoint_path,
            "error_message": self.error_message,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator."""
    
    # Device settings
    device: str = "auto"  # auto, mps, cuda, cpu
    
    # Training defaults
    default_epochs: int = 100
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    early_stopping_patience: int = 15
    
    # Paths
    output_dir: str = "checkpoints"
    onnx_output_dir: str = "models/onnx"
    
    # Model-specific overrides
    model_configs: Dict[str, Dict] = field(default_factory=dict)
    
    @classmethod
    def default(cls) -> "OrchestratorConfig":
        """Create default configuration optimized for M4."""
        return cls(
            device="auto",
            default_epochs=100,
            default_batch_size=32,
            default_learning_rate=0.001,
            early_stopping_patience=15,
            output_dir="checkpoints",
            onnx_output_dir="models/onnx",
            model_configs={
                "emotion_recognizer": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "input_dim": 128,
                    "output_dim": 64,
                },
                "melody_transformer": {
                    "epochs": 80,
                    "batch_size": 64,
                    "learning_rate": 0.0005,
                    "input_dim": 64,
                    "output_dim": 128,
                },
                "harmony_predictor": {
                    "epochs": 60,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "input_dim": 128,
                    "output_dim": 64,
                },
                "dynamics_engine": {
                    "epochs": 50,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "input_dim": 32,
                    "output_dim": 16,
                },
                "groove_predictor": {
                    "epochs": 50,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "input_dim": 64,
                    "output_dim": 32,
                },
            },
        )


class M4TrainingOrchestrator:
    """
    Orchestrates training of all models on Apple Silicon.
    
    Automatically detects MPS availability and optimizes for M4.
    """
    
    # Model training order (dependencies first)
    MODEL_ORDER = [
        "emotion_recognizer",
        "dynamics_engine",
        "groove_predictor",
        "harmony_predictor",
        "melody_transformer",
    ]
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig.default()
        self.device = self._detect_device()
        self.results: Dict[str, TrainingResult] = {}
        
        # Ensure output directories exist
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.onnx_output_dir).mkdir(parents=True, exist_ok=True)
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        if self.config.device != "auto":
            return self.config.device
        
        try:
            import torch
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("✅ Detected Apple Silicon (MPS) - using Metal Performance Shaders")
                return "mps"
            
            # Check for CUDA
            if torch.cuda.is_available():
                print(f"✅ Detected CUDA GPU - {torch.cuda.get_device_name(0)}")
                return "cuda"
            
            print("⚠️ No GPU detected - using CPU (training will be slower)")
            return "cpu"
            
        except ImportError:
            print("⚠️ PyTorch not installed - using CPU")
            return "cpu"
    
    def train_all(self, epochs_override: Optional[int] = None) -> Dict[str, TrainingResult]:
        """
        Train all models in the pipeline.
        
        Args:
            epochs_override: Override default epochs for all models
        
        Returns:
            Dictionary of model name → TrainingResult
        """
        print("\n" + "=" * 60)
        print("🚀 M4 Training Orchestrator - Starting Full Pipeline")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Models to train: {len(self.MODEL_ORDER)}")
        print("=" * 60 + "\n")
        
        total_start = time.time()
        
        for model_name in self.MODEL_ORDER:
            result = self.train_model(model_name, epochs_override)
            self.results[model_name] = result
            
            if result.success:
                print(f"✅ {model_name}: Accuracy={result.accuracy:.2%}, "
                      f"Time={result.training_time_seconds:.1f}s")
            else:
                print(f"❌ {model_name}: {result.error_message}")
        
        total_time = time.time() - total_start
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Training Summary")
        print("=" * 60)
        
        successful = sum(1 for r in self.results.values() if r.success)
        print(f"Successful: {successful}/{len(self.MODEL_ORDER)}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        for name, result in self.results.items():
            status = "✅" if result.success else "❌"
            print(f"  {status} {name}: {result.accuracy:.2%} accuracy")
        
        print("=" * 60 + "\n")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def train_model(
        self,
        model_name: str,
        epochs_override: Optional[int] = None,
    ) -> TrainingResult:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            epochs_override: Override configured epochs
        
        Returns:
            TrainingResult with training outcome
        """
        print(f"\n{'─' * 40}")
        print(f"🔧 Training: {model_name}")
        print(f"{'─' * 40}")
        
        model_config = self.config.model_configs.get(model_name, {})
        epochs = epochs_override or model_config.get("epochs", self.config.default_epochs)
        batch_size = model_config.get("batch_size", self.config.default_batch_size)
        lr = model_config.get("learning_rate", self.config.default_learning_rate)
        
        print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        
        start_time = time.time()
        
        try:
            # Try to use actual training infrastructure
            result = self._run_training(model_name, epochs, batch_size, lr)
            result.training_time_seconds = time.time() - start_time
            return result
            
        except Exception as e:
            return TrainingResult(
                model_name=model_name,
                success=False,
                training_time_seconds=time.time() - start_time,
                error_message=str(e),
            )
    
    def _run_training(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> TrainingResult:
        """Run actual training using penta_core.ml infrastructure."""
        try:
            import torch
            from penta_core.ml.training.architectures import create_model
            from penta_core.ml.training.losses import get_loss_function
            from penta_core.ml.datasets.synthetic import create_synthetic_dataset
        except ImportError as e:
            return TrainingResult(
                model_name=model_name,
                success=False,
                error_message=f"Import error: {e}",
            )
        
        model_config = self.config.model_configs.get(model_name, {})
        input_dim = model_config.get("input_dim", 64)
        output_dim = model_config.get("output_dim", 32)
        
        # Create model
        try:
            model = create_model(model_name, input_dim, output_dim)
        except Exception as e:
            return TrainingResult(
                model_name=model_name,
                success=False,
                error_message=f"Model creation failed: {e}",
            )
        
        # Move to device
        device = torch.device(self.device)
        model = model.to(device)
        
        # Create synthetic dataset for training
        try:
            train_dataset = create_synthetic_dataset(
                model_name,
                num_samples=1000,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            val_dataset = create_synthetic_dataset(
                model_name,
                num_samples=200,
                input_dim=input_dim,
                output_dim=output_dim,
            )
        except Exception as e:
            return TrainingResult(
                model_name=model_name,
                success=False,
                error_message=f"Dataset creation failed: {e}",
            )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
        )
        
        # Training setup
        loss_fn = get_loss_function(model_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_accuracy = 0.0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_x)
                    val_loss += loss_fn(outputs, batch_y).item()
                    
                    # Calculate accuracy (for classification tasks)
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        predictions = outputs.argmax(dim=1)
                        targets = batch_y.argmax(dim=1) if batch_y.dim() > 1 else batch_y
                        correct += (predictions == targets).sum().item()
                        total += batch_y.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total if total > 0 else 0.0
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = Path(self.config.output_dir) / model_name / "best_model.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            
            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: "
                      f"loss={avg_val_loss:.4f}, acc={accuracy:.2%}")
        
        checkpoint_path = Path(self.config.output_dir) / model_name / "best_model.pt"
        
        return TrainingResult(
            model_name=model_name,
            success=True,
            accuracy=best_accuracy,
            loss=best_val_loss,
            epochs_completed=epoch + 1,
            checkpoint_path=str(checkpoint_path),
        )
    
    def export_to_onnx(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Export trained models to ONNX format.
        
        Args:
            model_names: Models to export, or None for all
        
        Returns:
            Dictionary of model name → export success
        """
        print("\n" + "=" * 60)
        print("📦 Exporting Models to ONNX")
        print("=" * 60)
        
        models_to_export = model_names or self.MODEL_ORDER
        results = {}
        
        for model_name in models_to_export:
            success = self._export_model_onnx(model_name)
            results[model_name] = success
            
            status = "✅" if success else "❌"
            print(f"  {status} {model_name}")
        
        return results
    
    def _export_model_onnx(self, model_name: str) -> bool:
        """Export a single model to ONNX."""
        try:
            import torch
            from penta_core.ml.training.architectures import create_model
            from penta_core.ml.export import export_to_onnx
        except ImportError:
            return False
        
        model_config = self.config.model_configs.get(model_name, {})
        input_dim = model_config.get("input_dim", 64)
        output_dim = model_config.get("output_dim", 32)
        
        # Load trained model
        checkpoint_path = Path(self.config.output_dir) / model_name / "best_model.pt"
        if not checkpoint_path.exists():
            print(f"    ⚠️ No checkpoint found for {model_name}")
            return False
        
        try:
            model = create_model(model_name, input_dim, output_dim)
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            model.eval()
            
            # Export
            onnx_path = Path(self.config.onnx_output_dir) / f"{model_name}.onnx"
            export_to_onnx(model, onnx_path, input_dim)
            
            return True
        except Exception as e:
            print(f"    ⚠️ Export failed: {e}")
            return False
    
    def _save_results(self) -> None:
        """Save training results to JSON."""
        results_path = Path(self.config.output_dir) / "training_results.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "config": {
                "default_epochs": self.config.default_epochs,
                "default_batch_size": self.config.default_batch_size,
            },
            "results": {name: result.to_dict() for name, result in self.results.items()},
        }
        
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"📝 Results saved to {results_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="M4 Training Orchestrator for KmiDi ML Pipeline"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models in the pipeline",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=M4TrainingOrchestrator.MODEL_ORDER,
        help="Train a specific model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override epochs for training",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export trained models to ONNX",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use for training",
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    config = OrchestratorConfig.default()
    config.device = args.device
    
    orchestrator = M4TrainingOrchestrator(config)
    
    if args.all:
        orchestrator.train_all(epochs_override=args.epochs)
    elif args.model:
        result = orchestrator.train_model(args.model, epochs_override=args.epochs)
        if result.success:
            print(f"✅ Training complete: {result.accuracy:.2%} accuracy")
        else:
            print(f"❌ Training failed: {result.error_message}")
            sys.exit(1)
    
    if args.export:
        orchestrator.export_to_onnx()
    
    if not (args.all or args.model or args.export):
        parser.print_help()


if __name__ == "__main__":
    main()
