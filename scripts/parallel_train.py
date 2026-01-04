#!/usr/bin/env python3
"""
Parallel Training Coordinator for Multiple Codespaces
======================================================

Distributes model training across multiple Codespaces for faster completion.

Architecture:
  Codespace A (crispy-funicular)    Codespace B (cautious-couscous)
  ┌──────────────────────────┐     ┌──────────────────────────┐
  │ emotion_recognizer       │     │ harmony_predictor        │
  │ dynamics_engine          │     │ melody_transformer       │
  │                          │     │ groove_predictor         │
  └──────────────────────────┘     └──────────────────────────┘
                 │                            │
                 └──────────┬─────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Local Mac      │
                   │  /Volumes/sbdrive │
                   │  (datasets)     │
                   └─────────────────┘

Usage:
    # On Codespace A:
    python scripts/parallel_train.py --worker A

    # On Codespace B:
    python scripts/parallel_train.py --worker B

    # Or auto-detect based on hostname:
    python scripts/parallel_train.py --auto
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Model assignments for parallel training
# Mac (Worker C): Light/quantized/PEFT tasks - direct dataset access
# Codespaces (A, B): Heavier training runs - SSH mount to Mac datasets
WORKER_ASSIGNMENTS = {
    "A": {
        "name": "crispy-funicular",
        "models": ["melody_transformer", "emotion_recognizer"],
        "description": "Codespace - Heavy transformer training",
        "data_root_env": "KELLY_AUDIO_DATA_ROOT",
    },
    "B": {
        "name": "cautious-couscous",
        "models": ["harmony_predictor", "dynamics_engine"],
        "description": "Codespace - Heavy predictor training",
        "data_root_env": "KELLY_AUDIO_DATA_ROOT",
    },
    "C": {
        "name": "cursor-local",
        "models": ["groove_predictor"],
        "description": "Mac/Cursor - Light quantized task (MPS)",
        "data_root_env": "KELLY_AUDIO_DATA_ROOT",
        "data_root_default": "/Volumes/sbdrive/audio/datasets",
    },
}

# Training configuration optimized for Codespace resources
TRAINING_CONFIG = {
    "emotion_recognizer": {
        "epochs": 50,
        "batch_size": 16,
        "dataset": "m4singer",
        "priority": 1,
        "estimated_time": "2 hours",
    },
    "harmony_predictor": {
        "epochs": 30,
        "batch_size": 32,
        "dataset": "lakh_midi",
        "priority": 1,
        "estimated_time": "1 hour",
    },
    "melody_transformer": {
        "epochs": 50,
        "batch_size": 16,
        "dataset": "lakh_midi",
        "priority": 2,
        "estimated_time": "3 hours",
    },
    "dynamics_engine": {
        "epochs": 30,
        "batch_size": 32,
        "dataset": "m4singer",
        "priority": 2,
        "estimated_time": "1 hour",
    },
    "groove_predictor": {
        "epochs": 30,
        "batch_size": 32,
        "dataset": "lakh_midi",
        "priority": 3,
        "estimated_time": "1 hour",
    },
}


def detect_worker() -> str:
    """Detect which worker this environment is based on hostname."""
    hostname = socket.gethostname().lower()
    codespace_name = os.environ.get("CODESPACE_NAME", "").lower()

    # Check if running locally (Cursor/VS Code on Mac)
    if os.path.exists("/Volumes/sbdrive"):
        # Local Mac - likely Cursor
        return "C"

    for worker_id, config in WORKER_ASSIGNMENTS.items():
        if config["name"] in hostname or config["name"] in codespace_name:
            return worker_id

    # Fallback: ask user
    print("Could not auto-detect worker. Choose:")
    print("  A: crispy-funicular Codespace (emotion_recognizer, dynamics_engine)")
    print("  B: cautious-couscous Codespace (groove_predictor)")
    print("  C: Cursor/Local Mac (melody_transformer, harmony_predictor)")
    choice = input("Enter A, B, or C: ").strip().upper()
    return choice if choice in ["A", "B", "C"] else "C"


def get_data_dir(worker_id: str = None) -> Path:
    """Get the data directory path based on KELLY_AUDIO_DATA_ROOT."""
    # Check KELLY_AUDIO_DATA_ROOT first (explicit per-machine config)
    data_dir = os.environ.get("KELLY_AUDIO_DATA_ROOT")
    if data_dir and Path(data_dir).exists():
        return Path(data_dir)

    # Check legacy KMIDI_DATA_DIR
    data_dir = os.environ.get("KMIDI_DATA_DIR")
    if data_dir and Path(data_dir).exists():
        return Path(data_dir)

    # Worker-specific defaults
    if worker_id and worker_id in WORKER_ASSIGNMENTS:
        default = WORKER_ASSIGNMENTS[worker_id].get("data_root_default")
        if default and Path(default).exists():
            return Path(default)

    # Check common mount points
    for path in ["/data/datasets", "/Volumes/sbdrive/audio/datasets"]:
        if Path(path).exists():
            return Path(path)

    print("ERROR: Dataset directory not found")
    print("Set KELLY_AUDIO_DATA_ROOT environment variable:")
    print("  Mac:       export KELLY_AUDIO_DATA_ROOT=/Volumes/sbdrive/audio/datasets")
    print("  Codespace: export KELLY_AUDIO_DATA_ROOT=/data/datasets")
    sys.exit(1)


def check_dataset_available(dataset_name: str, data_dir: Path) -> bool:
    """Check if required dataset is available."""
    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        print(f"WARNING: {dataset_name} not found at {dataset_path}")
        return False
    return True


def train_model(model_name: str, config: dict, data_dir: Path) -> dict:
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Dataset: {config['dataset']}")
    print(f"Epochs: {config['epochs']}, Batch Size: {config['batch_size']}")
    print(f"Estimated Time: {config['estimated_time']}")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    # Check dataset
    if not check_dataset_available(config["dataset"], data_dir):
        return {
            "model": model_name,
            "status": "skipped",
            "reason": f"Dataset {config['dataset']} not available",
        }

    # Build training command
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", model_name,
        "--data", str(data_dir),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--export-onnx",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 4,  # 4 hour timeout
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if result.returncode == 0:
            return {
                "model": model_name,
                "status": "success",
                "duration_seconds": duration,
                "output": result.stdout[-1000:],  # Last 1000 chars
            }
        else:
            return {
                "model": model_name,
                "status": "failed",
                "duration_seconds": duration,
                "error": result.stderr[-1000:],
            }
    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "status": "timeout",
            "reason": "Training exceeded 4 hour timeout",
        }
    except Exception as e:
        return {
            "model": model_name,
            "status": "error",
            "reason": str(e),
        }


def run_worker(worker_id: str):
    """Run training for assigned models."""
    if worker_id not in WORKER_ASSIGNMENTS:
        print(f"ERROR: Unknown worker ID: {worker_id}")
        sys.exit(1)

    assignment = WORKER_ASSIGNMENTS[worker_id]
    models = assignment["models"]

    print(f"╔{'═'*58}╗")
    print(f"║  Worker {worker_id}: {assignment['name']:<47}║")
    print(f"║  {assignment['description']:<56}║")
    print(f"╚{'═'*58}╝")
    print()
    print(f"Assigned models: {', '.join(models)}")
    print()

    data_dir = get_data_dir()
    print(f"Data directory: {data_dir}")
    print()

    # Sort models by priority
    sorted_models = sorted(
        models,
        key=lambda m: TRAINING_CONFIG[m]["priority"]
    )

    results = []
    for model_name in sorted_models:
        config = TRAINING_CONFIG[model_name]
        result = train_model(model_name, config, data_dir)
        results.append(result)

        # Save intermediate results
        results_file = Path(f"training_results_worker_{worker_id}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        duration = r.get("duration_seconds", 0)
        print(f"  {status_icon} {r['model']}: {r['status']} ({duration:.0f}s)")

    print(f"\nResults saved to: training_results_worker_{worker_id}.json")
    print("\nNext steps:")
    print("  git add models/ checkpoints/ training_results_*.json")
    print("  git commit -m 'Trained models on worker {worker_id}'")
    print("  git push")


def show_plan():
    """Show the parallel training plan."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║           PARALLEL TRAINING PLAN                          ║")
    print("╠══════════════════════════════════════════════════════════╣")

    for worker_id, assignment in WORKER_ASSIGNMENTS.items():
        print(f"║                                                            ║")
        print(f"║  Worker {worker_id}: {assignment['name']:<47}║")
        print(f"║  {assignment['description']:<56}║")
        print(f"║                                                            ║")

        for model in assignment["models"]:
            config = TRAINING_CONFIG[model]
            print(f"║    • {model:<25} ({config['estimated_time']:<12}) ║")

    print("║                                                            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("To start training:")
    print("  Codespace A: python scripts/parallel_train.py --worker A")
    print("  Codespace B: python scripts/parallel_train.py --worker B")
    print()


def main():
    parser = argparse.ArgumentParser(description="Parallel training coordinator")
    parser.add_argument("--worker", choices=["A", "B", "C"], help="Worker ID")
    parser.add_argument("--auto", action="store_true", help="Auto-detect worker")
    parser.add_argument("--plan", action="store_true", help="Show training plan")
    args = parser.parse_args()

    if args.plan:
        show_plan()
        return

    if args.auto:
        worker_id = detect_worker()
    elif args.worker:
        worker_id = args.worker
    else:
        show_plan()
        return

    run_worker(worker_id)


if __name__ == "__main__":
    main()
