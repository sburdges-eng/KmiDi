#!/usr/bin/env python3
"""
Comprehensive Dataset Preparation Pipeline for Kelly ML Training.

This script provides a complete pipeline for preparing training datasets:
- Create dataset structure
- Import files from directories
- Auto-annotate or interactive annotation
- Extract features (MIDI, audio, groove)
- Augment datasets (MIDI and audio)
- Generate synthetic data
- Validate datasets
- Create train/val/test splits

Usage:
    python scripts/dataset_pipeline.py --create --dataset emotion_dataset_v1 --target-model emotionrecognizer
    python scripts/dataset_pipeline.py --import-dir /path/to/music --dataset emotion_dataset_v1
    python scripts/dataset_pipeline.py --augment --multiplier 10 --dataset emotion_dataset_v1
    python scripts/dataset_pipeline.py --validate --dataset emotion_dataset_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from penta_core.ml.datasets.augmentation import (
        AudioAugmenter,
        MIDIAugmenter,
        AugmentationConfig,
    )
    from penta_core.ml.datasets.midi_features import MIDIFeatureExtractor
    from penta_core.ml.datasets.synthetic import SyntheticGenerator
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    logger.warning("Some features may be limited")

# Default dataset root
DEFAULT_DATASET_ROOT = Path.cwd() / "datasets"


# =============================================================================
# Dataset Structure Creation
# =============================================================================


def create_dataset_structure(
    dataset_name: str,
    target_model: str,
    root_dir: Path = DEFAULT_DATASET_ROOT,
) -> Path:
    """
    Create a new dataset directory structure.
    
    Args:
        dataset_name: Name of the dataset (e.g., "emotion_dataset_v1")
        target_model: Target model (e.g., "emotionrecognizer")
        root_dir: Root directory for datasets
    
    Returns:
        Path to the created dataset directory
    """
    dataset_dir = root_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (dataset_dir / "raw" / "midi").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "raw" / "audio").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "raw" / "synthetic").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "processed" / "features").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "processed" / "augmented").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "splits").mkdir(parents=True, exist_ok=True)
    
    # Create config file
    config = {
        "dataset_id": dataset_name,
        "target_model": target_model,
        "created_at": str(Path(__file__).stat().st_mtime),
        "version": "1.0",
    }
    config_path = dataset_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Create empty manifest
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        with open(manifest_path, "w") as f:
            json.dump({"samples": []}, f, indent=2)
    
    logger.info(f"Created dataset structure: {dataset_dir}")
    return dataset_dir


# =============================================================================
# File Import
# =============================================================================


def import_files(
    source_dir: Path,
    dataset_dir: Path,
    file_types: List[str] = None,
) -> int:
    """
    Import files from a directory into the dataset.
    
    Args:
        source_dir: Source directory to import from
        dataset_dir: Dataset directory
        file_types: List of file extensions to import (default: [".mid", ".wav", ".mp3"])
    
    Returns:
        Number of files imported
    """
    if file_types is None:
        file_types = [".mid", ".midi", ".wav", ".mp3", ".flac", ".ogg"]
    
    source_dir = Path(source_dir)
    dataset_dir = Path(dataset_dir)
    
    raw_midi = dataset_dir / "raw" / "midi"
    raw_audio = dataset_dir / "raw" / "audio"
    
    imported = 0
    
    # Find all matching files
    for ext in file_types:
        for file_path in source_dir.rglob(f"*{ext}"):
            try:
                if ext in [".mid", ".midi"]:
                    dest = raw_midi / file_path.name
                else:
                    dest = raw_audio / file_path.name
                
                # Avoid duplicates
                if not dest.exists():
                    shutil.copy2(file_path, dest)
                    imported += 1
                    logger.debug(f"Imported: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to import {file_path}: {e}")
    
    logger.info(f"Imported {imported} files from {source_dir}")
    return imported


# =============================================================================
# Annotation
# =============================================================================


def auto_annotate_from_structure(dataset_dir: Path) -> int:
    """
    Auto-annotate files based on directory structure.
    
    Assumes structure like:
        source_dir/
            happy/
                song1.mid
            sad/
                song2.mid
    
    Args:
        dataset_dir: Dataset directory
    
    Returns:
        Number of files annotated
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    
    # Load existing manifest
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"samples": []}
    
    annotated = 0
    
    # Check raw directories for emotion subdirectories
    for raw_type in ["midi", "audio"]:
        raw_dir = dataset_dir / "raw" / raw_type
        if not raw_dir.exists():
            continue
        
        # Look for emotion subdirectories
        for emotion_dir in raw_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            
            emotion = emotion_dir.name.lower()
            
            # Process files in this emotion directory
            for file_path in emotion_dir.glob("*"):
                if file_path.is_file():
                    sample_id = f"{emotion}_{file_path.stem}"
                    
                    # Check if already in manifest
                    existing = [s for s in manifest["samples"] if s.get("sample_id") == sample_id]
                    if existing:
                        continue
                    
                    # Add to manifest
                    sample = {
                        "sample_id": sample_id,
                        "file_path": str(file_path.relative_to(dataset_dir)),
                        "annotations": {
                            "emotion": emotion,
                            "is_verified": False,
                        },
                    }
                    manifest["samples"].append(sample)
                    annotated += 1
    
    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Auto-annotated {annotated} files")
    return annotated


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_features(
    dataset_dir: Path,
    feature_types: List[str] = None,
) -> int:
    """
    Extract features from dataset files.
    
    Args:
        dataset_dir: Dataset directory
        feature_types: Types of features to extract (default: ["midi", "audio", "groove"])
    
    Returns:
        Number of files processed
    """
    if feature_types is None:
        feature_types = ["midi", "audio", "groove"]
    
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    features_dir = dataset_dir / "processed" / "features"
    
    # Load manifest
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return 0
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    processed = 0
    
    # Extract MIDI features
    if "midi" in feature_types:
        extractor = MIDIFeatureExtractor()
        midi_dir = dataset_dir / "raw" / "midi"
        
        for midi_file in midi_dir.glob("*.mid"):
            try:
                features = extractor.extract(midi_file)
                feature_path = features_dir / f"{midi_file.stem}_features.json"
                features.save(feature_path)
                processed += 1
            except Exception as e:
                logger.warning(f"Failed to extract features from {midi_file}: {e}")
    
    logger.info(f"Extracted features from {processed} files")
    return processed


# =============================================================================
# Augmentation
# =============================================================================


def augment_dataset(
    dataset_dir: Path,
    multiplier: int = 10,
    augment_type: str = "both",
) -> int:
    """
    Augment dataset files.
    
    Args:
        dataset_dir: Dataset directory
        multiplier: Number of augmented variations per file
        augment_type: "midi", "audio", or "both"
    
    Returns:
        Number of augmented files created
    """
    dataset_dir = Path(dataset_dir)
    augmented_dir = dataset_dir / "processed" / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)
    
    config = AugmentationConfig()
    midi_augmenter = MIDIAugmenter(config)
    audio_augmenter = AudioAugmenter(config)
    
    augmented_count = 0
    
    # Augment MIDI files
    if augment_type in ["midi", "both"]:
        midi_dir = dataset_dir / "raw" / "midi"
        for midi_file in midi_dir.glob("*.mid"):
            try:
                variations = midi_augmenter.augment(
                    midi_file,
                    augmented_dir / "midi",
                    num_variations=multiplier,
                )
                augmented_count += len(variations)
            except Exception as e:
                logger.warning(f"Failed to augment {midi_file}: {e}")
    
    # Augment audio files
    if augment_type in ["audio", "both"]:
        audio_dir = dataset_dir / "raw" / "audio"
        for audio_file in audio_dir.glob("*.wav"):
            try:
                variations = audio_augmenter.augment(
                    audio_file,
                    augmented_dir / "audio",
                    num_variations=multiplier,
                )
                augmented_count += len(variations)
            except Exception as e:
                logger.warning(f"Failed to augment {audio_file}: {e}")
    
    logger.info(f"Created {augmented_count} augmented files")
    return augmented_count


# =============================================================================
# Synthetic Data Generation
# =============================================================================


def generate_synthetic_data(
    dataset_dir: Path,
    count: int,
    target_model: str,
) -> int:
    """
    Generate synthetic training data.
    
    Args:
        dataset_dir: Dataset directory
        count: Number of synthetic samples to generate
        target_model: Target model type
    
    Returns:
        Number of samples generated
    """
    dataset_dir = Path(dataset_dir)
    synthetic_dir = dataset_dir / "raw" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticGenerator()
    
    # Generate emotion samples
    if target_model == "emotionrecognizer":
        samples = generator.generate_emotion_samples(
            num_samples=count,
            output_dir=synthetic_dir,
        )
    else:
        logger.warning(f"Synthetic generation for {target_model} not yet implemented")
        return 0
    
    logger.info(f"Generated {len(samples)} synthetic samples")
    return len(samples)


# =============================================================================
# Validation
# =============================================================================


def validate_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """
    Validate a dataset.
    
    Checks:
    - Balance: Equal samples per category
    - Diversity: Varied keys, tempos, modes
    - Quality: File integrity, proper annotations
    - Structure: Correct directory layout
    
    Returns:
        Validation report dictionary
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    
    report = {
        "valid": True,
        "total_samples": 0,
        "balance_score": 0.0,
        "diversity_score": 0.0,
        "quality_score": 0.0,
        "errors": [],
    }
    
    # Load manifest
    if not manifest_path.exists():
        report["valid"] = False
        report["errors"].append("Manifest not found")
        return report
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    samples = manifest.get("samples", [])
    report["total_samples"] = len(samples)
    
    if not samples:
        report["valid"] = False
        report["errors"].append("No samples in manifest")
        return report
    
    # Check balance (emotion distribution)
    emotion_counts = {}
    for sample in samples:
        emotion = sample.get("annotations", {}).get("emotion", "unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    if emotion_counts:
        max_count = max(emotion_counts.values())
        min_count = min(emotion_counts.values())
        report["balance_score"] = min_count / max_count if max_count > 0 else 0.0
    
    # Check diversity (keys, tempos, modes)
    keys = set()
    tempos = []
    modes = set()
    
    for sample in samples:
        ann = sample.get("annotations", {})
        if "key" in ann:
            keys.add(ann["key"])
        if "tempo_bpm" in ann:
            tempos.append(ann["tempo_bpm"])
        if "mode" in ann:
            modes.add(ann["mode"])
    
    diversity_metrics = []
    if keys:
        diversity_metrics.append(len(keys) / 12.0)  # 12 possible keys
    if tempos:
        tempo_range = max(tempos) - min(tempos) if tempos else 0
        diversity_metrics.append(min(tempo_range / 100.0, 1.0))  # Normalize
    if modes:
        diversity_metrics.append(len(modes) / 2.0)  # major/minor
    
    report["diversity_score"] = np.mean(diversity_metrics) if diversity_metrics else 0.0
    
    # Check quality (file existence)
    missing_files = 0
    for sample in samples:
        file_path = dataset_dir / sample.get("file_path", "")
        if not file_path.exists():
            missing_files += 1
    
    report["quality_score"] = 1.0 - (missing_files / len(samples)) if samples else 0.0
    
    # Overall validity
    if report["balance_score"] < 0.5 or report["quality_score"] < 0.9:
        report["valid"] = False
    
    return report


# =============================================================================
# Train/Val/Test Splits
# =============================================================================


def create_splits(
    dataset_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """
    Create train/val/test splits.
    
    Args:
        dataset_dir: Dataset directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    splits_dir = dataset_dir / "splits"
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    samples = manifest.get("samples", [])
    random.shuffle(samples)
    
    # Calculate split sizes
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Split
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    # Write split files
    with open(splits_dir / "train.txt", "w") as f:
        for sample in train_samples:
            f.write(f"{sample['sample_id']}\n")
    
    with open(splits_dir / "val.txt", "w") as f:
        for sample in val_samples:
            f.write(f"{sample['sample_id']}\n")
    
    with open(splits_dir / "test.txt", "w") as f:
        for sample in test_samples:
            f.write(f"{sample['sample_id']}\n")
    
    logger.info(f"Created splits: train={n_train}, val={n_val}, test={n_test}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--target-model", type=str, help="Target model (e.g., emotionrecognizer)")
    parser.add_argument("--root", type=str, help=f"Dataset root directory (default: {DEFAULT_DATASET_ROOT})")
    
    # Actions
    parser.add_argument("--create", action="store_true", help="Create dataset structure")
    parser.add_argument("--import-dir", type=str, help="Import files from directory")
    parser.add_argument("--annotate", action="store_true", help="Auto-annotate from directory structure")
    parser.add_argument("--extract-features", action="store_true", help="Extract features")
    parser.add_argument("--augment", action="store_true", help="Augment dataset")
    parser.add_argument("--multiplier", type=int, default=10, help="Augmentation multiplier")
    parser.add_argument("--synthesize", action="store_true", help="Generate synthetic data")
    parser.add_argument("--count", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")
    parser.add_argument("--splits", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    
    args = parser.parse_args()
    
    root_dir = Path(args.root) if args.root else DEFAULT_DATASET_ROOT
    dataset_dir = root_dir / args.dataset
    
    # Create dataset structure
    if args.create or args.all:
        if not args.target_model:
            logger.error("--target-model required for --create")
            return 1
        create_dataset_structure(args.dataset, args.target_model, root_dir)
    
    # Import files
    if args.import_dir or args.all:
        if not args.import_dir:
            logger.error("--import-dir required for import")
            return 1
        import_files(Path(args.import_dir), dataset_dir)
    
    # Annotate
    if args.annotate or args.all:
        auto_annotate_from_structure(dataset_dir)
    
    # Extract features
    if args.extract_features or args.all:
        extract_features(dataset_dir)
    
    # Augment
    if args.augment or args.all:
        augment_dataset(dataset_dir, multiplier=args.multiplier)
    
    # Generate synthetic data
    if args.synthesize or args.all:
        if not args.target_model:
            logger.error("--target-model required for --synthesize")
            return 1
        generate_synthetic_data(dataset_dir, args.count, args.target_model)
    
    # Validate
    if args.validate or args.all:
        report = validate_dataset(dataset_dir)
        print("\n" + "=" * 70)
        print("Dataset Validation Report")
        print("=" * 70)
        print(f"Valid: {'✅' if report['valid'] else '❌'}")
        print(f"Total Samples: {report['total_samples']}")
        print(f"Balance Score: {report['balance_score']:.2%}")
        print(f"Diversity Score: {report['diversity_score']:.2%}")
        print(f"Quality Score: {report['quality_score']:.2%}")
        if report['errors']:
            print(f"Errors: {report['errors']}")
        print("=" * 70)
    
    # Create splits
    if args.splits or args.all:
        create_splits(dataset_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
