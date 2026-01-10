#!/usr/bin/env python3
"""
Tier 2 Training: Fine-tune with LoRA adapters on custom MIDI dataset.

For Mac M4 Pro: ~2-4 hours training time, 8-16GB RAM usage

Usage:
    python train_tier2_lora.py \\
      --midi-dir /path/to/midi/files \\
      --emotion-dir /path/to/emotion/embeddings \\
      --epochs 10 \\
      --batch-size 8 \\
      --output-dir ./checkpoints/tier2_lora

Prepare emotion embeddings as JSON:
    emotion_dir/
    ├── song1.json  # [64 floats]
    ├── song2.json
    └── ...

Each JSON file should be a 64-dim emotion vector.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.tier2 import Tier2LORAfinetuner
from music_brain.tier1 import Tier1MIDIGenerator


def validate_dataset(midi_dir: Path, emotion_dir: Path) -> int:
    """Validate dataset integrity"""
    midi_files = list(midi_dir.glob("*.mid"))
    emotion_files = [emotion_dir / f"{f.stem}.json" for f in midi_files]

    # Check all pairs exist
    missing = []
    for mid_file, emo_file in zip(midi_files, emotion_files):
        if not emo_file.exists():
            missing.append(f"{mid_file.name} → {emo_file.name}")

    if missing:
        print(f"⚠ Missing emotion files for {len(missing)} MIDI files:")
        for m in missing[:5]:
            print(f"  {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    print(f"✓ Dataset: {len(midi_files)} MIDI files")
    return len(midi_files)


def prepare_emotion_embeddings(emotion_dir: Path) -> None:
    """Generate dummy emotion embeddings if needed"""
    emotion_dir.mkdir(parents=True, exist_ok=True)

    midi_files = list(emotion_dir.parent.glob("*.mid"))

    existing = list(emotion_dir.glob("*.json"))
    if existing:
        print(f"✓ Found {len(existing)} existing emotion embeddings")
        return

    if not midi_files:
        print("⚠ No MIDI files found to create embeddings for")
        return

    print(f"Creating dummy emotion embeddings for {len(midi_files)} MIDI files...")
    for midi_file in midi_files:
        # Create random 64-dim emotion vector
        emotion = np.random.randn(64).astype(float).tolist()

        emotion_file = emotion_dir / f"{midi_file.stem}.json"
        with open(emotion_file, "w") as f:
            json.dump(emotion, f)

    print(f"✓ Created {len(midi_files)} emotion embeddings in {emotion_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Tier 2 LoRA Fine-tuning on Custom MIDI Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_tier2_lora.py --midi-dir ./data/midi --emotion-dir ./data/emotions

  # Advanced: M4 Pro optimized
  python train_tier2_lora.py \\
    --midi-dir ./data/midi \\
    --emotion-dir ./data/emotions \\
    --epochs 10 \\
    --batch-size 8 \\
    --lora-rank 8 \\
    --device mps \\
    --output-dir ./checkpoints/melody_lora
        """
    )

    parser.add_argument("--midi-dir", type=str, required=True,
                       help="Directory with MIDI files")
    parser.add_argument("--emotion-dir", type=str, required=True,
                       help="Directory with emotion JSON files (64-dim vectors)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (8-16 for M4 Pro)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate for LoRA")
    parser.add_argument("--lora-rank", type=int, default=8,
                       help="LoRA rank (4-16 typical)")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                       help="LoRA alpha (usually 2x rank)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, mps, cuda, cpu")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/tier2_lora",
                       help="Where to save checkpoints")
    parser.add_argument("--save-every", type=int, default=2,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--validation-split", type=float, default=0.1,
                       help="Validation set fraction")
    parser.add_argument("--create-dummy-emotions", action="store_true",
                       help="Create dummy emotion embeddings if missing")

    args = parser.parse_args()

    midi_dir = Path(args.midi_dir)
    emotion_dir = Path(args.emotion_dir)

    # Validate paths
    if not midi_dir.exists():
        print(f"✗ MIDI directory not found: {midi_dir}")
        return 1

    emotion_dir.mkdir(parents=True, exist_ok=True)

    # Validate dataset
    print("="*70)
    print("Tier 2 LoRA Fine-tuning: Configuration")
    print("="*70)

    dataset_size = validate_dataset(midi_dir, emotion_dir)

    if dataset_size < 10:
        print("⚠ Small dataset (<10 files); consider adding more training data")

    if args.create_dummy_emotions:
        prepare_emotion_embeddings(emotion_dir)

    # Get MIDI files
    midi_paths = sorted(list(midi_dir.glob("*.mid")))
    emotion_paths = [emotion_dir / f"{f.stem}.json" for f in midi_paths]

    # Filter to only paired files
    midi_emotion_pairs = [
        (m, e) for m, e in zip(midi_paths, emotion_paths)
        if e.exists()
    ]

    if not midi_emotion_pairs:
        print("✗ No MIDI-emotion pairs found!")
        return 1

    midi_paths = [m for m, e in midi_emotion_pairs]
    emotion_paths = [e for m, e in midi_emotion_pairs]

    print(f"✓ Found {len(midi_paths)} MIDI-emotion pairs")
    print()

    # Training configuration
    print("Training Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Learning rate: {args.learning_rate:.2e}")
    print()

    # Load base model
    print("Loading base model...")
    from music_brain.models.melody_transformer import MelodyTransformer
    base_model = MelodyTransformer()

    # Create finetuner
    print("Setting up LoRA fine-tuner...")
    finetuner = Tier2LORAfinetuner(
        base_model=base_model,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        verbose=True
    )

    print()
    print("="*70)
    print("Starting training...")
    print("="*70)
    print()

    # Fine-tune
    try:
        history = finetuner.finetune_on_dataset(
            midi_paths=[str(p) for p in midi_paths],
            emotion_paths=[str(p) for p in emotion_paths],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            save_every_n_epochs=args.save_every,
            validation_split=args.validation_split
        )

        # Optional: Merge and export
        print()
        print("="*70)
        merged_path = Path(args.output_dir) / "merged_final.pt"
        finetuner.merge_and_export(str(merged_path))
        print(f"✓ Merged model saved to {merged_path}")

        print()
        print("✓ Training complete!")
        print(f"  Checkpoints: {args.output_dir}/")
        print(f"  Merged model: {merged_path}")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
