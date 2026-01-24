#!/usr/bin/env python3
"""
Training Data Preparation Helper

Helps prepare datasets for training stub models.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))


def create_dataset_structure(dataset_name: str, base_dir: Path = None):
    """Create directory structure for a dataset"""
    if base_dir is None:
        base_dir = project_root / "data" / "datasets"

    dataset_dir = base_dir / dataset_name

    # Create directory structure
    dirs = [
        dataset_dir / "train",
        dataset_dir / "val",
        dataset_dir / "test",
        dataset_dir / "metadata",
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

    return dataset_dir


def create_instrument_dataset_template(dataset_dir: Path):
    """Create template for instrument recognition dataset"""

    # Create metadata template
    metadata = {
        "dataset_name": "instrument_dataset_v1",
        "description": "Dual-head instrument recognition dataset",
        "version": "1.0.0",
        "total_samples": 0,
        "splits": {"train": 0, "val": 0, "test": 0},
        "annotations": {
            "technical": {
                "instrument_family": {
                    "classes": [
                        "piano",
                        "organ",
                        "guitar",
                        "strings_bowed",
                        "strings_plucked",
                        "brass",
                        "woodwind_reed",
                        "woodwind_flute",
                        "percussion_tuned",
                        "percussion_untuned",
                        "synth_lead",
                        "synth_pad",
                        "synth_bass",
                        "voice",
                        "ethnic",
                        "electronic",
                    ],
                    "count": 16,
                },
                "instrument_specific": {
                    "description": "Specific instrument within family",
                    "count": 32,
                },
                "technique": {
                    "classes": [
                        "sustained",
                        "staccato",
                        "legato",
                        "tremolo",
                        "trill",
                        "glissando",
                        "vibrato",
                        "bowed",
                        "plucked",
                        "picked",
                        "strummed",
                        "hammered",
                        "blown",
                        "slapped",
                        "palm_muted",
                        "harmonics",
                    ],
                    "count": 16,
                },
                "articulation": {
                    "classes": [
                        "normal",
                        "accented",
                        "marcato",
                        "tenuto",
                        "sforzando",
                        "pizzicato",
                        "con_sordino",
                        "flutter_tongue",
                    ],
                    "count": 8,
                },
                "register": {
                    "description": "Pitch register (low, mid, high, very_high)",
                    "count": 4,
                },
                "timbre_brightness": {"description": "Continuous value 0-1", "type": "regression"},
            },
            "emotional": {
                "expression_style": {
                    "classes": [
                        "aggressive",
                        "gentle",
                        "melancholic",
                        "joyful",
                        "mysterious",
                        "triumphant",
                        "playful",
                        "solemn",
                        "passionate",
                        "contemplative",
                        "anxious",
                        "serene",
                    ],
                    "count": 12,
                },
                "energy_level": {"description": "Ordinal 0-7", "type": "ordinal", "count": 8},
                "musical_role": {
                    "classes": [
                        "lead_melody",
                        "counter_melody",
                        "harmony",
                        "bass",
                        "rhythm",
                        "pad",
                        "accent",
                        "ambient",
                    ],
                    "count": 8,
                },
                "sentiment_valence": {"description": "Continuous -1 to 1", "type": "regression"},
                "sentiment_arousal": {"description": "Continuous 0 to 1", "type": "regression"},
                "dynamic_range": {"description": "Ordinal 0-3", "type": "ordinal", "count": 4},
                "human_feel": {"description": "Continuous 0-1", "type": "regression"},
                "vibrato_intensity": {"description": "Ordinal 0-3", "type": "ordinal", "count": 4},
                "attack_character": {"description": "Continuous 0-1", "type": "regression"},
                "sustain_character": {"description": "Continuous 0-1", "type": "regression"},
                "genre_feel": {"classes": ["classical", "jazz", "rock", "electronic"], "count": 4},
            },
        },
        "audio_format": {
            "sample_rate": 22050,
            "channels": 1,
            "duration_seconds": 3.0,
            "format": "wav",
        },
        "file_structure": {
            "audio_files": "train/audio/",
            "annotations": "train/annotations.json",
            "metadata": "metadata/",
        },
    }

    # Save metadata
    metadata_file = dataset_dir / "metadata" / "dataset_info.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Created metadata template: {metadata_file}")

    # Create annotation template
    annotation_template = {
        "audio_file": "example.wav",
        "technical": {
            "instrument_family": 0,
            "instrument_specific": 0,
            "technique": 0,
            "articulation": 0,
            "register": 0,
            "timbre_brightness": 0.5,
        },
        "emotional": {
            "expression_style": 0,
            "energy_level": 4,
            "musical_role": 0,
            "sentiment_valence": 0.0,
            "sentiment_arousal": 0.5,
            "dynamic_range": 2,
            "human_feel": 0.5,
            "vibrato_intensity": 1,
            "attack_character": 0.5,
            "sustain_character": 0.5,
            "genre_feel": 0,
        },
    }

    annotation_file = dataset_dir / "metadata" / "annotation_template.json"
    with open(annotation_file, "w") as f:
        json.dump(annotation_template, f, indent=2)

    print(f"✅ Created annotation template: {annotation_file}")

    return dataset_dir


def create_emotion_thesaurus_dataset_template(dataset_dir: Path):
    """Create template for emotion thesaurus dataset"""

    # Create metadata template
    metadata = {
        "dataset_name": "emotion_thesaurus_dataset_v1",
        "description": "6×6×6 emotion thesaurus classification dataset",
        "version": "1.0.0",
        "total_samples": 0,
        "splits": {"train": 0, "val": 0, "test": 0},
        "thesaurus_structure": {
            "base_emotions": 6,
            "sub_emotions": 36,
            "sub_sub_emotions": 216,
            "intensity_tiers": 6,
            "keys": 24,
        },
        "labels": {
            "base_emotions": ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST"],
            "intensity_tiers": ["subtle", "mild", "moderate", "strong", "intense", "overwhelming"],
            "keys": [
                "C_major",
                "C_minor",
                "Db_major",
                "Db_minor",
                "D_major",
                "D_minor",
                "Eb_major",
                "Eb_minor",
                "E_major",
                "E_minor",
                "F_major",
                "F_minor",
                "Gb_major",
                "Gb_minor",
                "G_major",
                "G_minor",
                "Ab_major",
                "Ab_minor",
                "A_major",
                "A_minor",
                "Bb_major",
                "Bb_minor",
                "B_major",
                "B_minor",
            ],
        },
        "node_id_formula": "base_idx * 36 + sub_idx * 6 + subsub_idx",
        "audio_format": {
            "sample_rate": 22050,
            "channels": 1,
            "duration_seconds": 3.0,
            "format": "wav",
        },
        "file_structure": {
            "audio_files": "train/audio/",
            "annotations": "train/annotations.json",
            "metadata": "metadata/",
        },
    }

    # Save metadata
    metadata_file = dataset_dir / "metadata" / "dataset_info.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Created metadata template: {metadata_file}")

    # Create annotation template
    annotation_template = {
        "audio_file": "example.wav",
        "emotion_node": {"base_emotion": 0, "sub_emotion": 0, "sub_sub_emotion": 0, "node_id": 0},
        "intensity_tier": 0,
        "key": 0,
        "hierarchical_labels": {
            "base": "HAPPY",
            "sub": "HAPPY_0",
            "sub_sub": "HAPPY_0_0",
            "intensity": "subtle",
            "key": "C_major",
        },
    }

    annotation_file = dataset_dir / "metadata" / "annotation_template.json"
    with open(annotation_file, "w") as f:
        json.dump(annotation_template, f, indent=2)

    print(f"✅ Created annotation template: {annotation_file}")

    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument(
        "--dataset",
        choices=["instrument", "emotion_thesaurus"],
        required=True,
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory (default: data/datasets)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING DATA PREPARATION")
    print("=" * 80)
    print()

    # Create dataset structure
    dataset_dir = create_dataset_structure(
        (
            f"{args.dataset}_dataset_v1"
            if args.dataset == "instrument"
            else "emotion_thesaurus_dataset_v1"
        ),
        args.output_dir,
    )

    print()
    print(f"Preparing {args.dataset} dataset...")

    if args.dataset == "instrument":
        create_instrument_dataset_template(dataset_dir)
    else:
        create_emotion_thesaurus_dataset_template(dataset_dir)

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print(f"1. Add audio files to: {dataset_dir}/train/audio/")
    print(
        f"2. Create annotations.json with format from: {dataset_dir}/metadata/annotation_template.json"
    )
    print(f"3. Split data into train/val/test")
    print(f"4. Update dataset_info.json with actual counts")
    print()
    print("For training instructions, see:")
    print("  docs/MODEL_SETUP_GUIDE.md")
    print()
    print(f"Dataset directory: {dataset_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
