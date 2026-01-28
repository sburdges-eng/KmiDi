#!/usr/bin/env python3
"""
Emotion Instrument Library Catalog Tool

Scans the Google Drive emotion instrument library directory and generates
a JSON catalog mapping emotions to instruments to audio files.

Usage:
    python scripts/catalog_emotion_library.py [--output OUTPUT_PATH]
    python scripts/catalog_emotion_library.py --source SOURCE_PATH --output OUTPUT_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Audio file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aiff', '.aif', '.m4a', '.flac', '.ogg'}

# Base emotions from the library structure
BASE_EMOTIONS = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE']

# Instrument categories
INSTRUMENTS = ['drums', 'guitar', 'piano', 'vocals']


def extract_sample_id(filename: str) -> Optional[str]:
    """
    Extract sample ID from filename.

    Examples:
        "27836_InduMetal_Drums002_hearted.wav.mp3" -> "27836"
        "394820_Distorted Wah Growl 1.wav.mp3" -> "394820"
    """
    parts = filename.split('_')
    if parts and parts[0].isdigit():
        return parts[0]
    return None


def get_audio_metadata(file_path: Path) -> Dict[str, any]:
    """
    Get metadata for an audio file.

    Returns dict with filename, size, extension, and sample_id if extractable.
    """
    stat = file_path.stat()
    return {
        "filename": file_path.name,
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": file_path.suffix.lower(),
        "sample_id": extract_sample_id(file_path.name),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def scan_emotion_directory(emotion_dir: Path, emotion_name: str) -> Dict[str, List[Dict]]:
    """
    Scan an emotion directory (e.g., base/ANGRY/) and catalog all instruments.

    Returns structure:
    {
        "drums": [...],
        "guitar": [...],
        "piano": [...],
        "vocals": [...]
    }
    """
    catalog = {}

    # Scan each instrument directory
    for instrument in INSTRUMENTS:
        instrument_dir = emotion_dir / instrument
        if instrument_dir.exists() and instrument_dir.is_dir():
            files = []
            for file_path in instrument_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    files.append(get_audio_metadata(file_path))
            if files:
                catalog[instrument] = sorted(files, key=lambda x: x["filename"])

    return catalog


def scan_scale_library(scale_lib_dir: Path) -> Dict[str, any]:
    """
    Scan the Emotion_Scale_Library directory if present.

    Returns catalog of scale-based samples.
    """
    catalog = {}

    if not scale_lib_dir.exists() or not scale_lib_dir.is_dir():
        return catalog

    # Scan subdirectories in scale library
    for item in scale_lib_dir.iterdir():
        if item.is_dir():
            scale_name = item.name
            files = []
            for file_path in item.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    files.append(get_audio_metadata(file_path))
            if files:
                catalog[scale_name] = sorted(files, key=lambda x: x["filename"])

    return catalog


def generate_catalog(source_path: Path, output_path: Path) -> Dict[str, any]:
    """
    Generate complete catalog from source directory.

    Returns catalog dictionary.
    """
    catalog = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "source_path": str(source_path),
            "schema_version": "1.0.0",
        },
        "emotions": {},
        "scale_library": {},
    }

    emotion_lib_dir = source_path / "iDAW_Samples" / "Emotion_Instrument_Library"

    if not emotion_lib_dir.exists():
        print(f"WARNING: Emotion library directory not found: {emotion_lib_dir}")
        return catalog

    # Scan base emotions
    base_dir = emotion_lib_dir / "base"
    if base_dir.exists() and base_dir.is_dir():
        for emotion in BASE_EMOTIONS:
            emotion_dir = base_dir / emotion
            if emotion_dir.exists() and emotion_dir.is_dir():
                print(f"Scanning base/{emotion}...")
                catalog["emotions"][emotion] = {
                    "level": "base",
                    "instruments": scan_emotion_directory(emotion_dir, emotion)
                }

    # Scan sub emotions
    sub_dir = emotion_lib_dir / "sub"
    if sub_dir.exists() and sub_dir.is_dir():
        for item in sub_dir.iterdir():
            if item.is_dir():
                emotion_name = item.name
                print(f"Scanning sub/{emotion_name}...")
                if emotion_name not in catalog["emotions"]:
                    catalog["emotions"][emotion_name] = {}
                catalog["emotions"][emotion_name]["level"] = "sub"
                catalog["emotions"][emotion_name]["instruments"] = scan_emotion_directory(item, emotion_name)

    # Scan scale library if present
    scale_lib_dir = source_path / "iDAW_Samples" / "Emotion_Scale_Library"
    if scale_lib_dir.exists():
        print("Scanning Emotion_Scale_Library...")
        catalog["scale_library"] = scan_scale_library(scale_lib_dir)

    # Calculate statistics
    total_files = 0
    total_size_mb = 0.0
    emotion_counts = {}

    for emotion, emotion_data in catalog["emotions"].items():
        count = 0
        size = 0.0
        instruments_data = emotion_data.get("instruments", {})
        for instrument, files in instruments_data.items():
            count += len(files)
            size += sum(f["size_mb"] for f in files)
        emotion_counts[emotion] = {"files": count, "size_mb": round(size, 2), "level": emotion_data.get("level", "unknown")}
        total_files += count
        total_size_mb += size

    catalog["metadata"]["statistics"] = {
        "total_files": total_files,
        "total_size_mb": round(total_size_mb, 2),
        "emotions": emotion_counts,
    }

    return catalog


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Catalog emotion instrument library from Google Drive"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/Users/seanburdges/Library/CloudStorage/GoogleDrive-sburdges@gmail.com/My Drive/GOOGLE KELLY INFO",
        help="Source directory path (default: Google Drive location)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: data/emotion_instrument_library_catalog.json)",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"ERROR: Source directory does not exist: {source_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        # Default to project data directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        output_path = project_root / "data" / "emotion_instrument_library_catalog.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning emotion instrument library...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print()

    catalog = generate_catalog(source_path, output_path)

    # Write catalog to JSON
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    print()
    print("Catalog generated successfully!")
    print(f"Total files: {catalog['metadata']['statistics']['total_files']}")
    print(f"Total size: {catalog['metadata']['statistics']['total_size_mb']} MB")
    print()
    print("Emotion breakdown:")
    for emotion, stats in catalog['metadata']['statistics']['emotions'].items():
        print(f"  {emotion}: {stats['files']} files ({stats['size_mb']} MB)")

    print(f"\nCatalog saved to: {output_path}")


if __name__ == "__main__":
    main()
