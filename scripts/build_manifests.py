#!/usr/bin/env python3
"""
Build dataset manifests for v2 training pipeline.

Scans user-provided audio/MIDI directories and produces JSONL manifests
for Spectocloud and MIDI Generator training.

Usage:
    python scripts/build_manifests.py \\
        --audio-root /path/to/audio \\
        --midi-root /path/to/midi \\
        --out-dir data/manifests \\
        --val-split 0.05
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib


def get_file_hash(filepath: str) -> str:
    """Get deterministic hash of filepath for splitting."""
    return hashlib.md5(filepath.encode('utf-8')).hexdigest()


def find_audio_files(root: Path) -> List[Path]:
    """Find all audio files in directory tree."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(root.rglob(f'*{ext}'))
    
    return sorted(audio_files)


def find_midi_files(root: Path) -> List[Path]:
    """Find all MIDI files in directory tree."""
    midi_extensions = {'.mid', '.midi'}
    midi_files = []
    
    for ext in midi_extensions:
        midi_files.extend(root.rglob(f'*{ext}'))
    
    return sorted(midi_files)


def match_audio_midi_pairs(
    audio_files: List[Path],
    midi_files: List[Path]
) -> List[Tuple[Path, Path]]:
    """Match audio and MIDI files by stem (filename without extension)."""
    # Create lookup by stem
    midi_by_stem = {}
    for midi_file in midi_files:
        stem = midi_file.stem
        midi_by_stem[stem] = midi_file
    
    # Match audio files
    pairs = []
    for audio_file in audio_files:
        stem = audio_file.stem
        if stem in midi_by_stem:
            pairs.append((audio_file, midi_by_stem[stem]))
    
    return pairs


def extract_emotion_from_path(filepath: Path) -> Optional[List[float]]:
    """
    Try to extract emotion from directory or filename.
    
    Expected patterns:
    - Directory named: happy/, sad/, angry/, etc.
    - Filename with emotion: track_happy_001.wav
    
    Returns 3-element list [valence, arousal, intensity] or None.
    """
    # Simple emotion mapping (valence, arousal, intensity)
    emotion_map = {
        'happy': [0.8, 0.7, 0.7],
        'joy': [0.9, 0.8, 0.8],
        'sad': [-0.7, -0.5, 0.5],
        'melancholy': [-0.6, -0.4, 0.4],
        'angry': [-0.8, 0.9, 0.9],
        'rage': [-0.9, 1.0, 1.0],
        'calm': [0.3, -0.7, 0.3],
        'peaceful': [0.5, -0.8, 0.3],
        'excited': [0.7, 0.9, 0.8],
        'fearful': [-0.6, 0.8, 0.7],
        'fear': [-0.7, 0.85, 0.75],
        'neutral': [0.0, 0.0, 0.5],
        'tender': [0.6, -0.3, 0.4],
        'energetic': [0.5, 0.8, 0.8],
    }
    
    # Check parent directory name
    parent_name = filepath.parent.name.lower()
    if parent_name in emotion_map:
        return emotion_map[parent_name]
    
    # Check filename
    filename = filepath.stem.lower()
    for emotion, values in emotion_map.items():
        if emotion in filename:
            return values
    
    # Default neutral
    return [0.0, 0.0, 0.5]


def split_train_val(
    items: List,
    val_split: float,
    seed: int = 42
) -> Tuple[List, List]:
    """Deterministically split items into train/val sets."""
    # Create deterministic split based on item hash
    val_items = []
    train_items = []
    
    for item in items:
        # Use first element (path) for hashing if item is tuple
        if isinstance(item, tuple):
            key = str(item[0])
        else:
            key = str(item)
        
        # Use hash to determine split
        hash_val = int(get_file_hash(key), 16)
        if (hash_val % 10000) / 10000.0 < val_split:
            val_items.append(item)
        else:
            train_items.append(item)
    
    return train_items, val_items


def build_spectocloud_manifests(
    audio_root: Optional[Path],
    midi_root: Optional[Path],
    out_dir: Path,
    val_split: float,
    seed: int = 42
):
    """Build Spectocloud training manifests."""
    if not audio_root or not audio_root.exists():
        print(f"Warning: Audio root not found: {audio_root}")
        print("Skipping Spectocloud manifest generation.")
        return
    
    if not midi_root or not midi_root.exists():
        print(f"Warning: MIDI root not found: {midi_root}")
        print("Skipping Spectocloud manifest generation.")
        return
    
    print(f"Building Spectocloud manifests...")
    print(f"  Audio root: {audio_root}")
    print(f"  MIDI root: {midi_root}")
    
    # Find files
    audio_files = find_audio_files(audio_root)
    midi_files = find_midi_files(midi_root)
    
    print(f"  Found {len(audio_files)} audio files")
    print(f"  Found {len(midi_files)} MIDI files")
    
    # Match pairs
    pairs = match_audio_midi_pairs(audio_files, midi_files)
    print(f"  Matched {len(pairs)} audio-MIDI pairs")
    
    if len(pairs) == 0:
        print("  Warning: No matching pairs found!")
        return
    
    # Split
    train_pairs, val_pairs = split_train_val(pairs, val_split, seed)
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Write manifests
    train_manifest = out_dir / 'spectocloud_train.jsonl'
    val_manifest = out_dir / 'spectocloud_val.jsonl'
    
    for manifest_path, pairs_list in [
        (train_manifest, train_pairs),
        (val_manifest, val_pairs)
    ]:
        with open(manifest_path, 'w') as f:
            for audio_path, midi_path in pairs_list:
                # Extract emotion from path
                emotion = extract_emotion_from_path(audio_path)
                
                # Create manifest entry
                entry = {
                    'audio_path': str(audio_path.absolute()),
                    'midi_path': str(midi_path.absolute()),
                    'emotion': emotion,
                    # target_pointcloud_path is optional - will be generated on-the-fly
                }
                
                f.write(json.dumps(entry) + '\n')
        
        print(f"  Wrote: {manifest_path}")


def build_midi_manifests(
    midi_root: Optional[Path],
    out_dir: Path,
    val_split: float,
    seed: int = 42
):
    """Build MIDI Generator training manifests."""
    if not midi_root or not midi_root.exists():
        print(f"Warning: MIDI root not found: {midi_root}")
        print("Skipping MIDI manifest generation.")
        return
    
    print(f"Building MIDI Generator manifests...")
    print(f"  MIDI root: {midi_root}")
    
    # Find MIDI files
    midi_files = find_midi_files(midi_root)
    print(f"  Found {len(midi_files)} MIDI files")
    
    if len(midi_files) == 0:
        print("  Warning: No MIDI files found!")
        return
    
    # Split
    train_files, val_files = split_train_val(midi_files, val_split, seed)
    print(f"  Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Write manifests
    train_manifest = out_dir / 'midi_train.jsonl'
    val_manifest = out_dir / 'midi_val.jsonl'
    
    for manifest_path, files_list in [
        (train_manifest, train_files),
        (val_manifest, val_files)
    ]:
        with open(manifest_path, 'w') as f:
            for midi_path in files_list:
                # Extract emotion from path
                emotion = extract_emotion_from_path(midi_path)
                
                # Create manifest entry
                entry = {
                    'midi_path': str(midi_path.absolute()),
                    'emotion': emotion,
                }
                
                f.write(json.dumps(entry) + '\n')
        
        print(f"  Wrote: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build dataset manifests for v2 training pipeline'
    )
    parser.add_argument(
        '--audio-root',
        type=str,
        help='Root directory containing audio files'
    )
    parser.add_argument(
        '--midi-root',
        type=str,
        help='Root directory containing MIDI files'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/manifests',
        help='Output directory for manifests (default: data/manifests)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.05,
        help='Validation split ratio (default: 0.05)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for deterministic splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    audio_root = Path(args.audio_root) if args.audio_root else None
    midi_root = Path(args.midi_root) if args.midi_root else None
    out_dir = Path(args.out_dir)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Building Dataset Manifests")
    print("=" * 60)
    
    # Build Spectocloud manifests (requires both audio and MIDI)
    if audio_root and midi_root:
        build_spectocloud_manifests(audio_root, midi_root, out_dir, args.val_split, args.seed)
    else:
        print("Skipping Spectocloud manifests (requires --audio-root and --midi-root)")
    
    print()
    
    # Build MIDI manifests (requires only MIDI)
    if midi_root:
        build_midi_manifests(midi_root, out_dir, args.val_split, args.seed)
    else:
        print("Skipping MIDI manifests (requires --midi-root)")
    
    print()
    print("=" * 60)
    print("Manifest generation complete!")
    print(f"Output directory: {out_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
