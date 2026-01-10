#!/usr/bin/env python3
"""
Prepare chord progression dataset for training.

Loads from Data_Files/chord_progressions_db.json and converts to training format.
Creates train/val/test splits.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_chord_progressions(data_file: Path) -> List[Dict]:
    """Load chord progressions from JSON file."""
    if not data_file.exists():
        raise FileNotFoundError(f"Chord progression file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'progressions' in data:
        return data['progressions']
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        # Try to extract progressions from dict
        progressions = []
        for key, value in data.items():
            if isinstance(value, list):
                progressions.extend(value)
            elif isinstance(value, dict) and 'chords' in value:
                progressions.append(value)
        return progressions


def encode_chord_progression(progression: List[str], vocab_size: int = 48) -> np.ndarray:
    """
    Encode chord progression to numerical format.
    
    Args:
        progression: List of chord symbols (e.g., ['C', 'Am', 'F', 'G'])
        vocab_size: Size of chord vocabulary
        
    Returns:
        Encoded progression as numpy array
    """
    # Simple encoding: map each chord to an index
    # In production, would use a proper chord vocabulary mapping
    chord_to_idx = {}
    encoded = []
    
    for chord in progression:
        # Normalize chord symbol (remove extensions, case-insensitive)
        chord_base = chord.strip().split('/')[0].split('(')[0].upper()
        
        if chord_base not in chord_to_idx:
            chord_to_idx[chord_base] = len(chord_to_idx)
            if len(chord_to_idx) >= vocab_size:
                break
        
        encoded.append(chord_to_idx[chord_base])
    
    return np.array(encoded, dtype=np.int32)


def create_sequences(progressions: List[Dict], sequence_length: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training sequences from progressions.
    
    Args:
        progressions: List of progression dictionaries
        sequence_length: Length of input/output sequences
        
    Returns:
        (X, y) tuple of sequences
    """
    X = []
    y = []
    
    for prog in progressions:
        if 'chords' not in prog:
            continue
        
        chords = prog['chords']
        if not isinstance(chords, list):
            continue
        
        # Encode progression
        encoded = encode_chord_progression(chords)
        
        if len(encoded) < sequence_length + 1:
            continue
        
        # Create sliding window sequences
        for i in range(len(encoded) - sequence_length):
            X.append(encoded[i:i + sequence_length])
            y.append(encoded[i + sequence_length])
    
    if not X:
        return np.array([]), np.array([])
    
    return np.array(X), np.array(y)


def split_dataset(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """
    Split dataset into train/val/test.
    
    Args:
        X: Input sequences
        y: Target sequences
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if len(X) == 0:
        return (np.array([]),) * 6
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Split
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    X_train = X_shuffled[:n_train]
    y_train = y_shuffled[:n_train]
    X_val = X_shuffled[n_train:n_train + n_val]
    y_val = y_shuffled[n_train:n_train + n_val]
    X_test = X_shuffled[n_train + n_val:]
    y_test = y_shuffled[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_dataset(X_train, y_train, X_val, y_val, X_test, y_test, output_dir: Path):
    """Save dataset splits to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_test.npy', y_test)
    
    # Save metadata
    metadata = {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'sequence_length': X_train.shape[1] if len(X_train) > 0 else 0,
        'vocab_size': int(y_train.max() + 1) if len(y_train) > 0 else 0,
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_dir}")
    print(f"  Train: {metadata['train_samples']} samples")
    print(f"  Val:   {metadata['val_samples']} samples")
    print(f"  Test:  {metadata['test_samples']} samples")


def main():
    # Paths
    data_file = project_root / 'Data_Files' / 'chord_progressions_db.json'
    output_dir = project_root / 'data' / 'chord_progressions' / 'processed'
    
    print("="*70)
    print("Chord Progression Dataset Preparation")
    print("="*70)
    
    # Load progressions
    print(f"\nLoading chord progressions from: {data_file}")
    progressions = load_chord_progressions(data_file)
    print(f"  Loaded {len(progressions)} progressions")
    
    # Create sequences
    print(f"\nCreating sequences...")
    X, y = create_sequences(progressions, sequence_length=8)
    print(f"  Created {len(X)} sequences")
    
    if len(X) == 0:
        print("\n⚠ No sequences created. Check chord progression format.")
        return 1
    
    # Split dataset
    print(f"\nSplitting dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    
    # Save
    print(f"\nSaving dataset...")
    save_dataset(X_train, y_train, X_val, y_val, X_test, y_test, output_dir)
    
    print("\n✓ Dataset preparation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

