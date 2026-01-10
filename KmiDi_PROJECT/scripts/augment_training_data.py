#!/usr/bin/env python3
"""
Training data augmentation pipeline.

Applies:
- Pitch shifting
- Time stretching
- Noise injection
- Volume variation
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from penta_core.dsp.parrot_dsp import phase_vocoder_pitch_shift, time_stretch


def augment_pitch_shift(audio: np.ndarray, sample_rate: float, semitones_range: Tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    """Apply random pitch shifting."""
    semitones = random.uniform(*semitones_range)
    audio_list = audio.tolist()
    augmented = phase_vocoder_pitch_shift(
        audio_list,
        semitones,
        sample_rate=sample_rate,
        preserve_formants=False,
    )
    return np.array(augmented, dtype=audio.dtype)


def augment_time_stretch(audio: np.ndarray, factor_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Apply random time stretching."""
    factor = random.uniform(*factor_range)
    audio_list = audio.tolist()
    augmented = time_stretch(
        audio_list,
        factor,
        sample_rate=44100.0,
        use_phase_vocoder=True,
    )
    return np.array(augmented, dtype=audio.dtype)


def augment_add_noise(audio: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """Add random noise."""
    noise = np.random.randn(*audio.shape) * noise_level
    return audio + noise


def augment_volume(audio: np.ndarray, db_range: Tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
    """Apply random volume variation."""
    db = random.uniform(*db_range)
    gain = 10 ** (db / 20.0)
    return audio * gain


def augment_audio(audio: np.ndarray, sample_rate: float, apply_all: bool = False) -> List[np.ndarray]:
    """
    Apply augmentation to audio.
    
    Args:
        audio: Input audio samples
        sample_rate: Sample rate
        apply_all: If True, apply all augmentations. If False, randomly select one.
        
    Returns:
        List of augmented audio samples
    """
    augmentations = []
    
    if apply_all:
        # Apply all augmentations
        aug = augment_pitch_shift(audio, sample_rate)
        augmentations.append(aug)
        
        aug = augment_time_stretch(audio)
        augmentations.append(aug)
        
        aug = augment_add_noise(audio)
        augmentations.append(aug)
        
        aug = augment_volume(audio)
        augmentations.append(aug)
    else:
        # Randomly select one augmentation
        augment_func = random.choice([
            lambda: augment_pitch_shift(audio, sample_rate),
            lambda: augment_time_stretch(audio),
            lambda: augment_add_noise(audio),
            lambda: augment_volume(audio),
        ])
        augmentations.append(augment_func())
    
    return augmentations


def main():
    print("="*70)
    print("Training Data Augmentation Pipeline")
    print("="*70)
    print("\nThis script augments training data using:")
    print("  - Pitch shifting (phase vocoder)")
    print("  - Time stretching (phase vocoder)")
    print("  - Noise injection")
    print("  - Volume variation")
    print("\nNote: Run this on your training dataset files.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

