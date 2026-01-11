"""
Lightweight audio augmentations for training loops.

Includes:
- Time-stretch
- Pitch-shift
- Simple EQ-like filtering

All functions operate on waveform numpy arrays and a sample rate.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import librosa


@dataclass
class AugmentationConfig:
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    pitch_shift_semitones: Tuple[float, float] = (-2.0, 2.0)
    eq_low_gain_db: Tuple[float, float] = (-3.0, 3.0)
    eq_high_gain_db: Tuple[float, float] = (-3.0, 3.0)
    apply_prob: float = 0.8


def _random_in_range(rng: np.random.Generator, span: Tuple[float, float]) -> float:
    return float(rng.uniform(span[0], span[1]))


def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def eq_tilt(audio: np.ndarray, sr: int, low_gain_db: float, high_gain_db: float) -> np.ndarray:
    """
    Simple tilt EQ: apply low-shelf/high-shelf style gains via FFT magnitude shaping.
    """
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    # Normalize to 0..1 and interpolate gains linearly
    norm = np.clip(freqs / freqs.max(), 0.0, 1.0)
    gains = low_gain_db + (high_gain_db - low_gain_db) * norm
    amp = 10 ** (gains / 20.0)
    processed = np.fft.irfft(spectrum * amp, n=len(audio))
    return processed.astype(audio.dtype, copy=False)


def build_augmentation_pipeline(
    cfg: AugmentationConfig, seed: Optional[int] = None
) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Returns a callable that applies random augmentations with the given config.
    """
    rng = np.random.default_rng(seed)

    def apply(audio: np.ndarray, sr: int) -> np.ndarray:
        if rng.random() > cfg.apply_prob:
            return audio

        augmented = audio
        # Time-stretch
        rate = _random_in_range(rng, cfg.time_stretch_range)
        if abs(rate - 1.0) > 1e-3:
            augmented = time_stretch(augmented, sr=sr, rate=rate)

        # Pitch-shift
        semitones = _random_in_range(rng, cfg.pitch_shift_semitones)
        if abs(semitones) > 1e-3:
            augmented = pitch_shift(augmented, sr=sr, semitones=semitones)

        # EQ tilt
        low_db = _random_in_range(rng, cfg.eq_low_gain_db)
        high_db = _random_in_range(rng, cfg.eq_high_gain_db)
        if abs(low_db) > 0.25 or abs(high_db) > 0.25:
            augmented = eq_tilt(augmented, sr=sr, low_gain_db=low_db, high_gain_db=high_db)

        return augmented

    return apply


__all__ = [
    "AugmentationConfig",
    "build_augmentation_pipeline",
    "time_stretch",
    "pitch_shift",
    "eq_tilt",
]
