"""
Audio Edge Maps: structural proxies (onset envelope, spectral flux) for structure-centric alignment.

Extract explicit, symbolic structural markers from audio so the LALM can align to temporal
boundaries (drops, chord changes, beat grid) instead of only dense mel features. Used by
StructXLIP training to reduce temporal blur and ground instructions like "drop at bar 16, beat 3".
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def extract_onset_envelope(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Onset strength envelope (1D) aligned to time frames.

    High values at transients; use as structural proxy for "drop" / "hit" boundaries.
    """
    if _HAS_LIBROSA:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        return onset_env.astype(np.float32)
    # Fallback: simple energy derivative (half-wave rectified)
    n_frames = (len(y) - n_fft) // hop_length + 1
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        chunk = y[start : start + n_fft]
        if len(chunk) >= n_fft:
            energy[i] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
    diff = np.diff(energy, prepend=energy[0])
    return np.maximum(diff, 0.0).astype(np.float32)


def extract_spectral_flux(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: Optional[int] = 80,
) -> np.ndarray:
    """
    Frame-to-frame spectral flux (1D per frame): L2 change in magnitude spectrum (or mel).

    Peaks at chord/spectral change boundaries. Returns one value per hop_length frame.
    """
    if _HAS_LIBROSA:
        if n_mels:
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels
            )
        else:
            S = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft))
        S = S.astype(np.float32)
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        # Same length as frames: prepend zero for first frame
        flux = np.concatenate([[0.0], flux])
        return flux.astype(np.float32)
    # Fallback: STFT magnitude diff by hand
    n_frames = (len(y) - n_fft) // hop_length + 1
    n_bins = n_fft // 2 + 1
    mag = np.zeros((n_bins, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        chunk = y[start : start + n_fft]
        if len(chunk) == n_fft:
            windowed = chunk * np.hanning(n_fft)
            fft = np.fft.rfft(windowed)
            mag[:, i] = np.abs(fft)
    diff = np.diff(mag, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    flux = np.concatenate([[0.0], flux]).astype(np.float32)
    return flux


def extract_audio_edge_maps(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: Optional[int] = 80,
    include_onset: bool = True,
    include_flux: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract onset envelope and spectral flux as Audio Edge Maps (aligned to same time grid).

    Returns dict with keys "onset_envelope", "spectral_flux" (each 1D, same number of frames).
    Use these as structural proxies to align with structural text in StructXLIP losses.
    """
    out: Dict[str, np.ndarray] = {}
    if include_onset:
        out["onset_envelope"] = extract_onset_envelope(y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    if include_flux:
        out["spectral_flux"] = extract_spectral_flux(
            y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels
        )
    # Ensure same length (flux may have one extra frame from diff)
    if "onset_envelope" in out and "spectral_flux" in out:
        n = min(len(out["onset_envelope"]), len(out["spectral_flux"]))
        out["onset_envelope"] = out["onset_envelope"][:n]
        out["spectral_flux"] = out["spectral_flux"][:n]
    return out
