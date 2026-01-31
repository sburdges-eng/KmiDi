"""Audio processing utilities for ML training."""

import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import soundfile as fallback
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Try to import librosa as another fallback
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def load_audio(
    file_path: str | Path,
    target_sr: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and resample if needed.

    Args:
        file_path: Path to audio file (mp3, wav, aif, etc.)
        target_sr: Target sample rate
        mono: Convert to mono if True
        duration: Max duration in seconds (None for full file)

    Returns:
        Tuple of (waveform tensor, sample_rate)
    """
    file_path = Path(file_path)

    # Try different backends
    waveform = None
    sr = None

    # Try librosa first (best MP3 support)
    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(str(file_path), sr=None, mono=False, duration=duration)
            if y.ndim == 1:
                y = y[np.newaxis, :]
            waveform = torch.from_numpy(y).float()
        except Exception:
            pass

    # Try soundfile for WAV/FLAC
    if waveform is None and SOUNDFILE_AVAILABLE:
        try:
            data, sr = sf.read(str(file_path))
            if data.ndim == 1:
                data = data[:, np.newaxis]
            waveform = torch.from_numpy(data.T).float()
        except Exception:
            pass

    # Fallback to torchaudio (with sox backend if available)
    if waveform is None:
        try:
            waveform, sr = torchaudio.load(file_path, backend="sox")
        except Exception:
            try:
                waveform, sr = torchaudio.load(file_path, backend="soundfile")
            except Exception:
                # Last resort: return silence
                logger.warning("Could not load %s, returning silence", file_path)
                sr = target_sr
                samples = int(target_sr * (duration or 1.0))
                waveform = torch.zeros(1, samples)

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Trim to duration
    if duration is not None:
        max_samples = int(duration * sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    return waveform, sr


def normalize_audio(waveform: torch.Tensor, method: str = "peak") -> torch.Tensor:
    """
    Normalize audio waveform.

    Args:
        waveform: Input waveform tensor
        method: "peak" for peak normalization, "rms" for RMS normalization

    Returns:
        Normalized waveform
    """
    if method == "peak":
        peak = torch.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
    elif method == "rms":
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            waveform = waveform / rms * 0.1  # Target RMS of 0.1
    return waveform


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    power: float = 2.0,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Extract mel spectrogram from waveform.

    Args:
        waveform: Input waveform tensor [channels, samples]
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        f_min: Minimum frequency
        f_max: Maximum frequency (None = sr/2)
        power: Exponent for spectrogram (1=amplitude, 2=power)
        normalized: Apply log normalization

    Returns:
        Mel spectrogram tensor [channels, n_mels, time]
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        power=power,
    )

    mel_spec = mel_transform(waveform)

    if normalized:
        # Convert to log scale (dB)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

    return mel_spec


def extract_mfcc(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_mfcc: int = 40,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Extract MFCC features from waveform.

    Args:
        waveform: Input waveform tensor
        sample_rate: Audio sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        MFCC tensor [channels, n_mfcc, time]
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
        },
    )
    return mfcc_transform(waveform)


def pad_or_trim(
    waveform: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad or trim waveform to target length.

    Args:
        waveform: Input waveform [channels, samples]
        target_length: Target number of samples
        pad_value: Value to use for padding

    Returns:
        Waveform with exactly target_length samples
    """
    current_length = waveform.shape[1]

    if current_length > target_length:
        # Trim
        return waveform[:, :target_length]
    elif current_length < target_length:
        # Pad
        padding = torch.full(
            (waveform.shape[0], target_length - current_length),
            pad_value,
            dtype=waveform.dtype,
        )
        return torch.cat([waveform, padding], dim=1)
    return waveform
