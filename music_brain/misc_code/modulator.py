"""
Creative voice modulation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None
    sf = None

from music_brain.voice.presets import MODULATION_PRESETS


@dataclass
class ModulationSettings:
    formant_shift: float = 0.0
    noise_amount: float = 0.0
    low_pass_hz: Optional[float] = None
    band_limit: Optional[Tuple[int, int]] = None
    saturation: float = 0.0
    bit_depth: Optional[int] = None


def get_modulation_preset(name: str) -> ModulationSettings:
    preset = MODULATION_PRESETS.get(name.lower())
    if not preset:
        raise ValueError(f"Unknown voice modulation preset '{name}'.")
    return ModulationSettings(**preset)


class VoiceModulator:
    """Applies formant, filtering, and saturation effects to vocal stems."""

    def __init__(self, settings: Optional[ModulationSettings] = None):
        self.settings = settings or ModulationSettings()

    def process_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        if librosa is None or sf is None:
            raise ImportError("Install audio extras: pip install -e .[audio]")

        audio, sr = librosa.load(input_path, sr=None)
        processed = self.process_array(audio, sr)
        output = output_path or f"{Path(input_path).stem}_mod.wav"
        sf.write(output, processed, sr)
        return output

    def process_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if librosa is None:
            raise ImportError("Install librosa to modulate voices.")

        y = audio.copy()
        if self.settings.formant_shift:
            y = librosa.effects.pitch_shift(y, sample_rate, n_steps=self.settings.formant_shift)
        if self.settings.band_limit:
            low, high = self.settings.band_limit
            y = self._band_limit(y, sample_rate, low, high)
        if self.settings.low_pass_hz:
            y = self._low_pass(y, sample_rate, self.settings.low_pass_hz)
        if self.settings.bit_depth:
            y = self._bitcrush(y, self.settings.bit_depth)
        if self.settings.saturation:
            sat = self.settings.saturation
            y = np.tanh(y * (1 + sat * 5))
        if self.settings.noise_amount:
            noise = np.random.normal(scale=self.settings.noise_amount, size=y.shape)
            y = y + noise
        return np.clip(y, -1.0, 1.0)

    def _low_pass(self, audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / sr)
        fft[freqs > cutoff_hz] = 0
        return np.fft.irfft(fft, n=audio.size)

    def _band_limit(self, audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / sr)
        mask = (freqs >= low) & (freqs <= high)
        fft[~mask] = 0
        return np.fft.irfft(fft, n=audio.size)

    def _bitcrush(self, audio: np.ndarray, bits: int) -> np.ndarray:
        levels = 2 ** bits
        return np.round(audio * levels) / levels

