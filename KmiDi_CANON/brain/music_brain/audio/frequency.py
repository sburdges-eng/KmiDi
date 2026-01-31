"""
Frequency-domain utilities for lightweight audio analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def frequency_to_note_name(frequency: float) -> str:
    """
    Convert a frequency in Hz to the nearest note name.
    """

    if frequency <= 0:
        return "N/A"

    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    index = int(round(midi_note)) % 12
    return NOTE_NAMES[index]


@dataclass
class FrequencySpectrum:
    frequencies: np.ndarray
    magnitudes: np.ndarray


class FrequencyAnalyzer:
    """
    Lightweight FFT utilities for audio feature extraction.
    """

    def fft_analysis(self, audio_data: np.ndarray, sample_rate: int) -> FrequencySpectrum:
        """
        Compute the frequency spectrum for a mono waveform.
        """

        if audio_data.size == 0:
            raise ValueError("Audio data is empty.")

        magnitudes = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(audio_data.size, d=1.0 / sample_rate)
        return FrequencySpectrum(frequencies=freqs, magnitudes=magnitudes)

    def pitch_detection(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Estimate the dominant frequency using the FFT peak.
        """

        spectrum = self.fft_analysis(audio_data, sample_rate)
        if spectrum.magnitudes.size < 2:
            return 0.0

        # Skip DC component
        peak_index = int(np.argmax(spectrum.magnitudes[1:]) + 1)
        return spectrum.frequencies[peak_index]

    def harmonic_content(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Return rough harmonic content percentages for octave bands.
        """

        spectrum = self.fft_analysis(audio_data, sample_rate)
        total_energy = np.sum(spectrum.magnitudes) or 1e-9

        bands = {
            "low": (20, 200),
            "low_mid": (200, 1000),
            "high_mid": (1000, 4000),
            "high": (4000, 12000),
        }

        content = {}
        for band_name, (low, high) in bands.items():
            mask = (spectrum.frequencies >= low) & (spectrum.frequencies < high)
            content[band_name] = float(np.sum(spectrum.magnitudes[mask]) / total_energy)
        return content


__all__ = ["FrequencyAnalyzer", "FrequencySpectrum", "frequency_to_note_name"]

