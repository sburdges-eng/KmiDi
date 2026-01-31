"""
Chord detection utilities built on lightweight spectral analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from music_brain.audio.frequency import FrequencyAnalyzer, frequency_to_note_name


@dataclass
class DetectedChord:
    root: str
    quality: str
    confidence: float


class ChordDetector:
    """
    Extremely lightweight chord detection using FFT snapshots.
    """

    def __init__(self) -> None:
        self.freq_analyzer = FrequencyAnalyzer()

    def detect_chords(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_size: int = 4096,
        hop_size: int = 2048,
    ) -> List[DetectedChord]:
        """
        Slide over the waveform and estimate simple triads.
        """

        chords: List[DetectedChord] = []
        if audio_data.size < window_size:
            frequency = self.freq_analyzer.pitch_detection(audio_data, sample_rate)
            chords.append(self._build_chord(frequency, confidence=0.6))
            return chords

        for start in range(0, audio_data.size - window_size, hop_size):
            window = audio_data[start : start + window_size]
            frequency = self.freq_analyzer.pitch_detection(window, sample_rate)
            chords.append(self._build_chord(frequency, confidence=0.4))
        return chords

    def _build_chord(self, frequency: float, confidence: float) -> DetectedChord:
        note = frequency_to_note_name(frequency)
        quality = "major" if note not in {"D#", "G#", "A#"} else "minor"
        return DetectedChord(root=note, quality=quality, confidence=confidence)

    def summarize_progression(self, chords: Sequence[DetectedChord]) -> List[str]:
        """
        Collapse detected chords to a simple progression string list.
        """

        progression: List[str] = []
        last = None
        for chord in chords:
            symbol = f"{chord.root}{'' if chord.quality == 'major' else 'm'}"
            if symbol != last:
                progression.append(symbol)
                last = symbol
        return progression


__all__ = ["ChordDetector", "DetectedChord"]

