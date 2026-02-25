"""
Audio package exports and compatibility shims.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

from .render import render_midi_to_audio
from .voice_synthesis import synthesize_guide_vocal
from .instrument_synth import synthesize_instruments_from_midi


@dataclass
class AudioFeatures:
    bpm: float = 120.0
    key: str = "C"
    mode: str = "major"

    def to_dict(self) -> Dict[str, Any]:
        return {"bpm": self.bpm, "key": self.key, "mode": self.mode}


@dataclass
class AudioAnalysis:
    bpm: float = 120.0
    key: str = "C"
    mode: str = "major"

    def to_dict(self) -> Dict[str, Any]:
        return {"bpm": self.bpm, "key": self.key, "mode": self.mode}


class AudioAnalyzer:
    """Lightweight compatibility analyzer used by API wrappers."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def detect_bpm(self, samples: np.ndarray, sample_rate: int) -> Tuple[float, Dict[str, Any]]:
        return 120.0, {}

    def detect_key(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, str]:
        return "C", "major"

    def analyze_audio(self, samples: np.ndarray, sample_rate: int) -> AudioAnalysis:
        return AudioAnalysis()

    def analyze_file(self, path: str) -> AudioAnalysis:
        return AudioAnalysis()


def analyze_feel(samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    return {"energy": 0.5, "warmth": 0.5}


__all__ = [
    "AudioAnalyzer",
    "AudioAnalysis",
    "AudioFeatures",
    "analyze_feel",
    "render_midi_to_audio",
    "synthesize_guide_vocal",
    "synthesize_instruments_from_midi",
]
