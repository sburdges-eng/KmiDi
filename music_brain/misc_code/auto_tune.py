"""
Auto-tune style pitch correction utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:  # Optional heavy dependencies
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    librosa = None
    sf = None

from music_brain.audio import AudioAnalyzer
from music_brain.voice.presets import AUTO_TUNE_PRESETS

SCALE_PATTERNS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}


def note_name_to_midi(note: str) -> int:
    return librosa.note_to_midi(note) if librosa else 60


def build_scale(key: str, mode: str) -> List[int]:
    root = note_name_to_midi(key)
    pattern = SCALE_PATTERNS.get(mode.lower(), SCALE_PATTERNS["major"])
    return [(root + interval) % 12 for interval in pattern]


@dataclass
class AutoTuneSettings:
    strength: float = 0.7
    retune_speed: float = 0.5
    vibrato_preserve: float = 0.8
    formant_shift: float = 0.0


def get_auto_tune_preset(name: str) -> AutoTuneSettings:
    preset = AUTO_TUNE_PRESETS.get(name.lower())
    if not preset:
        raise ValueError(f"Unknown auto-tune preset '{name}'.")
    return AutoTuneSettings(**preset)


class AutoTuneProcessor:
    """Offline auto-tune processor aligned with the DAiW intent system."""

    def __init__(self, settings: Optional[AutoTuneSettings] = None):
        self.settings = settings or AutoTuneSettings()
        self.analyzer = AudioAnalyzer()

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        key: Optional[str] = None,
        mode: str = "major",
    ) -> str:
        if librosa is None or sf is None:
            raise ImportError("Install audio extras: pip install -e .[audio]")

        audio, sr = librosa.load(input_path, sr=None)
        corrected = self.process_array(audio, sr, key=key, mode=mode)
        output = output_path or f"{Path(input_path).stem}_tuned.wav"
        sf.write(output, corrected, sr)
        return output

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        key: Optional[str] = None,
        mode: str = "major",
    ) -> np.ndarray:
        if librosa is None:
            raise ImportError("Install librosa to use AutoTuneProcessor.")

        resolved_key = key or self._detect_key(audio, sample_rate)
        scale = build_scale(resolved_key, mode)

        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=2048,
        )

        midi = librosa.hz_to_midi(f0)
        pitch_diffs = self._quantize_diff(midi, scale)
        median_shift = np.nanmedian(pitch_diffs) if pitch_diffs.size else 0.0
        n_steps = median_shift * self.settings.strength
        corrected = librosa.effects.pitch_shift(audio, sample_rate, n_steps=n_steps)
        return corrected

    def _detect_key(self, audio: np.ndarray, sample_rate: int) -> str:
        analysis = self.analyzer.analyze_waveform(audio, sample_rate)
        return analysis.key

    def _quantize_diff(self, midi_track: np.ndarray, scale: Iterable[int]) -> np.ndarray:
        if midi_track is None:
            return np.array([])

        diffs = []
        for midi_val in midi_track:
            if np.isnan(midi_val):
                continue
            pitch_class = int(round(midi_val)) % 12
            nearest = min(scale, key=lambda s: min(abs((pitch_class - s) % 12), abs((s - pitch_class) % 12)))
            diff = (nearest - pitch_class) % 12
            if diff > 6:
                diff -= 12
            diffs.append(diff)
        return np.array(diffs, dtype=float)

