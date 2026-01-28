"""
Intent-aware vocal synthesis (guide vocals / robotic tones).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover
    sf = None

from music_brain.voice.presets import VOICE_PROFILES


@dataclass
class SynthConfig:
    timbre: str = "breathy"
    vibrato: float = 0.2
    dynamics: float = 0.7


def get_voice_profile(name: str) -> SynthConfig:
    profile = VOICE_PROFILES.get(name)
    if not profile:
        raise ValueError(f"Unknown voice profile '{name}'.")
    return SynthConfig(**profile)


class VoiceSynthesizer:
    """Generates simple guide vocals as WAV files."""

    def __init__(self, config: Optional[SynthConfig] = None):
        self.config = config or SynthConfig()

    def synthesize_guide(
        self,
        lyrics: str,
        melody_midi: Iterable[int],
        tempo_bpm: int = 82,
        output_path: str = "guide_vocal.wav",
        sample_rate: int = 44100,
    ) -> str:
        if sf is None:
            raise ImportError("Install audio extras (soundfile) for synthesis.")

        phonemes = self._lyrics_to_phonemes(lyrics)
        durations = self._estimate_durations(len(melody_midi), tempo_bpm)
        audio = self._render(melody_midi, durations, phonemes, sample_rate)
        sf.write(output_path, audio, sample_rate)
        return output_path

    def _lyrics_to_phonemes(self, lyrics: str) -> List[str]:
        tokens = lyrics.split()
        return [token[:2].lower() or "ah" for token in tokens]

    def _estimate_durations(self, note_count: int, tempo_bpm: int) -> List[float]:
        beat_duration = 60.0 / tempo_bpm
        return [beat_duration] * note_count

    def _render(
        self,
        melody_midi: Iterable[int],
        durations: List[float],
        phonemes: List[str],
        sample_rate: int,
    ) -> np.ndarray:
        audio = np.array([], dtype=np.float32)
        notes = list(melody_midi)
        for idx, (midi_note, duration) in enumerate(zip(notes, durations)):
            freq = 440.0 * (2 ** ((midi_note - 69) / 12))
            samples = int(duration * sample_rate)
            t = np.linspace(0, duration, samples, endpoint=False)
            vibrato = self.config.vibrato * np.sin(2 * np.pi * 5 * t)
            waveform = np.sin(2 * np.pi * (freq + vibrato * 5) * t)
            envelope = self._envelope(samples)
            phoneme_noise = self._phoneme_color(phonemes[idx % len(phonemes)], samples, sample_rate)
            rendered = (waveform * envelope) * self.config.dynamics + phoneme_noise
            audio = np.concatenate([audio, rendered.astype(np.float32)])
        return np.clip(audio, -1.0, 1.0)

    def _envelope(self, samples: int) -> np.ndarray:
        attack = int(samples * 0.05)
        release = int(samples * 0.1)
        sustain = samples - attack - release
        attack_curve = np.linspace(0, 1, max(attack, 1))
        sustain_curve = np.ones(max(sustain, 1))
        release_curve = np.linspace(1, 0, max(release, 1))
        return np.concatenate([attack_curve, sustain_curve, release_curve])[:samples]

    def _phoneme_color(self, phoneme: str, samples: int, sample_rate: int) -> np.ndarray:
        color_noise = np.random.normal(scale=0.005, size=samples)
        if phoneme in {"sh", "ss"}:
            return color_noise * 3
        if phoneme in {"oo", "oh"}:
            return self._low_pass(color_noise, sample_rate, 2000)
        return color_noise

    def _low_pass(self, signal: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / sr)
        fft[freqs > cutoff_hz] = 0
        return np.fft.irfft(fft, n=signal.size)

    def speak_text(
        self,
        text: str,
        output_path: str = "spoken_prompt.wav",
        profile: Optional[str] = None,
        tempo_bpm: int = 80,
        sample_rate: int = 44100,
    ) -> str:
        """
        Text-to-talk helper for announcing names or rap cadences.
        """
        if sf is None:
            raise ImportError("Install audio extras (soundfile) for synthesis.")

        original_config = self.config
        if profile:
            self.config = get_voice_profile(profile)

        words = text.split()
        durations = self._estimate_durations(len(words), tempo_bpm)
        pitches = self._derive_monotone_pitches(len(words))

        audio = self._render(
            melody_midi=pitches,
            durations=durations,
            phonemes=self._lyrics_to_phonemes(text),
            sample_rate=sample_rate,
        )
        sf.write(output_path, audio, sample_rate)
        self.config = original_config
        return output_path

    def _derive_monotone_pitches(self, count: int) -> List[int]:
        base = 67  # G3-ish
        pattern = [0, 2, -1, 3]
        return [base + pattern[i % len(pattern)] for i in range(count)]

