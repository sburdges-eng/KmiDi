"""
Instrument Synthesizer - Convert sung notes to different instruments

Takes MIDI notes extracted from voice and synthesizes them as
different instruments (piano, guitar, strings, etc.).
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from scipy import signal
import soundfile as sf

from music_brain.voice.pitch_controller import PitchController


@dataclass
class InstrumentConfig:
    """Configuration for instrument synthesis."""
    sample_rate: int = 44100
    attack_time: float = 0.01
    decay_time: float = 0.1
    sustain_level: float = 0.7
    release_time: float = 0.2
    brightness: float = 0.5  # 0=dark, 1=bright
    harmonics: int = 5  # Number of harmonics

    def to_dict(self) -> Dict:
        return {
            "sample_rate": self.sample_rate,
            "attack_time": self.attack_time,
            "decay_time": self.decay_time,
            "sustain_level": self.sustain_level,
            "release_time": self.release_time,
            "brightness": self.brightness,
            "harmonics": self.harmonics
        }


# Instrument presets
INSTRUMENT_PRESETS = {
    "piano": InstrumentConfig(
        attack_time=0.01,
        decay_time=0.1,
        sustain_level=0.3,
        release_time=0.3,
        brightness=0.6,
        harmonics=8
    ),
    "guitar": InstrumentConfig(
        attack_time=0.02,
        decay_time=0.2,
        sustain_level=0.5,
        release_time=0.4,
        brightness=0.7,
        harmonics=6
    ),
    "strings": InstrumentConfig(
        attack_time=0.1,
        decay_time=0.05,
        sustain_level=0.8,
        release_time=0.3,
        brightness=0.5,
        harmonics=10
    ),
    "flute": InstrumentConfig(
        attack_time=0.05,
        decay_time=0.1,
        sustain_level=0.9,
        release_time=0.2,
        brightness=0.8,
        harmonics=4
    ),
    "trumpet": InstrumentConfig(
        attack_time=0.02,
        decay_time=0.1,
        sustain_level=0.7,
        release_time=0.2,
        brightness=0.9,
        harmonics=8
    ),
    "violin": InstrumentConfig(
        attack_time=0.08,
        decay_time=0.05,
        sustain_level=0.85,
        release_time=0.3,
        brightness=0.6,
        harmonics=12
    ),
}


class InstrumentSynthesizer:
    """
    Synthesizes MIDI notes as different instruments.
    """

    def __init__(self, instrument: str = "piano", sample_rate: int = 44100):
        """
        Initialize instrument synthesizer.

        Args:
            instrument: Instrument name (piano, guitar, strings, etc.)
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate
        self.pitch_controller = PitchController(sample_rate)

        if instrument in INSTRUMENT_PRESETS:
            self.config = INSTRUMENT_PRESETS[instrument]
        else:
            self.config = InstrumentConfig(sample_rate=sample_rate)

    def synthesize_notes(
        self,
        midi_notes: List[int],
        note_durations: List[float],
        velocities: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Synthesize MIDI notes as the configured instrument.

        Args:
            midi_notes: List of MIDI note numbers
            note_durations: Duration of each note in seconds
            velocities: Optional velocity for each note (0-1)

        Returns:
            Audio signal
        """
        if velocities is None:
            velocities = [0.8] * len(midi_notes)

        # Calculate total duration
        total_duration = sum(note_durations)
        total_samples = int(total_duration * self.sample_rate)
        audio = np.zeros(total_samples)

        current_sample = 0

        for midi_note, duration, velocity in zip(midi_notes, note_durations, velocities):
            frequency = self.pitch_controller.midi_to_frequency(midi_note)
            note_samples = int(duration * self.sample_rate)

            # Synthesize note
            note_audio = self._synthesize_note(frequency, note_samples, velocity)

            # Add to output
            end_sample = min(current_sample + note_samples, total_samples)
            audio[current_sample:end_sample] += note_audio[:end_sample - current_sample]

            current_sample += note_samples

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9

        return audio

    def _synthesize_note(
        self,
        frequency: float,
        num_samples: int,
        velocity: float = 0.8
    ) -> np.ndarray:
        """
        Synthesize a single note.

        Args:
            frequency: Frequency in Hz
            num_samples: Number of samples
            velocity: Velocity (0-1)

        Returns:
            Audio samples
        """
        t = np.arange(num_samples) / self.sample_rate

        # Generate harmonics
        waveform = np.zeros(num_samples)

        for h in range(1, self.config.harmonics + 1):
            # Harmonic amplitude based on brightness
            if h == 1:
                amplitude = 1.0
            else:
                # Higher harmonics for brighter sounds
                amplitude = (1.0 / h) * (0.5 + self.config.brightness * 0.5)

            harmonic = np.sin(2 * np.pi * frequency * h * t) * amplitude
            waveform += harmonic

        # Normalize harmonics
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))

        # Apply ADSR envelope
        envelope = self._generate_adsr(num_samples)
        waveform = waveform * envelope * velocity

        # Apply brightness filter (high-frequency emphasis)
        if self.config.brightness > 0.5:
            # High-pass filter for brightness
            cutoff = 2000 + (self.config.brightness - 0.5) * 4000
            b, a = signal.butter(2, cutoff, btype='high', fs=self.sample_rate)
            waveform = signal.filtfilt(b, a, waveform)

        return waveform

    def _generate_adsr(self, num_samples: int) -> np.ndarray:
        """Generate ADSR envelope."""
        attack_samples = int(self.config.attack_time * self.sample_rate)
        decay_samples = int(self.config.decay_time * self.sample_rate)
        release_samples = int(self.config.release_time * self.sample_rate)

        envelope = np.ones(num_samples)

        # Attack
        if attack_samples > 0 and attack_samples < num_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, num_samples - release_samples)
        if decay_end > decay_start:
            envelope[decay_start:decay_end] = np.linspace(
                1, self.config.sustain_level, decay_end - decay_start
            )

        # Sustain
        sustain_start = decay_end
        sustain_end = num_samples - release_samples
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = self.config.sustain_level

        # Release
        if release_samples > 0 and sustain_end < num_samples:
            envelope[sustain_end:] = np.linspace(
                self.config.sustain_level, 0, release_samples
            )

        return envelope

    def save_audio(self, audio: np.ndarray, output_path: str) -> None:
        """Save audio to file."""
        try:
            sf.write(output_path, audio, self.sample_rate)
        except Exception as e:
            print(f"Error saving audio: {e}")


def get_instrument_preset(instrument: str) -> InstrumentConfig:
    """Get instrument preset configuration."""
    if instrument in INSTRUMENT_PRESETS:
        return INSTRUMENT_PRESETS[instrument]
    else:
        return InstrumentConfig()
