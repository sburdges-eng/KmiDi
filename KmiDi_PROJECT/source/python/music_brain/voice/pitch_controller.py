"""
Pitch Controller - MIDI to Frequency Conversion with Expression

Converts MIDI notes to frequency curves with portamento, vibrato,
pitch bends, and other expressive controls.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PitchCurve:
    """Represents a pitch curve over time."""
    frequencies: np.ndarray  # Frequency in Hz at each sample
    sample_rate: int
    duration_seconds: float

    def to_dict(self) -> Dict:
        return {
            "frequencies": self.frequencies.tolist(),
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds
        }


@dataclass
class ExpressionParams:
    """Expression parameters for pitch control."""
    vibrato_rate: float = 5.0  # Hz
    vibrato_depth: float = 0.02  # Semitones
    portamento_time: float = 0.05  # Seconds
    pitch_bend_range: float = 2.0  # Semitones
    dynamics: Optional[List[float]] = None  # Per-note dynamics (0-1)

    def to_dict(self) -> Dict:
        return {
            "vibrato_rate": self.vibrato_rate,
            "vibrato_depth": self.vibrato_depth,
            "portamento_time": self.portamento_time,
            "pitch_bend_range": self.pitch_bend_range,
            "dynamics": self.dynamics
        }


class PitchController:
    """
    Controls pitch with expression (portamento, vibrato, bends).
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize pitch controller.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def midi_to_frequency(self, midi_note: int) -> float:
        """
        Convert MIDI note to frequency.

        Args:
            midi_note: MIDI note number (0-127)

        Returns:
            Frequency in Hz
        """
        return 440.0 * (2 ** ((midi_note - 69) / 12.0))

    def frequency_to_midi(self, frequency: float) -> float:
        """
        Convert frequency to MIDI note (can be fractional).

        Args:
            frequency: Frequency in Hz

        Returns:
            MIDI note number (can be fractional)
        """
        return 69 + 12 * np.log2(frequency / 440.0)

    def create_pitch_curve(
        self,
        midi_notes: List[int],
        note_durations: List[float],
        expression: Optional[ExpressionParams] = None
    ) -> PitchCurve:
        """
        Create pitch curve from MIDI notes with expression.

        Args:
            midi_notes: List of MIDI note numbers
            note_durations: Duration of each note in seconds
            expression: Optional expression parameters

        Returns:
            PitchCurve with frequency over time
        """
        if expression is None:
            expression = ExpressionParams()

        # Calculate total duration
        total_duration = sum(note_durations)
        total_samples = int(total_duration * self.sample_rate)

        # Initialize frequency array
        frequencies = np.zeros(total_samples)

        current_sample = 0

        for i, (midi_note, duration) in enumerate(zip(midi_notes, note_durations)):
            target_freq = self.midi_to_frequency(midi_note)
            note_samples = int(duration * self.sample_rate)

            # Calculate portamento
            if i > 0 and expression.portamento_time > 0:
                prev_freq = self.midi_to_frequency(midi_notes[i - 1])
                portamento_samples = int(expression.portamento_time * self.sample_rate)
                portamento_samples = min(portamento_samples, note_samples // 4)

                # Linear interpolation for portamento
                for j in range(portamento_samples):
                    t = j / portamento_samples
                    freq = prev_freq + (target_freq - prev_freq) * t
                    if current_sample + j < total_samples:
                        frequencies[current_sample + j] = freq

                start_sample = current_sample + portamento_samples
            else:
                start_sample = current_sample

            # Generate vibrato
            vibrato_samples = note_samples - (start_sample - current_sample)
            t = np.arange(vibrato_samples) / self.sample_rate

            # Vibrato modulation
            vibrato = np.sin(2 * np.pi * expression.vibrato_rate * t)
            vibrato_semitones = vibrato * expression.vibrato_depth

            # Apply vibrato to frequency
            for j, semitones in enumerate(vibrato_semitones):
                freq = target_freq * (2 ** (semitones / 12.0))
                if start_sample + j < total_samples:
                    frequencies[start_sample + j] = freq

            current_sample += note_samples

        return PitchCurve(
            frequencies=frequencies,
            sample_rate=self.sample_rate,
            duration_seconds=total_duration
        )

    def add_pitch_bend(
        self,
        pitch_curve: PitchCurve,
        bend_points: List[Tuple[float, float]]
    ) -> PitchCurve:
        """
        Add pitch bends to a pitch curve.

        Args:
            pitch_curve: Base pitch curve
            bend_points: List of (time_seconds, bend_semitones) tuples

        Returns:
            Modified pitch curve
        """
        frequencies = pitch_curve.frequencies.copy()

        for time_sec, bend_semitones in bend_points:
            sample_idx = int(time_sec * self.sample_rate)
            if 0 <= sample_idx < len(frequencies):
                # Apply bend (multiplicative in frequency space)
                frequencies[sample_idx:] *= (2 ** (bend_semitones / 12.0))

        return PitchCurve(
            frequencies=frequencies,
            sample_rate=pitch_curve.sample_rate,
            duration_seconds=pitch_curve.duration_seconds
        )

    def extract_pitch_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch (F0) from audio signal.

        Uses autocorrelation-based pitch detection.

        Args:
            audio: Audio signal (mono)
            sample_rate: Sample rate

        Returns:
            Tuple of (frequencies, times) arrays
        """
        try:
            import librosa
            # Use librosa's pitch detection
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate)
            return f0, times
        except ImportError:
            # Fallback: simple autocorrelation
            return self._simple_pitch_detection(audio, sample_rate)

    def _simple_pitch_detection(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple autocorrelation-based pitch detection (fallback).
        """
        # Frame size for analysis
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.01 * sample_rate)  # 10ms hop

        frequencies = []
        times = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]

            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # Find first peak after zero lag
            min_period = int(sample_rate / 2000)  # Max 2000 Hz
            max_period = int(sample_rate / 80)  # Min 80 Hz

            if max_period < len(autocorr):
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    if peak_idx > 0:
                        freq = sample_rate / peak_idx
                        if 80 <= freq <= 2000:  # Valid range
                            frequencies.append(freq)
                        else:
                            frequencies.append(0.0)
                    else:
                        frequencies.append(0.0)
                else:
                    frequencies.append(0.0)
            else:
                frequencies.append(0.0)

            times.append(i / sample_rate)

        return np.array(frequencies), np.array(times)

    def audio_to_midi_notes(
        self,
        audio: np.ndarray,
        sample_rate: int,
        note_duration: float = 0.25
    ) -> List[int]:
        """
        Extract MIDI notes from audio (sung notes).

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            note_duration: Expected note duration in seconds

        Returns:
            List of MIDI note numbers
        """
        # Extract pitch
        frequencies, times = self.extract_pitch_from_audio(audio, sample_rate)

        # Convert to MIDI notes
        midi_notes = []
        samples_per_note = int(note_duration * sample_rate / (sample_rate / len(frequencies)))

        for i in range(0, len(frequencies), samples_per_note):
            # Get frequency for this time window
            window_freqs = frequencies[i:i + samples_per_note]
            window_freqs = window_freqs[window_freqs > 0]  # Remove unvoiced

            if len(window_freqs) > 0:
                # Use median frequency
                median_freq = np.median(window_freqs)
                midi_note = int(round(self.frequency_to_midi(median_freq)))
                midi_note = max(0, min(127, midi_note))  # Clamp to valid range
                midi_notes.append(midi_note)

        return midi_notes if midi_notes else [60]  # Default to C4 if no notes found
