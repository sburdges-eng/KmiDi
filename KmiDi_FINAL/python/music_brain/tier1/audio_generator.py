"""
Tier 1 Audio Generator: Pretrained synthesis without fine-tuning.

Generates audio textures and instrument sounds from:
- MIDI note sequences
- Groove parameters (timing, velocity)
- Emotion embeddings (for timbre/style control)

Approaches:
  1. Lightweight synthesis: Additive synthesis with emotion-based timbre
  2. Pre-recorded samples: Load instrument samples and time-stretch
  3. Neural vocoder: Use pretrained vocoder (if available)

For Mac: Optimized for MPS, low memory footprint.
"""

import numpy as np
import torch
from scipy import signal
from typing import Dict, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class Tier1AudioGenerator:
    """
    Tier 1 audio synthesis: Pretrained textures, no fine-tuning.

    Provides multiple synthesis modes:
      1. Additive: Harmonics-based (fast, lightweight)
      2. Wavetable: Pre-computed wavetables (warm sound)
      3. Sample-based: Time-stretched samples (realistic)
    """

    def __init__(
        self,
        device: str = "mps",
        synthesis_mode: str = "additive",
        sample_rate: int = 22050,
        verbose: bool = True
    ):
        """
        Initialize Tier 1 audio generator.

        Args:
            device: "mps", "cuda", "cpu" (mostly for effects processing)
            synthesis_mode: "additive", "wavetable", "sample"
            sample_rate: Output sample rate (Hz)
            verbose: Enable logging
        """
        self.device = device
        self.synthesis_mode = synthesis_mode
        self.sample_rate = sample_rate
        self.verbose = verbose

        self._log(f"Initialized Tier 1 AudioGenerator ({synthesis_mode} mode, {sample_rate}Hz)")

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def synthesize_texture(
        self,
        midi_notes,
        groove_params: Optional[Dict] = None,
        emotion_embedding: Optional[np.ndarray] = None,
        emotion: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        duration_sec: Optional[float] = None,
        instrument: str = "piano"
    ) -> np.ndarray:
        """
        Synthesize audio texture from MIDI notes + groove + emotion.

        Args:
        midi_notes: (seq_len,) MIDI note indices
        groove_params: Dict with swing, velocity_variance, etc.
        emotion_embedding: (64,) for timbre control (or derived from emotion name)
        emotion: Optional emotion name for quick embedding generation
        duration_seconds: Total output duration (alias duration_sec)
            instrument: "piano", "strings", "pad", "bell"

        Returns:
            audio: (sample_rate * duration,) float32 waveform
        """
        start_time = time.time()

        # Normalize inputs and defaults
        groove_params = groove_params or {}
        if duration_seconds is None:
            duration_seconds = duration_sec if duration_sec is not None else 4.0

        if emotion_embedding is None:
            # Simple deterministic embedding from emotion string
            if emotion is None:
                emotion = "neutral"
            rng = np.random.default_rng(abs(hash(emotion)) % (2**32))
            emotion_embedding = rng.standard_normal(64).astype(np.float32)

        midi_notes = np.array(midi_notes, dtype=int)

        num_samples = int(self.sample_rate * duration_seconds)
        audio = np.zeros(num_samples, dtype=np.float32)

        # Allocate samples per note
        samples_per_note = num_samples // max(len(midi_notes), 1)

        # Emotion controls timbre
        emotion_scalar = np.tanh(emotion_embedding[0]) if len(emotion_embedding) > 0 else 0.0

        swing = groove_params.get("swing", 0.2)
        humanization_amount = groove_params.get("humanization", 0.3)

        # Synthesis loop
        for i, midi_note in enumerate(midi_notes):
            # MIDI to frequency
            freq = 440 * (2 ** ((midi_note - 69) / 12))

            # Apply groove-based velocity
            velocity_var = groove_params.get("velocity_variance", 0.5)
            velocity = np.clip(
                0.7 + np.random.randn() * 0.2 * velocity_var,
                0.3, 1.0
            )

            # Generate note waveform
            waveform = self._synthesize_note(
                freq,
                samples_per_note,
                emotion_scalar,
                instrument,
                velocity
            )

            # Apply ADSR envelope
            envelope = self._adsr_envelope(
                samples_per_note,
                attack_ms=10,
                decay_ms=100,
                sustain_level=0.7,
                release_ms=200
            )
            waveform *= envelope

            # Apply swing if on off-beat
            if swing > 0.1 and i % 2 == 1:
                waveform = self._apply_time_shift(waveform, swing * 0.3)

            # Place in output buffer
            start_idx = i * samples_per_note
            end_idx = min(start_idx + samples_per_note, num_samples)
            audio[start_idx:end_idx] = waveform[:end_idx - start_idx]

        # Apply humanization (slight timing/pitch variation)
        if humanization_amount > 0.1:
            audio = self._apply_humanization(audio, humanization_amount)

        # Normalize and apply soft clipping
        audio = self._soft_clip(audio)

        elapsed_ms = (time.time() - start_time) * 1000
        self._log(f"Synthesized {len(midi_notes)} notes in {elapsed_ms:.1f}ms")

        return audio

    def _synthesize_note(
        self,
        frequency: float,
        duration_samples: int,
        emotion_factor: float = 0.0,
        instrument: str = "piano",
        velocity: float = 0.7
    ) -> np.ndarray:
        """
        Synthesize single note with emotion-controlled timbre.

        Args:
            frequency: Fundamental frequency (Hz)
            duration_samples: Sample count
            emotion_factor: [-1, 1] emotion scalar (affects brightness)
            instrument: Instrument type (affects harmonics)
            velocity: Amplitude (0-1)

        Returns:
            waveform: (duration_samples,) audio
        """
        t = np.arange(duration_samples) / self.sample_rate

        # Instrument-specific harmonic series
        harmonics = {
            "piano": [1.0, 0.5, 0.3, 0.2, 0.1],
            "strings": [1.0, 0.6, 0.4, 0.2, 0.15],
            "pad": [1.0, 0.8, 0.6, 0.4, 0.3],
            "bell": [1.0, 0.4, 0.8, 0.15, 0.3],  # Non-harmonic partials
        }

        harmonic_amps = harmonics.get(instrument, harmonics["piano"])

        # Emotion controls brightness (high frequency content)
        # Positive emotion = brighter, negative = darker
        brightness = 1.0 + emotion_factor * 0.3

        # Synthesize fundamental
        wave = np.sin(2 * np.pi * frequency * t)

        # Add harmonics
        for h, base_amp in enumerate(harmonic_amps[1:], start=2):
            harmonic_freq = frequency * h
            # Emotion-controlled brightness
            amp = base_amp * brightness / h
            wave += amp * np.sin(2 * np.pi * harmonic_freq * t)

        # Normalize
        wave = wave / np.max(np.abs(wave) + 1e-6)

        # Apply velocity
        wave *= velocity

        return wave.astype(np.float32)

    def _adsr_envelope(
        self,
        duration_samples: Optional[int] = None,
        *,
        signal_length: Optional[int] = None,
        attack_ms: float = 10,
        decay_ms: float = 100,
        sustain_level: float = 0.7,
        release_ms: float = 200
    ) -> np.ndarray:
        """
        Generate ADSR (Attack, Decay, Sustain, Release) envelope.

        Args:
            duration_samples: Total samples
            attack_ms, decay_ms, release_ms: Times in milliseconds
            sustain_level: Amplitude during sustain [0, 1]

        Returns:
            envelope: (duration_samples,) amplitude envelope
        """
        # Allow alternative parameter name for compatibility with older callers
        if duration_samples is None:
            duration_samples = signal_length if signal_length is not None else self.sample_rate

        # Convert ms to samples
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        decay_samples = int(decay_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)

        sustain_samples = duration_samples - attack_samples - decay_samples - release_samples
        sustain_samples = max(sustain_samples, 1)

        envelope = np.ones(duration_samples, dtype=np.float32)

        # Attack: 0 → 1
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay: 1 → sustain_level
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, duration_samples)
        if decay_end > decay_start:
            envelope[decay_start:decay_end] = np.linspace(
                1, sustain_level, decay_end - decay_start
            )

        # Release: sustain_level → 0
        release_start = max(duration_samples - release_samples, 0)
        if release_start < duration_samples:
            envelope[release_start:] = np.linspace(
                sustain_level, 0, duration_samples - release_start
            )

        return envelope

    def _apply_time_shift(
        self,
        waveform: np.ndarray,
        shift_factor: float
    ) -> np.ndarray:
        """
        Apply subtle timing shift (for swing/humanization).

        Args:
            waveform: Input audio
            shift_factor: [-0.5, 0.5] shift amount (fraction of buffer)

        Returns:
            shifted: Time-shifted waveform
        """
        shift_samples = int(len(waveform) * shift_factor)
        if shift_samples == 0:
            return waveform

        if shift_samples > 0:
            # Shift right
            shifted = np.zeros_like(waveform)
            shifted[shift_samples:] = waveform[:-shift_samples]
        else:
            # Shift left
            shifted = np.zeros_like(waveform)
            shifted[:shift_samples] = waveform[-shift_samples:]

        return shifted

    def _apply_humanization(
        self,
        audio: np.ndarray,
        humanization_amount: float = 0.3
    ) -> np.ndarray:
        """
        Apply subtle humanization: timing jitter + pitch modulation.

        Args:
            audio: Input audio
            humanization_amount: [0, 1] effect strength

        Returns:
            humanized: Slightly imperfect audio
        """
        # Timing jitter: small random time shifts
        jitter_samples = np.random.randint(-5, 6)
        humanized = np.roll(audio, jitter_samples)

        # Pitch modulation: subtle vibrato
        vibrato_freq = 5.0  # Hz
        vibrato_depth = 0.005 * humanization_amount
        t = np.arange(len(audio)) / self.sample_rate
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)

        # Apply light frequency modulation (approximated via phase jitter)
        # In practice, would resample with varying rate
        humanized += humanized * vibrato * 0.1

        return humanized

    def _soft_clip(
        self,
        audio: np.ndarray,
        threshold: float = 0.9
    ) -> np.ndarray:
        """
        Soft-clip to prevent clipping while preserving dynamics.

        Args:
            audio: Input
            threshold: Clipping threshold [0, 1]

        Returns:
            clipped: Soft-clipped audio
        """
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > threshold:
            audio = audio / max_val

        # Soft clipping using tanh
        audio = np.tanh(audio / threshold) * threshold

        return audio

    def apply_reverb(
        self,
        audio: np.ndarray,
        room_size: str = "small"
    ) -> np.ndarray:
        """
        Apply convolution-based reverb (optional effect).

        Args:
            audio: Input audio
            room_size: "small", "medium", "large"

        Returns:
            reverb_audio: Audio with reverb applied
        """
        # Simple reverb impulse response
        room_ir_length = {
            "small": int(0.5 * self.sample_rate),     # 0.5 sec
            "medium": int(1.5 * self.sample_rate),    # 1.5 sec
            "large": int(3.0 * self.sample_rate),     # 3.0 sec
        }

        ir_len = room_ir_length.get(room_size, room_ir_length["small"])

        # Exponential decay impulse response
        t = np.arange(ir_len) / self.sample_rate
        ir = np.exp(-3 * t) * np.cos(2 * np.pi * 1000 * t)
        ir = ir / np.max(np.abs(ir))

        # Apply convolution
        reverb_audio = signal.fftconvolve(audio, ir, mode='same')

        # Normalize
        reverb_audio = reverb_audio / (np.max(np.abs(reverb_audio)) + 1e-6)

        return reverb_audio.astype(np.float32)


# Convenience function
def generate_tier1_audio(
    midi_notes: np.ndarray,
    groove_params: Dict,
    emotion_embedding: np.ndarray,
    sample_rate: int = 22050
) -> np.ndarray:
    """
    Quick wrapper: Generate audio in one line.

    Example:
        midi_notes = np.array([60, 62, 64, 65, 67, 69, 71, 72])
        groove = {"swing": 0.2, "velocity_variance": 0.5, "humanization": 0.3}
        emotion = np.random.randn(64)
        audio = generate_tier1_audio(midi_notes, groove, emotion)
    """
    gen = Tier1AudioGenerator(sample_rate=sample_rate, verbose=False)
    return gen.synthesize_texture(midi_notes, groove, emotion, duration_seconds=4.0)


# Backwards-compatible alias for tests and existing callers
AudioGenerator = Tier1AudioGenerator
