"""
Enhanced Formant Synthesizer - Improved singing voice synthesis

Provides better phoneme synthesis with consonants, formant filtering,
and expression controls. Used for quick previews.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from scipy import signal
import soundfile as sf

from music_brain.voice.phoneme_processor import PhonemeSequence, Phoneme
from music_brain.voice.pitch_controller import PitchController, PitchCurve


@dataclass
class FormantConfig:
    """Configuration for formant synthesis."""
    sample_rate: int = 44100
    formant_emphasis: float = 0.6
    breathiness: float = 0.2
    vibrato_rate: float = 5.0
    vibrato_depth: float = 0.02

    def to_dict(self) -> Dict:
        return {
            "sample_rate": self.sample_rate,
            "formant_emphasis": self.formant_emphasis,
            "breathiness": self.breathiness,
            "vibrato_rate": self.vibrato_rate,
            "vibrato_depth": self.vibrato_depth
        }


# Formant frequencies for vowels (F1, F2, F3 in Hz)
VOWEL_FORMANTS = {
    "AA": (730, 1090, 2440),  # father
    "AE": (660, 1720, 2410),  # cat
    "AH": (730, 1090, 2440),  # but
    "AO": (570, 840, 2410),   # law
    "AW": (660, 1170, 2440),  # cow
    "AY": (660, 1720, 2410),  # hide
    "EH": (530, 1840, 2480),  # red
    "ER": (490, 1350, 1690),  # her
    "EY": (530, 1840, 2480),  # ate
    "IH": (390, 1990, 2550),  # it
    "IY": (270, 2290, 3010),  # eat
    "OW": (570, 840, 2410),   # show
    "OY": (570, 840, 2410),   # toy
    "UH": (440, 1020, 2240),  # book
    "UW": (300, 870, 2240),   # two
}

# Consonant characteristics
CONSONANT_PARAMS = {
    # Stops (plosives)
    "B": {"type": "stop", "freq": 0, "noise": False},
    "D": {"type": "stop", "freq": 0, "noise": False},
    "G": {"type": "stop", "freq": 0, "noise": False},
    "K": {"type": "stop", "freq": 0, "noise": True},
    "P": {"type": "stop", "freq": 0, "noise": True},
    "T": {"type": "stop", "freq": 0, "noise": True},

    # Fricatives
    "CH": {"type": "fricative", "freq": 2000, "noise": True},
    "F": {"type": "fricative", "freq": 1500, "noise": True},
    "HH": {"type": "fricative", "freq": 1000, "noise": True},
    "S": {"type": "fricative", "freq": 6000, "noise": True},
    "SH": {"type": "fricative", "freq": 2500, "noise": True},
    "TH": {"type": "fricative", "freq": 1500, "noise": True},
    "V": {"type": "fricative", "freq": 1500, "noise": False},
    "Z": {"type": "fricative", "freq": 6000, "noise": False},
    "ZH": {"type": "fricative", "freq": 2500, "noise": False},

    # Nasals
    "M": {"type": "nasal", "freq": 300, "noise": False},
    "N": {"type": "nasal", "freq": 300, "noise": False},
    "NG": {"type": "nasal", "freq": 300, "noise": False},

    # Liquids
    "L": {"type": "liquid", "freq": 500, "noise": False},
    "R": {"type": "liquid", "freq": 1500, "noise": False},
    "W": {"type": "glide", "freq": 500, "noise": False},
    "Y": {"type": "glide", "freq": 2000, "noise": False},
}


class SingingSynthesizer:
    """
    Enhanced formant-based singing synthesizer.
    """

    def __init__(self, config: Optional[FormantConfig] = None):
        """
        Initialize synthesizer.

        Args:
            config: Formant synthesis configuration
        """
        self.config = config or FormantConfig()
        self.pitch_controller = PitchController(self.config.sample_rate)

    def synthesize(
        self,
        phoneme_sequence: PhonemeSequence,
        pitch_curve: PitchCurve,
        expression: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Synthesize audio from phonemes and pitch.

        Args:
            phoneme_sequence: Phoneme sequence with timing
            pitch_curve: Pitch curve over time
            expression: Optional expression parameters

        Returns:
            Audio signal
        """
        if expression is None:
            expression = {}

        # Calculate total samples
        total_samples = int(phoneme_sequence.total_duration_ms / 1000.0 * self.config.sample_rate)
        audio = np.zeros(total_samples)

        # Ensure pitch curve matches duration
        if len(pitch_curve.frequencies) < total_samples:
            # Extend pitch curve
            extended_freqs = np.zeros(total_samples)
            extended_freqs[:len(pitch_curve.frequencies)] = pitch_curve.frequencies
            if len(pitch_curve.frequencies) > 0:
                extended_freqs[len(pitch_curve.frequencies):] = pitch_curve.frequencies[-1]
            pitch_curve = PitchCurve(
                frequencies=extended_freqs,
                sample_rate=pitch_curve.sample_rate,
                duration_seconds=total_samples / self.config.sample_rate
            )
        elif len(pitch_curve.frequencies) > total_samples:
            pitch_curve.frequencies = pitch_curve.frequencies[:total_samples]

        # Synthesize each phoneme
        for phoneme in phoneme_sequence.phonemes:
            start_sample = int(phoneme.start_time_ms / 1000.0 * self.config.sample_rate)
            duration_samples = int(phoneme.duration_ms / 1000.0 * self.config.sample_rate)

            if start_sample + duration_samples > total_samples:
                duration_samples = total_samples - start_sample

            if duration_samples <= 0:
                continue

            # Get frequency for this phoneme
            freq_samples = pitch_curve.frequencies[start_sample:start_sample + duration_samples]
            avg_freq = np.mean(freq_samples) if len(freq_samples) > 0 else 220.0

            # Synthesize phoneme
            phoneme_audio = self._synthesize_phoneme(
                phoneme.symbol,
                avg_freq,
                duration_samples,
                freq_samples
            )

            # Add to output
            end_sample = min(start_sample + duration_samples, total_samples)
            audio[start_sample:end_sample] += phoneme_audio[:end_sample - start_sample]

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9

        return audio

    def _synthesize_phoneme(
        self,
        phoneme: str,
        base_frequency: float,
        num_samples: int,
        frequency_curve: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize a single phoneme.

        Args:
            phoneme: Phoneme symbol
            base_frequency: Base frequency in Hz
            num_samples: Number of samples
            frequency_curve: Optional frequency curve for this phoneme

        Returns:
            Audio samples
        """
        t = np.arange(num_samples) / self.config.sample_rate

        if phoneme in VOWEL_FORMANTS:
            # Vowel synthesis
            return self._synthesize_vowel(phoneme, base_frequency, t, frequency_curve)
        elif phoneme in CONSONANT_PARAMS:
            # Consonant synthesis
            return self._synthesize_consonant(phoneme, base_frequency, t)
        else:
            # Silence or unknown
            return np.zeros(num_samples)

    def _synthesize_vowel(
        self,
        phoneme: str,
        base_frequency: float,
        t: np.ndarray,
        frequency_curve: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Synthesize a vowel using formant filtering."""
        # Get formant frequencies
        f1, f2, f3 = VOWEL_FORMANTS[phoneme]

        # Generate glottal pulse
        if frequency_curve is not None and len(frequency_curve) == len(t):
            # Use frequency curve
            waveform = np.zeros(len(t))
            for i, freq in enumerate(frequency_curve):
                if freq > 0:
                    period = 1.0 / freq
                    phase = (t[i] % period) / period
                    pulse = np.sin(np.pi * phase) ** 2 * (1 - phase)
                    waveform[i] = pulse
        else:
            # Constant frequency
            period = 1.0 / base_frequency
            phase = (t % period) / period
            waveform = np.sin(np.pi * phase) ** 2 * (1 - phase)

        # Add harmonics
        for h in range(2, 6):
            if frequency_curve is not None and len(frequency_curve) == len(t):
                harmonic = np.sin(2 * np.pi * frequency_curve * h * t) / h
            else:
                harmonic = np.sin(2 * np.pi * base_frequency * h * t) / h
            waveform += harmonic * 0.3 / h

        # Apply formant filters
        waveform = self._apply_formant_filters(waveform, f1, f2, f3)

        # Add breathiness
        if self.config.breathiness > 0:
            noise = np.random.randn(len(t)) * self.config.breathiness * 0.1
            waveform = waveform + noise

        # Apply envelope
        envelope = self._generate_envelope(len(t))
        waveform = waveform * envelope

        return waveform

    def _synthesize_consonant(
        self,
        phoneme: str,
        base_frequency: float,
        t: np.ndarray
    ) -> np.ndarray:
        """Synthesize a consonant."""
        params = CONSONANT_PARAMS[phoneme]
        num_samples = len(t)

        if params["type"] == "stop":
            # Plosive: brief burst of noise or silence
            if params["noise"]:
                # Voiceless stop: noise burst
                noise = np.random.randn(num_samples) * 0.3
                envelope = np.exp(-t * 50)  # Quick decay
                return noise * envelope
            else:
                # Voiced stop: brief silence or low-frequency pulse
                return np.zeros(num_samples)

        elif params["type"] == "fricative":
            # Fricative: noise filtered at characteristic frequency
            noise = np.random.randn(num_samples)

            # Bandpass filter at characteristic frequency
            if params["freq"] > 0:
                b, a = signal.butter(4, [params["freq"] * 0.7, params["freq"] * 1.3],
                                     btype='band', fs=self.config.sample_rate)
                noise = signal.filtfilt(b, a, noise)

            # Add voicing if not noise-only
            if not params["noise"] and base_frequency > 0:
                voicing = np.sin(2 * np.pi * base_frequency * t) * 0.3
                noise = noise + voicing

            envelope = self._generate_envelope(num_samples)
            return noise * envelope * 0.5

        elif params["type"] == "nasal":
            # Nasal: low-frequency formant
            waveform = np.sin(2 * np.pi * base_frequency * t)
            # Apply nasal formant (around 300 Hz)
            waveform = self._apply_formant_filters(waveform, 300, 1200, 2400)
            envelope = self._generate_envelope(num_samples)
            return waveform * envelope * 0.6

        elif params["type"] in ["liquid", "glide"]:
            # Liquid/glide: formant-like with characteristic frequency
            waveform = np.sin(2 * np.pi * base_frequency * t)
            if params["freq"] > 0:
                waveform = self._apply_formant_filters(waveform, params["freq"], params["freq"] * 2, params["freq"] * 4)
            envelope = self._generate_envelope(num_samples)
            return waveform * envelope * 0.7

        return np.zeros(num_samples)

    def _apply_formant_filters(
        self,
        waveform: np.ndarray,
        f1: float,
        f2: float,
        f3: float
    ) -> np.ndarray:
        """Apply formant filtering to waveform."""
        # Create parallel formant filters
        filtered = np.zeros_like(waveform)

        # Formant 1
        b1, a1 = signal.butter(2, [f1 * 0.8, f1 * 1.2], btype='band', fs=self.config.sample_rate)
        filtered += signal.filtfilt(b1, a1, waveform) * 1.0

        # Formant 2
        b2, a2 = signal.butter(2, [f2 * 0.8, f2 * 1.2], btype='band', fs=self.config.sample_rate)
        filtered += signal.filtfilt(b2, a2, waveform) * 0.6

        # Formant 3
        b3, a3 = signal.butter(2, [f3 * 0.8, f3 * 1.2], btype='band', fs=self.config.sample_rate)
        filtered += signal.filtfilt(b3, a3, waveform) * 0.3

        # Mix with original
        return waveform * (1 - self.config.formant_emphasis) + filtered * self.config.formant_emphasis

    def _generate_envelope(self, num_samples: int) -> np.ndarray:
        """Generate ADSR envelope."""
        attack = int(num_samples * 0.1)
        decay = int(num_samples * 0.1)
        release = int(num_samples * 0.2)
        sustain_level = 0.7

        envelope = np.ones(num_samples)

        # Attack
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)

        # Decay
        if decay > 0 and attack + decay < num_samples:
            envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)

        # Sustain
        sustain_end = num_samples - release
        if sustain_end > attack + decay:
            envelope[attack + decay:sustain_end] = sustain_level

        # Release
        if release > 0 and sustain_end < num_samples:
            envelope[sustain_end:] = np.linspace(sustain_level, 0, release)

        return envelope

    def save_audio(self, audio: np.ndarray, output_path: str) -> None:
        """Save audio to file."""
        try:
            sf.write(output_path, audio, self.config.sample_rate)
        except Exception as e:
            print(f"Error saving audio: {e}")
            # Fallback to manual WAV writing
            self._write_wav_manual(audio, output_path)

    def _write_wav_manual(self, audio: np.ndarray, output_path: str) -> None:
        """Write WAV file manually."""
        import struct

        audio_int = (audio * 32767).astype(np.int16)

        with open(output_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(audio_int) * 2))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<I", self.config.sample_rate))
            f.write(struct.pack("<I", self.config.sample_rate * 2))
            f.write(struct.pack("<H", 2))
            f.write(struct.pack("<H", 16))
            f.write(b"data")
            f.write(struct.pack("<I", len(audio_int) * 2))
            f.write(audio_int.tobytes())
