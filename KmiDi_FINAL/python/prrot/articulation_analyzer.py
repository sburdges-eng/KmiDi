"""
Articulation Analyzer - Speaker-Specific Articulation Analysis

Analyzes speaker-specific articulation tendencies from reference audio.
"""

import logging
import numpy as np
from typing import Dict, List
from pathlib import Path

from .voice_profile import VoiceProfile, ConsonantProfile, PhonemeType

logger = logging.getLogger(__name__)


class ArticulationAnalyzer:
    """Speaker-specific articulation analyzer"""

    def __init__(self):
        """Initialize articulation analyzer"""
        pass

    def analyze_articulation(
        self,
        audio_samples: np.ndarray,
        sample_rate: int,
        phoneme_sequence: List[PhonemeType],
        phoneme_boundaries: List[tuple],  # (start_ms, end_ms)
    ) -> Dict[int, ConsonantProfile]:
        """
        Analyze speaker-specific articulation for consonants

        Args:
            audio_samples: Audio samples
            sample_rate: Sample rate
            phoneme_sequence: Sequence of phonemes
            phoneme_boundaries: Phoneme boundaries in milliseconds

        Returns:
            Dictionary mapping PhonemeType to ConsonantProfile
        """
        logger.info("Analyzing speaker-specific articulation")

        consonant_profiles = {}

        # Analyze each consonant phoneme
        for i, phoneme in enumerate(phoneme_sequence):
            if phoneme >= PhonemeType.B and phoneme <= PhonemeType.Y:  # Consonants
                start_ms, end_ms = phoneme_boundaries[i]
                start_sample = int((start_ms / 1000.0) * sample_rate)
                end_sample = int((end_ms / 1000.0) * sample_rate)

                if start_sample < len(audio_samples) and end_sample <= len(audio_samples):
                    segment = audio_samples[start_sample:end_sample]
                    profile = self._analyze_consonant_segment(segment, sample_rate, phoneme)
                    consonant_profiles[int(phoneme)] = profile

        return consonant_profiles

    def _analyze_consonant_segment(
        self, segment: np.ndarray, sample_rate: int, phoneme: PhonemeType
    ) -> ConsonantProfile:
        """Analyze a single consonant segment"""
        profile = ConsonantProfile()

        # Analyze attack time (energy rise)
        energy = np.abs(segment)
        energy_threshold = np.max(energy) * 0.1

        attack_samples = 0
        for i, e in enumerate(energy):
            if e > energy_threshold:
                attack_samples = i
                break

        profile.attack_time_ms = (attack_samples / sample_rate) * 1000.0

        # Analyze release time (energy decay)
        release_samples = 0
        for i in range(len(energy) - 1, -1, -1):
            if energy[i] > energy_threshold:
                release_samples = len(energy) - i
                break

        profile.release_time_ms = (release_samples / sample_rate) * 1000.0

        # Analyze strength (peak energy)
        peak_energy = np.max(energy)
        profile.strength_scalar = float(peak_energy)

        # Analyze burst energy (for stops)
        if phoneme in [
            PhonemeType.B,
            PhonemeType.D,
            PhonemeType.G,
            PhonemeType.K,
            PhonemeType.P,
            PhonemeType.T,
        ]:
            # Stops have initial burst
            burst_window = int(sample_rate * 0.01)  # 10ms
            burst_energy = (
                np.max(energy[:burst_window]) if len(energy) > burst_window else peak_energy
            )
            profile.burst_energy = float(burst_energy)

        return profile
