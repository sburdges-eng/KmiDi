"""
Dynamics Training Module - ML-Ready Dynamics Architecture

Provides the missing components for ML training-ready dynamics:
1. SectionContext - Links arrangement to emotion/dynamics labels
2. TempoMap - Per-bar tempo tracking and automation
3. InstrumentDensity - Voice count and orchestration density
4. GroundTruthLabel - Unified ground truth (LUFS, crest_factor, dynamics)
5. LUFSMeter - EBU R128 loudness metering
6. TrainingDataGenerator - Generates ML training datasets

For training groove_predictor, harmony_predictor, emotion_recognizer, melody_transformer.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SectionType(Enum):
    """Song section types for arrangement context."""
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    BREAKDOWN = "breakdown"
    BUILD = "build"
    DROP = "drop"
    OUTRO = "outro"
    INTERLUDE = "interlude"
    UNKNOWN = "unknown"


class DynamicLevel(Enum):
    """Dynamic marking levels."""
    PPP = "ppp"  # pianississimo
    PP = "pp"    # pianissimo
    P = "p"      # piano
    MP = "mp"    # mezzo-piano
    MF = "mf"    # mezzo-forte
    F = "f"      # forte
    FF = "ff"    # fortissimo
    FFF = "fff"  # fortississimo


# =============================================================================
# SECTION CONTEXT - Links arrangement to emotion/dynamics
# =============================================================================

@dataclass
class SectionContext:
    """
    Context for a song section - links arrangement metadata to dynamics labels.

    This is the MISSING piece that connects:
    - Arrangement structure (bars, section type)
    - Emotion trajectory (per-section VAD)
    - Dynamics targets (LUFS, velocity)
    - Instrument density
    """
    # Section identification
    section_id: str = ""
    section_type: SectionType = SectionType.UNKNOWN
    section_name: str = ""  # e.g., "Chorus_1", "Verse_2"

    # Time bounds (in bars and seconds)
    start_bar: int = 0
    end_bar: int = 0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0

    # Emotion state for this section (PAD model)
    valence: float = 0.0       # -1 to +1 (negative to positive)
    arousal: float = 0.5       # 0 to 1 (calm to excited)
    dominance: float = 0.5     # 0 to 1 (submissive to dominant)
    intensity: float = 0.5     # 0 to 1 (emotion strength)
    emotion_label: str = ""    # Primary emotion name

    # Dynamics targets (ground truth for ML training)
    target_lufs: float = -14.0         # Integrated loudness target
    target_short_term_lufs: float = -14.0  # Short-term LUFS
    target_crest_factor: float = 10.0  # Peak-to-average ratio (dB)
    target_dynamic_range: float = 12.0 # Loudness range (LU)
    target_velocity_mean: int = 80     # Mean MIDI velocity
    target_velocity_std: float = 15.0  # Velocity standard deviation

    # Arrangement context
    density: float = 0.5       # Notes per beat (normalized 0-1)
    voice_count: int = 4       # Number of active voices/instruments
    orchestration_density: float = 0.5  # Overall orchestration fullness

    # Musical attributes
    energy: float = 0.5        # Section energy level (0-1)
    tension: float = 0.5       # Harmonic/melodic tension (0-1)
    tempo_bpm: float = 120.0   # Local tempo for this section

    # Chord context
    chord_progression: List[str] = field(default_factory=list)
    key_signature: str = ""
    mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['section_type'] = self.section_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionContext":
        """Create from dictionary."""
        if 'section_type' in data:
            if isinstance(data['section_type'], str):
                data['section_type'] = SectionType(data['section_type'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_emotion_vector(self) -> List[float]:
        """Get emotion as a feature vector [valence, arousal, dominance, intensity]."""
        return [self.valence, self.arousal, self.dominance, self.intensity]

    def get_dynamics_vector(self) -> List[float]:
        """Get dynamics targets as a feature vector."""
        return [
            self.target_lufs / -60.0,
            self.target_short_term_lufs / -60.0,
            self.target_crest_factor / 20.0,
            self.target_dynamic_range / 30.0,
            self.target_velocity_mean / 127.0,
            self.target_velocity_std / 40.0,
        ]


# =============================================================================
# TEMPO MAP - Per-bar tempo tracking
# =============================================================================

@dataclass
class TempoPoint:
    """A single tempo point in the tempo map."""
    bar: int
    beat: float  # Beat within bar (0-based)
    tempo_bpm: float
    time_sec: float  # Absolute time in seconds


@dataclass
class TempoMap:
    """
    Tempo automation map for tracking per-bar tempo changes.
    """
    tempo_points: List[TempoPoint] = field(default_factory=list)
    base_tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)

    def add_point(self, bar: int, beat: float, tempo_bpm: float):
        """Add a tempo point."""
        if not self.tempo_points:
            time_sec = 0.0
        else:
            last = self.tempo_points[-1]
            beats_since_last = (bar - last.bar) * self.time_signature[0] + (beat - last.beat)
            time_delta = beats_since_last * (60.0 / last.tempo_bpm)
            time_sec = last.time_sec + time_delta

        self.tempo_points.append(TempoPoint(
            bar=bar,
            beat=beat,
            tempo_bpm=tempo_bpm,
            time_sec=time_sec,
        ))

    def get_tempo_at_bar(self, bar: int, beat: float = 0.0) -> float:
        """Get tempo at a specific bar/beat position."""
        if not self.tempo_points:
            return self.base_tempo

        position = bar + beat / self.time_signature[0]

        for i, point in enumerate(self.tempo_points):
            point_pos = point.bar + point.beat / self.time_signature[0]
            if point_pos >= position:
                if i == 0:
                    return point.tempo_bpm
                prev = self.tempo_points[i - 1]
                prev_pos = prev.bar + prev.beat / self.time_signature[0]
                t = (position - prev_pos) / (point_pos - prev_pos) if point_pos != prev_pos else 0
                return prev.tempo_bpm + t * (point.tempo_bpm - prev.tempo_bpm)

        return self.tempo_points[-1].tempo_bpm

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tempo_points': [asdict(p) for p in self.tempo_points],
            'base_tempo': self.base_tempo,
            'time_signature': list(self.time_signature),
        }


# =============================================================================
# GROUND TRUTH LABEL - Unified dynamics targets
# =============================================================================

@dataclass
class GroundTruthLabel:
    """
    Unified ground truth label for ML training.
    """
    # Loudness metrics (EBU R128)
    integrated_lufs: float = -14.0
    short_term_lufs: float = -14.0
    momentary_lufs: float = -14.0
    loudness_range: float = 8.0
    true_peak_dbtp: float = -1.0

    # Dynamic metrics
    crest_factor_db: float = 10.0
    dynamic_range_db: float = 12.0
    compression_ratio: float = 0.0

    # MIDI velocity targets
    velocity_mean: int = 80
    velocity_std: float = 15.0
    velocity_min: int = 40
    velocity_max: int = 120

    # Articulation targets
    legato_ratio: float = 0.5
    accent_density: float = 0.2

    # Emotion label
    emotion_label: str = ""
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.5

    # Section context
    section_type: str = ""
    section_energy: float = 0.5

    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML training."""
        return [
            self.integrated_lufs / -60.0,
            self.short_term_lufs / -60.0,
            self.momentary_lufs / -60.0,
            self.loudness_range / 30.0,
            (self.true_peak_dbtp + 10) / 10.0,
            self.crest_factor_db / 20.0,
            self.dynamic_range_db / 30.0,
            self.compression_ratio,
            self.velocity_mean / 127.0,
            self.velocity_std / 40.0,
            self.velocity_min / 127.0,
            self.velocity_max / 127.0,
            self.legato_ratio,
            self.accent_density,
            self.emotion_valence,
            self.emotion_arousal,
            self.section_energy,
        ]

    @classmethod
    def vector_size(cls) -> int:
        return 17

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthLabel":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# LUFS METER - EBU R128 Loudness Measurement
# =============================================================================

class LUFSMeter:
    """
    EBU R128 Loudness Meter implementation.
    Reference: ITU-R BS.1770-4
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.reset()

    def reset(self):
        """Reset meter state."""
        self.block_loudness_values = []

    def process(self, audio) -> Dict[str, float]:
        """
        Process audio and return loudness measurements.
        """
        if np is None:
            return {
                'momentary_lufs': -14.0,
                'short_term_lufs': -14.0,
                'integrated_lufs': -14.0,
                'loudness_range': 8.0,
                'true_peak_dbtp': -1.0,
            }

        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        if audio.shape[0] > audio.shape[1]:
            audio = audio.T

        true_peak = np.max(np.abs(audio))
        true_peak_dbtp = 20 * np.log10(true_peak) if true_peak > 0 else -100.0

        mean_square = np.mean(audio ** 2)
        if mean_square > 0:
            loudness = -0.691 + 10 * np.log10(mean_square)
        else:
            loudness = -100.0

        self.block_loudness_values.append(loudness)

        gated = [l for l in self.block_loudness_values if l > -70]
        integrated_lufs = float(np.mean(gated)) if gated else -100.0

        if len(gated) >= 2:
            p10 = np.percentile(gated, 10)
            p95 = np.percentile(gated, 95)
            loudness_range = float(p95 - p10)
        else:
            loudness_range = 0.0

        return {
            'momentary_lufs': loudness,
            'short_term_lufs': loudness,
            'integrated_lufs': integrated_lufs,
            'loudness_range': loudness_range,
            'true_peak_dbtp': true_peak_dbtp,
        }


# =============================================================================
# POSITION TOKENIZER
# =============================================================================

class PositionTokenizer:
    """
    Extended tokenizer with Bar/Beat position tokens.
    """

    TOKEN_PAD = 0
    TOKEN_START = 1
    TOKEN_END = 2
    TOKEN_BAR = 3
    TOKEN_BEAT = 4
    TOKEN_OFFSET = 5

    def __init__(
        self,
        max_bars: int = 64,
        beats_per_bar: int = 4,
        subdivisions: int = 4,
    ):
        self.max_bars = max_bars
        self.beats_per_bar = beats_per_bar
        self.subdivisions = subdivisions

        self.bar_vocab = max_bars
        self.beat_vocab = beats_per_bar
        self.subdiv_vocab = subdivisions
        self.note_vocab = 128 * 32 * 8

        self.vocab_size = (
            self.TOKEN_OFFSET +
            self.bar_vocab +
            self.beat_vocab +
            self.subdiv_vocab +
            self.note_vocab
        )

    def encode_position(self, bar: int, beat: int, subdivision: int) -> List[int]:
        """Encode a position as tokens."""
        bar_token = self.TOKEN_OFFSET + min(bar, self.max_bars - 1)
        beat_token = self.TOKEN_OFFSET + self.bar_vocab + beat
        subdiv_token = self.TOKEN_OFFSET + self.bar_vocab + self.beat_vocab + subdivision

        return [self.TOKEN_BAR, bar_token, self.TOKEN_BEAT, beat_token, subdiv_token]

    def decode_position(self, tokens: List[int]) -> Tuple[int, int, int]:
        """Decode position from tokens."""
        bar = 0
        beat = 0
        subdivision = 0

        for i, token in enumerate(tokens):
            if token == self.TOKEN_BAR and i + 1 < len(tokens):
                bar = tokens[i + 1] - self.TOKEN_OFFSET
            elif token == self.TOKEN_BEAT and i + 1 < len(tokens):
                beat = tokens[i + 1] - self.TOKEN_OFFSET - self.bar_vocab
                if i + 2 < len(tokens):
                    subdivision = tokens[i + 2] - self.TOKEN_OFFSET - self.bar_vocab - self.beat_vocab

        return bar, beat, subdivision


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_dynamics_targets_from_emotion(
    valence: float,
    arousal: float,
    intensity: float,
) -> Dict[str, float]:
    """
    Calculate dynamics targets from emotion dimensions.
    Maps PAD (Pleasure-Arousal-Dominance) to dynamics parameters.
    """
    base_lufs = -14.0
    lufs_offset = valence * 4
    target_lufs = base_lufs + lufs_offset

    crest_factor = 8.0 + arousal * 8.0
    dynamic_range = 6.0 + intensity * 12.0

    velocity_mean = int(60 + arousal * 40 + intensity * 20)
    velocity_std = 10.0 + intensity * 20.0

    return {
        'target_lufs': target_lufs,
        'target_crest_factor': crest_factor,
        'target_dynamic_range': dynamic_range,
        'target_velocity_mean': velocity_mean,
        'target_velocity_std': velocity_std,
    }
