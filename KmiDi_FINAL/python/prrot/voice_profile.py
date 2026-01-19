"""
Voice Profile - Python data structures for voice profiles

Compatible with C++ VoiceProfile structures, using JSON for serialization.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import IntEnum
from pathlib import Path


class PhonemeType(IntEnum):
    """CMU Phoneme types"""

    # Vowels
    AA = 0
    AE = 1
    AH = 2
    AO = 3
    AW = 4
    AY = 5
    EH = 6
    ER = 7
    EY = 8
    IH = 9
    IY = 10
    OW = 11
    OY = 12
    UH = 13
    UW = 14

    # Consonants - Stops
    B = 15
    D = 16
    G = 17
    K = 18
    P = 19
    T = 20

    # Consonants - Fricatives
    CH = 21
    DH = 22
    F = 23
    HH = 24
    JH = 25
    S = 26
    SH = 27
    TH = 28
    V = 29
    Z = 30
    ZH = 31

    # Consonants - Nasals
    M = 32
    N = 33
    NG = 34

    # Consonants - Liquids/Glides
    L = 35
    R = 36
    W = 37
    Y = 38

    # Silence
    SIL = 39

    # Unknown
    UNKNOWN = 255


@dataclass
class PhonemeDurationStats:
    """Phoneme duration statistics"""

    mean_ms: float = 0.0
    variance_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    sample_count: int = 0


@dataclass
class VowelSustainCharacteristics:
    """Vowel sustain characteristics"""

    sustain_multiplier: float = 1.0
    decay_rate: float = 0.0
    formant_stability: float = 0.0


@dataclass
class ConsonantProfile:
    """Consonant attack/release profile"""

    attack_time_ms: float = 0.0
    release_time_ms: float = 0.0
    strength_scalar: float = 1.0
    burst_energy: float = 0.0


@dataclass
class TransitionStats:
    """Transition statistics between phonemes"""

    mean_transition_time_ms: float = 0.0
    variance_ms: float = 0.0
    smoothness: float = 0.0
    sample_count: int = 0


@dataclass
class VibratoCharacteristics:
    """Vibrato characteristics"""

    rate_mean_hz: float = 5.5
    rate_variance_hz: float = 1.0
    depth_mean_cents: float = 15.0
    depth_variance_cents: float = 5.0
    onset_time_ms: float = 50.0
    probability: float = 0.7


@dataclass
class PitchStabilityProfile:
    """Pitch stability profile"""

    drift_rate_cents_per_sec: float = 0.0
    stability_variance: float = 0.0
    correction_rate: float = 0.0


@dataclass
class ArticulationVariance:
    """Articulation variance characteristics"""

    consistency_score: float = 0.0
    variance_range_min: float = 0.0
    variance_range_max: float = 0.0
    per_phoneme_variance: Dict[int, float] = field(default_factory=dict)


@dataclass
class ProminenceCharacteristics:
    """Prominence tendencies"""

    stress_amplitude_multiplier: float = 1.0
    stress_duration_multiplier: float = 1.2
    emphasis_probability: float = 0.3


@dataclass
class BreathNoiseCharacteristics:
    """Breath and noise characteristics"""

    breath_weight: float = 0.1
    noise_floor_db: float = -40.0
    breath_marker_probability: float = 0.2
    typical_breath_duration_ms: float = 200.0


@dataclass
class VoiceProfile:
    """Complete voice profile"""

    # Metadata
    profile_id: str
    profile_name: str = ""
    created_timestamp: int = 0
    version: int = 1

    # Phoneme inventory
    phoneme_inventory: List[int] = field(default_factory=list)  # List of PhonemeType values

    # Phoneme duration distributions
    phoneme_durations: Dict[int, Dict[str, Any]] = field(
        default_factory=dict
    )  # PhonemeType -> PhonemeDurationStats

    # Vowel sustain
    vowel_sustain: Dict[str, Any] = field(
        default_factory=lambda: asdict(VowelSustainCharacteristics())
    )

    # Consonant profiles
    consonant_profiles: Dict[int, Dict[str, Any]] = field(
        default_factory=dict
    )  # PhonemeType -> ConsonantProfile

    # Transition statistics
    transitions: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # "from,to" -> TransitionStats

    # Breath and noise
    breath_noise: Dict[str, Any] = field(
        default_factory=lambda: asdict(BreathNoiseCharacteristics())
    )

    # Vibrato
    vibrato: Dict[str, Any] = field(default_factory=lambda: asdict(VibratoCharacteristics()))

    # Pitch stability
    pitch_stability: Dict[str, Any] = field(default_factory=lambda: asdict(PitchStabilityProfile()))

    # Articulation variance
    articulation_variance: Dict[str, Any] = field(
        default_factory=lambda: asdict(ArticulationVariance())
    )

    # Prominence
    prominence: Dict[str, Any] = field(default_factory=lambda: asdict(ProminenceCharacteristics()))

    # Phoneme bias table
    phoneme_bias: Dict[int, float] = field(default_factory=dict)  # PhonemeType -> bias value

    # Optional: articulation biases (speech impediments, etc.)
    articulation_biases: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        """Create from dictionary"""
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "VoiceProfile":
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))

    def save(self, file_path: Path) -> None:
        """Save to JSON file"""
        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, file_path: Path) -> "VoiceProfile":
        """Load from JSON file"""
        with open(file_path, "r") as f:
            return cls.from_json(f.read())

    def is_valid(self) -> bool:
        """Check if profile is valid"""
        return bool(self.profile_id) and self.version > 0


# asdict is already imported from dataclasses at the top
