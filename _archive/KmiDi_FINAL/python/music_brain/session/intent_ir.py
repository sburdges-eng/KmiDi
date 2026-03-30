"""
Intent IR v1 - Python Types and Utilities

Python dataclass mirror of C IntentFrame struct for type hints and JSON serialization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import IntEnum
import json
import hashlib


class IntentSource(IntEnum):
    """Intent source enum matching C definition."""
    UI_DIRECT = 0
    UI_EDIT = 1
    ML_TEXT = 2
    ML_AUDIO = 3
    PRESET = 4
    AUTOMATION = 5


@dataclass
class IntentMeta:
    """Metadata for routing, compatibility, and debugging."""
    ir_version: int = 1
    intent_id: int = 0
    session_id: int = 0


@dataclass
class EmotionState:
    """Emotion state with VAD and optional discrete mapping."""
    valence: float = 0.0      # [-1.0, 1.0]
    arousal: float = 0.5     # [0.0, 1.0]
    dominance: float = 0.5   # [0.0, 1.0]
    discrete_id: int = -1     # -1 if unused
    intensity: float = 0.0    # [0.0, 1.0]
    confidence: float = 1.0   # [0.0, 1.0]


@dataclass
class MusicalIntent:
    """Musical intent biases and tendencies."""
    tempo_bias: float = 0.0          # -1.0 → +1.0
    rhythmic_density: float = 0.5    # 0.0 → 1.0
    groove_strength: float = 0.5     # 0.0 → 1.0
    harmonic_tension: float = 0.5     # 0.0 → 1.0
    harmonic_motion: float = 0.5      # 0.0 → 1.0
    mode_preference: int = 0          # -1, 0, +1
    melodic_activity: float = 0.5    # 0.0 → 1.0
    contour_variance: float = 0.5    # 0.0 → 1.0
    dynamic_range: float = 0.5       # 0.0 → 1.0
    texture_density: float = 0.5     # 0.0 → 1.0


@dataclass
class TimeScope:
    """Time scope for intent application."""
    start_bar: int = -1        # -1 = immediate
    end_bar: int = -1          # -1 = open-ended
    fade_in_beats: float = 0.0
    fade_out_beats: float = 0.0


@dataclass
class IntentConstraints:
    """Constraints to limit generation."""
    allowed_engines_mask: int = 0xFFFFFFFF
    forbidden_engines_mask: int = 0
    max_cpu_cost: float = 1.0
    max_event_rate: float = 1000.0


@dataclass
class IntentProvenance:
    """Provenance tracking for debugging and trust."""
    source: IntentSource = IntentSource.UI_DIRECT
    user_override_weight: float = 0.5  # 0.0 → 1.0


@dataclass
class IntentFrame:
    """Top-level Intent IR frame."""
    meta: IntentMeta = field(default_factory=IntentMeta)
    emotion: EmotionState = field(default_factory=EmotionState)
    music: MusicalIntent = field(default_factory=MusicalIntent)
    time: TimeScope = field(default_factory=TimeScope)
    constraints: IntentConstraints = field(default_factory=IntentConstraints)
    provenance: IntentProvenance = field(default_factory=IntentProvenance)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "meta": {
                "ir_version": self.meta.ir_version,
                "intent_id": self.meta.intent_id,
                "session_id": self.meta.session_id,
            },
            "emotion": {
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
                "dominance": self.emotion.dominance,
                "discrete_id": self.emotion.discrete_id,
                "intensity": self.emotion.intensity,
                "confidence": self.emotion.confidence,
            },
            "music": {
                "tempo_bias": self.music.tempo_bias,
                "rhythmic_density": self.music.rhythmic_density,
                "groove_strength": self.music.groove_strength,
                "harmonic_tension": self.music.harmonic_tension,
                "harmonic_motion": self.music.harmonic_motion,
                "mode_preference": self.music.mode_preference,
                "melodic_activity": self.music.melodic_activity,
                "contour_variance": self.music.contour_variance,
                "dynamic_range": self.music.dynamic_range,
                "texture_density": self.music.texture_density,
            },
            "time": {
                "start_bar": self.time.start_bar,
                "end_bar": self.time.end_bar,
                "fade_in_beats": self.time.fade_in_beats,
                "fade_out_beats": self.time.fade_out_beats,
            },
            "constraints": {
                "allowed_engines_mask": self.constraints.allowed_engines_mask,
                "forbidden_engines_mask": self.constraints.forbidden_engines_mask,
                "max_cpu_cost": self.constraints.max_cpu_cost,
                "max_event_rate": self.constraints.max_event_rate,
            },
            "provenance": {
                "source": self.provenance.source.value,
                "user_override_weight": self.provenance.user_override_weight,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentFrame":
        """Create from dictionary."""
        return cls(
            meta=IntentMeta(
                ir_version=data.get("meta", {}).get("ir_version", 1),
                intent_id=data.get("meta", {}).get("intent_id", 0),
                session_id=data.get("meta", {}).get("session_id", 0),
            ),
            emotion=EmotionState(
                valence=data.get("emotion", {}).get("valence", 0.0),
                arousal=data.get("emotion", {}).get("arousal", 0.5),
                dominance=data.get("emotion", {}).get("dominance", 0.5),
                discrete_id=data.get("emotion", {}).get("discrete_id", -1),
                intensity=data.get("emotion", {}).get("intensity", 0.0),
                confidence=data.get("emotion", {}).get("confidence", 1.0),
            ),
            music=MusicalIntent(
                tempo_bias=data.get("music", {}).get("tempo_bias", 0.0),
                rhythmic_density=data.get("music", {}).get("rhythmic_density", 0.5),
                groove_strength=data.get("music", {}).get("groove_strength", 0.5),
                harmonic_tension=data.get("music", {}).get("harmonic_tension", 0.5),
                harmonic_motion=data.get("music", {}).get("harmonic_motion", 0.5),
                mode_preference=data.get("music", {}).get("mode_preference", 0),
                melodic_activity=data.get("music", {}).get("melodic_activity", 0.5),
                contour_variance=data.get("music", {}).get("contour_variance", 0.5),
                dynamic_range=data.get("music", {}).get("dynamic_range", 0.5),
                texture_density=data.get("music", {}).get("texture_density", 0.5),
            ),
            time=TimeScope(
                start_bar=data.get("time", {}).get("start_bar", -1),
                end_bar=data.get("time", {}).get("end_bar", -1),
                fade_in_beats=data.get("time", {}).get("fade_in_beats", 0.0),
                fade_out_beats=data.get("time", {}).get("fade_out_beats", 0.0),
            ),
            constraints=IntentConstraints(
                allowed_engines_mask=data.get("constraints", {}).get("allowed_engines_mask", 0xFFFFFFFF),
                forbidden_engines_mask=data.get("constraints", {}).get("forbidden_engines_mask", 0),
                max_cpu_cost=data.get("constraints", {}).get("max_cpu_cost", 1.0),
                max_event_rate=data.get("constraints", {}).get("max_event_rate", 1000.0),
            ),
            provenance=IntentProvenance(
                source=IntentSource(data.get("provenance", {}).get("source", 0)),
                user_override_weight=data.get("provenance", {}).get("user_override_weight", 0.5),
            ),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "IntentFrame":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def generate_intent_id(self) -> int:
        """Generate a stable hash for intent_id based on frame contents."""
        # Create a deterministic hash from the frame contents
        frame_dict = self.to_dict()
        # Remove intent_id from hash calculation
        frame_dict["meta"]["intent_id"] = 0
        frame_str = json.dumps(frame_dict, sort_keys=True)
        hash_obj = hashlib.sha256(frame_str.encode())
        # Take first 8 bytes as uint64
        return int.from_bytes(hash_obj.digest()[:8], byteorder='big')
