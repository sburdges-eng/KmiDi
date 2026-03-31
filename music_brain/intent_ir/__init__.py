"""
Intent IR v1 - Python dataclasses matching IR spec

This module provides Python dataclasses that match the Rust Intent IR v1
specification. These are used for emitting IntentFrames from Python ML code.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import json
import warnings

# Intent IR version
INTENT_IR_VERSION = 1


class IntentSource(IntEnum):
    """Intent source - provenance tracking"""
    UI_DIRECT = 0
    UI_EDIT = 1
    ML_TEXT = 2
    ML_AUDIO = 3
    PRESET = 4
    AUTOMATION = 5


@dataclass
class IntentMeta:
    """Intent Meta - Routing, compatibility, debugging"""
    ir_version: int = INTENT_IR_VERSION
    intent_id: int = 0
    session_id: int = 0


@dataclass
class EmotionState:
    """Emotion State - VAD coordinates with optional discrete mapping.

    NOTE: discrete_id and intensity are deprecated. They are not part of
    the canonical emotion_schema.json v1 contract. Use EmotionStateSchema
    from music_brain.engine_api.schema for new code.
    """
    valence: float = 0.0  # [-1.0, 1.0]
    arousal: float = 0.5  # [0.0, 1.0]
    dominance: float = 0.5  # [0.0, 1.0]
    discrete_id: int = -1  # DEPRECATED - not in emotion_schema.json v1
    intensity: float = 0.0  # DEPRECATED - not in emotion_schema.json v1
    confidence: float = 0.0  # [0.0, 1.0]

    def __post_init__(self):
        if self.discrete_id != -1:
            warnings.warn(
                "EmotionState.discrete_id is deprecated and will be removed. "
                "Use EmotionStateSchema from music_brain.engine_api.schema.",
                DeprecationWarning, stacklevel=2,
            )
        if self.intensity != 0.0:
            warnings.warn(
                "EmotionState.intensity is deprecated and will be removed. "
                "Use EmotionStateSchema from music_brain.engine_api.schema.",
                DeprecationWarning, stacklevel=2,
            )


@dataclass
class MusicalIntent:
    """Musical Intent - Biases and tendencies (no notes, no MIDI)"""
    tempo_bias: float = 0.0  # [-1.0, 1.0]
    rhythmic_density: float = 0.5  # [0.0, 1.0]
    groove_strength: float = 0.5  # [0.0, 1.0]
    harmonic_tension: float = 0.5  # [0.0, 1.0]
    harmonic_motion: float = 0.5  # [0.0, 1.0]
    mode_preference: int = 0  # -1 (minor), 0 (neutral), +1 (major)
    melodic_activity: float = 0.5  # [0.0, 1.0]
    contour_variance: float = 0.5  # [0.0, 1.0]
    dynamic_range: float = 0.5  # [0.0, 1.0]
    texture_density: float = 0.5  # [0.0, 1.0]


@dataclass
class TimeScope:
    """Time Scope - Intent without time is noise"""
    start_bar: int = -1  # Inclusive, -1 = immediate
    end_bar: int = -1  # Exclusive, -1 = open-ended
    fade_in_beats: float = 0.0  # 0.0 = hard
    fade_out_beats: float = 0.0  # 0.0 = hard


@dataclass
class IntentConstraints:
    """Intent Constraints - Limit generation, not force it"""
    allowed_engines_mask: int = 0xFFFFFFFF  # Bitmask of allowed engines
    forbidden_engines_mask: int = 0  # Bitmask of forbidden engines
    max_cpu_cost: float = 1.0  # Hint, not guarantee
    max_event_rate: float = float('inf')  # Max event rate


@dataclass
class IntentProvenance:
    """Intent Provenance - Debugging and trust"""
    source: IntentSource = IntentSource.UI_DIRECT
    user_override_weight: float = 1.0  # [0.0, 1.0] - 0.0 = ML dominates, 1.0 = user dominates


@dataclass
class IntentFrame:
    """IntentFrame - Top-level unit representing one musical intention"""
    meta: IntentMeta = field(default_factory=IntentMeta)
    emotion: EmotionState = field(default_factory=EmotionState)
    music: MusicalIntent = field(default_factory=MusicalIntent)
    time: TimeScope = field(default_factory=TimeScope)
    constraints: IntentConstraints = field(default_factory=IntentConstraints)
    provenance: IntentProvenance = field(default_factory=IntentProvenance)

    def to_json(self) -> str:
        """Serialize IntentFrame to JSON"""
        # Convert enum to int for JSON serialization
        data = {
            'meta': {
                'ir_version': self.meta.ir_version,
                'intent_id': self.meta.intent_id,
                'session_id': self.meta.session_id,
            },
            'emotion': {
                'valence': self.emotion.valence,
                'arousal': self.emotion.arousal,
                'dominance': self.emotion.dominance,
                'discrete_id': self.emotion.discrete_id,
                'intensity': self.emotion.intensity,
                'confidence': self.emotion.confidence,
            },
            'music': {
                'tempo_bias': self.music.tempo_bias,
                'rhythmic_density': self.music.rhythmic_density,
                'groove_strength': self.music.groove_strength,
                'harmonic_tension': self.music.harmonic_tension,
                'harmonic_motion': self.music.harmonic_motion,
                'mode_preference': self.music.mode_preference,
                'melodic_activity': self.music.melodic_activity,
                'contour_variance': self.music.contour_variance,
                'dynamic_range': self.music.dynamic_range,
                'texture_density': self.music.texture_density,
            },
            'time': {
                'start_bar': self.time.start_bar,
                'end_bar': self.time.end_bar,
                'fade_in_beats': self.time.fade_in_beats,
                'fade_out_beats': self.time.fade_out_beats,
            },
            'constraints': {
                'allowed_engines_mask': self.constraints.allowed_engines_mask,
                'forbidden_engines_mask': self.constraints.forbidden_engines_mask,
                'max_cpu_cost': self.constraints.max_cpu_cost,
                'max_event_rate': self.constraints.max_event_rate if self.constraints.max_event_rate != float('inf') else -1.0,  # noqa: E501

            },
            'provenance': {
                'source': int(self.provenance.source),
                'user_override_weight': self.provenance.user_override_weight,
            },
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'IntentFrame':
        """Deserialize IntentFrame from JSON"""
        data = json.loads(json_str)

        # Handle max_event_rate special case
        max_event_rate = data['constraints']['max_event_rate']
        if max_event_rate == -1.0:
            max_event_rate = float('inf')

        return cls(
            meta=IntentMeta(**data['meta']),
            emotion=EmotionState(**data['emotion']),
            music=MusicalIntent(**data['music']),
            time=TimeScope(**data['time']),
            constraints=IntentConstraints(
                allowed_engines_mask=data['constraints']['allowed_engines_mask'],
                forbidden_engines_mask=data['constraints']['forbidden_engines_mask'],
                max_cpu_cost=data['constraints']['max_cpu_cost'],
                max_event_rate=max_event_rate,
            ),
            provenance=IntentProvenance(
                source=IntentSource(data['provenance']['source']),
                user_override_weight=data['provenance']['user_override_weight'],
            ),
        )
