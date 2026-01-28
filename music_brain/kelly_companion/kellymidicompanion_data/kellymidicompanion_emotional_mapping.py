"""
Emotional State Mapping System.

Maps emotional states to musical parameters for Logic Pro integration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class TimingFeel(Enum):
    """Rhythmic timing feel."""
    STRAIGHT = "straight"
    SWUNG = "swung"
    LAID_BACK = "laid_back"
    PUSHED = "pushed"
    RUBATO = "rubato"


@dataclass
class EmotionalState:
    """
    Represents an emotional state with valence/arousal dimensions.

    Attributes:
        valence: -1.0 (negative) to 1.0 (positive)
        arousal: 0.0 (calm) to 1.0 (excited)
        primary_emotion: The main emotion identified
        secondary_emotions: Additional emotions present
    """
    valence: float = 0.0
    arousal: float = 0.5
    primary_emotion: str = "neutral"
    secondary_emotions: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Clamp values
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))


@dataclass
class MusicalParameters:
    """
    Musical parameters derived from emotional state.
    """
    tempo_suggested: int = 100
    key_suggested: str = "C"
    mode_suggested: str = "major"
    dissonance: float = 0.0  # 0-1, amount of harmonic tension
    density: float = 0.5  # 0-1, arrangement density
    timing_feel: TimingFeel = TimingFeel.STRAIGHT
    dynamics_range: float = 0.5  # 0-1, dynamic variation
    reverb_amount: float = 0.3  # 0-1
    brightness: float = 0.5  # 0-1, EQ brightness


# Valence/Arousal to musical parameter mappings
VALENCE_AROUSAL_MAPPINGS = {
    # High valence, high arousal (Joy, Excitement)
    (1, 1): {
        "tempo_range": (120, 160),
        "modes": ["Ionian", "Lydian", "Mixolydian"],
        "dissonance": 0.1,
        "brightness": 0.8,
        "reverb": 0.3,
    },
    # High valence, low arousal (Peace, Contentment)
    (1, 0): {
        "tempo_range": (60, 90),
        "modes": ["Ionian", "Lydian"],
        "dissonance": 0.0,
        "brightness": 0.5,
        "reverb": 0.5,
    },
    # Low valence, high arousal (Anger, Fear)
    (-1, 1): {
        "tempo_range": (100, 180),
        "modes": ["Phrygian", "Locrian", "Aeolian"],
        "dissonance": 0.7,
        "brightness": 0.6,
        "reverb": 0.2,
    },
    # Low valence, low arousal (Sadness, Grief)
    (-1, 0): {
        "tempo_range": (50, 80),
        "modes": ["Aeolian", "Dorian", "Phrygian"],
        "dissonance": 0.3,
        "brightness": 0.3,
        "reverb": 0.6,
    },
}


def get_parameters_for_state(state: EmotionalState) -> MusicalParameters:
    """
    Convert an emotional state to musical parameters.

    Args:
        state: EmotionalState with valence and arousal

    Returns:
        MusicalParameters object
    """
    # Quantize valence and arousal to quadrant
    v_key = 1 if state.valence >= 0 else -1
    a_key = 1 if state.arousal >= 0.5 else 0

    mapping = VALENCE_AROUSAL_MAPPINGS.get((v_key, a_key), VALENCE_AROUSAL_MAPPINGS[(1, 0)])

    # Interpolate tempo based on arousal
    tempo_min, tempo_max = mapping["tempo_range"]
    tempo = int(tempo_min + (tempo_max - tempo_min) * state.arousal)

    # Select mode (first is primary)
    mode = mapping["modes"][0]

    # Key selection based on emotion
    key_map = {
        "grief": "F",
        "sad": "D",
        "anger": "E",
        "fear": "B",
        "joy": "G",
        "love": "A",
        "neutral": "C",
    }
    key = key_map.get(state.primary_emotion.lower().split()[0], "C")

    # Timing feel based on arousal
    if state.arousal > 0.7:
        timing = TimingFeel.PUSHED
    elif state.arousal < 0.3:
        timing = TimingFeel.LAID_BACK
    elif state.valence < -0.5:
        timing = TimingFeel.RUBATO
    else:
        timing = TimingFeel.STRAIGHT

    return MusicalParameters(
        tempo_suggested=tempo,
        key_suggested=key,
        mode_suggested=mode.lower(),
        dissonance=mapping["dissonance"],
        density=0.3 + state.arousal * 0.5,
        timing_feel=timing,
        dynamics_range=0.3 + abs(state.valence) * 0.4,
        reverb_amount=mapping["reverb"],
        brightness=mapping["brightness"],
    )


def emotion_to_valence_arousal(emotion: str) -> tuple:
    """
    Map a primary emotion category to valence/arousal coordinates.

    Returns:
        (valence, arousal) tuple
    """
    emotion_map = {
        # Positive emotions
        "joy": (0.8, 0.7),
        "happiness": (0.7, 0.5),
        "euphoria": (0.9, 0.9),
        "love": (0.8, 0.6),
        "serenity": (0.6, 0.2),
        "contentment": (0.5, 0.3),
        "hope": (0.6, 0.5),

        # Negative emotions
        "sad": (-0.7, 0.3),
        "grief": (-0.8, 0.4),
        "melancholy": (-0.5, 0.2),
        "despair": (-0.9, 0.3),
        "loneliness": (-0.6, 0.2),

        # High arousal negative
        "anger": (-0.6, 0.9),
        "rage": (-0.8, 1.0),
        "frustration": (-0.5, 0.7),
        "hostility": (-0.7, 0.8),

        # Fear family
        "fear": (-0.7, 0.8),
        "terror": (-0.9, 0.9),
        "anxiety": (-0.5, 0.7),
        "dread": (-0.6, 0.6),

        # Surprise (neutral valence)
        "surprise": (0.2, 0.8),
        "shock": (-0.1, 0.9),
        "amazement": (0.4, 0.8),

        # Disgust
        "disgust": (-0.5, 0.5),
        "revulsion": (-0.7, 0.6),
        "contempt": (-0.4, 0.4),
    }

    return emotion_map.get(emotion.lower(), (0.0, 0.5))
