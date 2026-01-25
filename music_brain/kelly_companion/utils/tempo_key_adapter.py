"""
Tempo and Key Adaptive MIDI Generation.
Adjusts emotion-to-MIDI mapping based on locked BPM and key parameters.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TempoClass(Enum):
    """Tempo classification affecting emotional expression."""
    GLACIAL = "glacial"  # <50 BPM - Doom, despair, funeral
    SLOW = "slow"  # 50-80 BPM - Grief, introspection
    MODERATE = "moderate"  # 80-120 BPM - Neutral, conversational
    FAST = "fast"  # 120-160 BPM - Energy, anxiety, joy
    FRANTIC = "frantic"  # >160 BPM - Panic, euphoria, rage


class KeyBrightness(Enum):
    """Key signature brightness affecting emotional color."""
    VERY_DARK = "very_dark"  # 5+ flats/naturals in minor
    DARK = "dark"  # 3-4 flats in minor
    NEUTRAL = "neutral"  # C major, A minor
    BRIGHT = "bright"  # 3-4 sharps in major
    VERY_BRIGHT = "very_bright"  # 5+ sharps in major


@dataclass
class AdaptiveParameters:
    """Parameters that change based on tempo/key context."""
    note_density: float  # Notes per bar
    articulation_length: float  # Note duration as % of beat
    velocity_variance: float  # Dynamic range
    harmonic_complexity: int  # Chord extensions (7, 9, 11, 13)
    syncopation_amount: float  # Off-beat emphasis
    register_shift: int  # Octave adjustment
    chord_voicing_spread: int  # Semitones between voices


KEY_BRIGHTNESS_MAP = {
    # Major keys
    "C": KeyBrightness.NEUTRAL, "G": KeyBrightness.BRIGHT, "D": KeyBrightness.BRIGHT,
    "A": KeyBrightness.VERY_BRIGHT, "E": KeyBrightness.VERY_BRIGHT,
    "B": KeyBrightness.VERY_BRIGHT, "F#": KeyBrightness.VERY_BRIGHT,
    "Db": KeyBrightness.DARK, "Ab": KeyBrightness.DARK,
    "Eb": KeyBrightness.DARK, "Bb": KeyBrightness.NEUTRAL, "F": KeyBrightness.NEUTRAL,
    
    # Minor keys
    "Am": KeyBrightness.NEUTRAL, "Em": KeyBrightness.DARK, "Bm": KeyBrightness.DARK,
    "F#m": KeyBrightness.DARK, "C#m": KeyBrightness.DARK,
    "G#m": KeyBrightness.VERY_DARK, "D#m": KeyBrightness.VERY_DARK,
    "Bbm": KeyBrightness.VERY_DARK, "Fm": KeyBrightness.VERY_DARK,
    "Cm": KeyBrightness.DARK, "Gm": KeyBrightness.DARK, "Dm": KeyBrightness.DARK,
}


def classify_tempo(bpm: int) -> TempoClass:
    """Classify tempo into emotional range."""
    if bpm < 50:
        return TempoClass.GLACIAL
    elif bpm < 80:
        return TempoClass.SLOW
    elif bpm < 120:
        return TempoClass.MODERATE
    elif bpm < 160:
        return TempoClass.FAST
    else:
        return TempoClass.FRANTIC


def get_key_brightness(key: str, mode: str = "major") -> KeyBrightness:
    """Get brightness classification for key."""
    lookup_key = key if mode == "major" else f"{key}m"
    return KEY_BRIGHTNESS_MAP.get(lookup_key, KeyBrightness.NEUTRAL)


def adapt_emotion_to_tempo(
    emotion: str,
    bpm: int,
    original_bpm: int
) -> Dict[str, float]:
    """
    Adjust emotional expression based on tempo change.
    
    Args:
        emotion: Target emotion
        bpm: New locked tempo
        original_bpm: Suggested tempo from emotion
    
    Returns:
        Adjustment multipliers for musical parameters
    """
    tempo_class = classify_tempo(bpm)
    original_class = classify_tempo(original_bpm)
    
    # Base adjustments by tempo class
    adjustments = {
        TempoClass.GLACIAL: {
            "note_density": 0.3,  # Very sparse
            "articulation": 1.2,  # Longer notes
            "velocity_variance": 0.6,  # Less dynamic range (heavy)
            "syncopation": 0.2,  # Almost none
            "harmonic_complexity": 1.3,  # More extensions
        },
        TempoClass.SLOW: {
            "note_density": 0.5,
            "articulation": 1.0,
            "velocity_variance": 1.0,
            "syncopation": 0.4,
            "harmonic_complexity": 1.2,
        },
        TempoClass.MODERATE: {
            "note_density": 1.0,
            "articulation": 1.0,
            "velocity_variance": 1.0,
            "syncopation": 1.0,
            "harmonic_complexity": 1.0,
        },
        TempoClass.FAST: {
            "note_density": 1.4,
            "articulation": 0.7,  # Shorter notes
            "velocity_variance": 1.2,  # More dynamic
            "syncopation": 1.3,
            "harmonic_complexity": 0.8,  # Simpler chords
        },
        TempoClass.FRANTIC: {
            "note_density": 1.8,
            "articulation": 0.5,  # Staccato
            "velocity_variance": 1.5,
            "syncopation": 1.6,
            "harmonic_complexity": 0.6,  # Very simple
        }
    }
    
    # Emotion-specific tempo sensitivities
    emotion_lower = emotion.lower()
    
    # Grief/sadness at fast tempos = anxiety not sadness
    if "grief" in emotion_lower or "sad" in emotion_lower:
        if tempo_class in [TempoClass.FAST, TempoClass.FRANTIC]:
            adjustments[tempo_class]["note_density"] *= 1.2
            adjustments[tempo_class]["syncopation"] *= 1.4
            adjustments[tempo_class]["velocity_variance"] *= 1.3
    
    # Joy at slow tempos = contentment/peace not excitement
    elif "joy" in emotion_lower or "happy" in emotion_lower:
        if tempo_class in [TempoClass.GLACIAL, TempoClass.SLOW]:
            adjustments[tempo_class]["note_density"] *= 0.6
            adjustments[tempo_class]["syncopation"] *= 0.3
            adjustments[tempo_class]["harmonic_complexity"] *= 0.8
    
    # Anger at slow tempos = seething rage not explosion
    elif "anger" in emotion_lower or "rage" in emotion_lower:
        if tempo_class in [TempoClass.GLACIAL, TempoClass.SLOW]:
            adjustments[tempo_class]["velocity_variance"] *= 1.5
            adjustments[tempo_class]["harmonic_complexity"] *= 1.4
            adjustments[tempo_class]["articulation"] *= 1.3
    
    return adjustments[tempo_class]


def adapt_emotion_to_key(
    emotion: str,
    key: str,
    mode: str,
    original_key: str,
    original_mode: str
) -> Dict[str, float]:
    """
    Adjust emotional expression based on key change.
    
    Args:
        emotion: Target emotion
        key: New locked key
        mode: New mode (major/minor)
        original_key: Suggested key from emotion
        original_mode: Suggested mode
    
    Returns:
        Adjustment multipliers for musical parameters
    """
    brightness = get_key_brightness(key, mode)
    original_brightness = get_key_brightness(original_key, original_mode)
    
    # Base adjustments by key brightness
    adjustments = {
        KeyBrightness.VERY_DARK: {
            "register_shift": -12,  # Lower octave
            "chord_voicing_spread": 18,  # Wide, open voicings
            "harmonic_complexity": 1.4,
            "velocity_variance": 0.8,
        },
        KeyBrightness.DARK: {
            "register_shift": -7,
            "chord_voicing_spread": 14,
            "harmonic_complexity": 1.2,
            "velocity_variance": 0.9,
        },
        KeyBrightness.NEUTRAL: {
            "register_shift": 0,
            "chord_voicing_spread": 12,
            "harmonic_complexity": 1.0,
            "velocity_variance": 1.0,
        },
        KeyBrightness.BRIGHT: {
            "register_shift": 5,
            "chord_voicing_spread": 10,
            "harmonic_complexity": 0.9,
            "velocity_variance": 1.1,
        },
        KeyBrightness.VERY_BRIGHT: {
            "register_shift": 12,  # Higher octave
            "chord_voicing_spread": 8,  # Tighter voicings
            "harmonic_complexity": 0.8,
            "velocity_variance": 1.3,
        }
    }
    
    # Emotion-specific key sensitivities
    emotion_lower = emotion.lower()
    
    # Grief in bright keys = bittersweet not purely sad
    if "grief" in emotion_lower or "sad" in emotion_lower:
        if brightness in [KeyBrightness.BRIGHT, KeyBrightness.VERY_BRIGHT]:
            adjustments[brightness]["harmonic_complexity"] *= 1.3
            adjustments[brightness]["velocity_variance"] *= 1.2
    
    # Joy in dark keys = defiant joy not pure happiness
    elif "joy" in emotion_lower or "happy" in emotion_lower:
        if brightness in [KeyBrightness.VERY_DARK, KeyBrightness.DARK]:
            adjustments[brightness]["velocity_variance"] *= 1.4
            adjustments[brightness]["harmonic_complexity"] *= 1.2
    
    return adjustments[brightness]


def generate_adaptive_parameters(
    emotion: str,
    locked_bpm: Optional[int],
    locked_key: Optional[str],
    locked_mode: Optional[str],
    suggested_bpm: int,
    suggested_key: str,
    suggested_mode: str
) -> AdaptiveParameters:
    """
    Generate adapted parameters when user locks tempo/key.
    
    Args:
        emotion: Target emotion
        locked_bpm: User-locked tempo (None if not locked)
        locked_key: User-locked key (None if not locked)
        locked_mode: User-locked mode (None if not locked)
        suggested_bpm: System-suggested tempo from emotion
        suggested_key: System-suggested key from emotion
        suggested_mode: System-suggested mode from emotion
    
    Returns:
        AdaptiveParameters with adjusted values
    """
    # Start with baseline for emotion
    base_params = AdaptiveParameters(
        note_density=1.0,
        articulation_length=1.0,
        velocity_variance=1.0,
        harmonic_complexity=7,  # 7th chords default
        syncopation_amount=1.0,
        register_shift=0,
        chord_voicing_spread=12
    )
    
    # Apply tempo adaptations if locked
    if locked_bpm and locked_bpm != suggested_bpm:
        tempo_adj = adapt_emotion_to_tempo(emotion, locked_bpm, suggested_bpm)
        base_params.note_density *= tempo_adj["note_density"]
        base_params.articulation_length *= tempo_adj["articulation"]
        base_params.velocity_variance *= tempo_adj["velocity_variance"]
        base_params.syncopation_amount *= tempo_adj["syncopation"]
        base_params.harmonic_complexity = int(
            base_params.harmonic_complexity * tempo_adj["harmonic_complexity"]
        )
    
    # Apply key adaptations if locked
    if locked_key and (locked_key != suggested_key or locked_mode != suggested_mode):
        key_adj = adapt_emotion_to_key(
            emotion,
            locked_key,
            locked_mode or suggested_mode,
            suggested_key,
            suggested_mode
        )
        base_params.register_shift += key_adj["register_shift"]
        base_params.chord_voicing_spread = int(
            base_params.chord_voicing_spread * (key_adj["chord_voicing_spread"] / 12)
        )
        base_params.harmonic_complexity = int(
            base_params.harmonic_complexity * key_adj["harmonic_complexity"]
        )
        base_params.velocity_variance *= key_adj["velocity_variance"]
    
    return base_params


def should_regenerate_midi(
    current_bpm: int,
    current_key: str,
    current_mode: str,
    new_bpm: Optional[int],
    new_key: Optional[str],
    new_mode: Optional[str]
) -> bool:
    """
    Determine if MIDI needs regeneration based on parameter changes.
    
    Returns:
        True if changes cross emotional threshold requiring regeneration
    """
    if new_bpm and abs(new_bpm - current_bpm) > 20:
        # >20 BPM change = different tempo class likely
        return True
    
    if new_key and new_key != current_key:
        # Key change always requires regeneration
        return True
    
    if new_mode and new_mode != current_mode:
        # Mode change (major/minor) always requires regeneration
        return True
    
    return False


# Example usage integration
def regenerate_with_locked_params(
    emotion: str,
    locked_bpm: Optional[int] = None,
    locked_key: Optional[str] = None,
    locked_mode: Optional[str] = None
) -> Dict:
    """
    Example: Regenerate MIDI with locked user parameters.
    
    Returns:
        Dict with adapted parameters for MIDI generation
    """
    # Get emotion's natural suggestions
    from kellymidicompanion_emotional_mapping import get_parameters_for_state, EmotionalState
    from kellymidicompanion_emotion_api import emotion_to_valence_arousal
    
    valence, arousal = emotion_to_valence_arousal(emotion)
    state = EmotionalState(
        valence=valence,
        arousal=arousal,
        primary_emotion=emotion
    )
    suggested_params = get_parameters_for_state(state)
    
    # Generate adaptive parameters
    adapted = generate_adaptive_parameters(
        emotion=emotion,
        locked_bpm=locked_bpm,
        locked_key=locked_key,
        locked_mode=locked_mode,
        suggested_bpm=suggested_params.tempo_suggested,
        suggested_key=suggested_params.key_suggested,
        suggested_mode=suggested_params.mode_suggested
    )
    
    return {
        "emotion": emotion,
        "bpm": locked_bpm or suggested_params.tempo_suggested,
        "key": locked_key or suggested_params.key_suggested,
        "mode": locked_mode or suggested_params.mode_suggested,
        "adaptive_params": adapted,
        "original_suggestions": {
            "bpm": suggested_params.tempo_suggested,
            "key": suggested_params.key_suggested,
            "mode": suggested_params.mode_suggested
        }
    }
