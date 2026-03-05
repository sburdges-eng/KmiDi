"""
Dynamics Engine - Global volume/intensity curves and emotional breathing.

Part of Kelly MIDI Companion Tier 3.

Philosophy: Dynamics are the breath of a song. A song that's always loud
is exhausting. A song that's always quiet is forgettable. The space between
loud and soft is where emotion lives. Grief needs room to breathe. Rage
needs moments of terrifying quiet before the explosion. Joy needs release.
The dynamics ARE the emotional respiration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
import random
import math


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class DynamicMarking(Enum):
    """Traditional dynamic markings."""
    NIENTE = 0          # Nothing (silence)
    PPPP = 10           # As quiet as possible
    PPP = 20            # Pianississimo
    PP = 35             # Pianissimo
    P = 50              # Piano
    MP = 65             # Mezzo-piano
    MF = 80             # Mezzo-forte
    F = 95              # Forte
    FF = 110            # Fortissimo
    FFF = 120           # Fortississimo
    FFFF = 127          # As loud as possible


class DynamicShape(Enum):
    """Shapes for dynamic curves."""
    FLAT = "flat"                       # Constant level
    CRESCENDO = "crescendo"             # Gradual increase
    DECRESCENDO = "decrescendo"         # Gradual decrease (diminuendo)
    SWELL = "swell"                     # Crescendo then decrescendo
    INVERSE_SWELL = "inverse_swell"     # Decrescendo then crescendo
    ACCENT = "accent"                   # Sharp attack, quick decay
    SFORZANDO = "sforzando"             # Sudden loud then soft
    SUBITO = "subito"                   # Sudden change
    TERRACED = "terraced"               # Step changes
    WAVE = "wave"                       # Multiple swells
    BREATH = "breath"                   # Natural breathing pattern
    HEARTBEAT = "heartbeat"             # Rhythmic pulses
    STUTTER = "stutter"                 # Irregular bursts
    FADE_IN = "fade_in"                 # From nothing
    FADE_OUT = "fade_out"               # To nothing


class ExpressionType(Enum):
    """Types of dynamic expression."""
    VELOCITY = "velocity"               # Note attack strength
    VOLUME = "volume"                   # Overall level (CC7)
    EXPRESSION = "expression"           # Musical expression (CC11)
    BREATH = "breath_controller"        # Breath control (CC2)
    MODULATION = "modulation"           # Vibrato/movement (CC1)
    AFTERTOUCH = "aftertouch"           # Pressure after attack


class SilenceType(Enum):
    """Types of musical silence."""
    REST = "rest"                       # Standard rest
    BREATH = "breath"                   # Quick breath
    PAUSE = "pause"                     # Fermata-like hold
    DRAMATIC = "dramatic"               # Tension-building silence
    DECAY = "decay"                     # Let ring and fade
    CUT = "cut"                         # Abrupt stop


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DynamicPoint:
    """A point on the dynamics curve."""
    position: float          # 0.0 - 1.0 through section/song
    velocity: int            # 0-127
    expression: int = 100    # CC11 value
    volume: int = 100        # CC7 value
    marking: Optional[DynamicMarking] = None


@dataclass
class DynamicCurve:
    """A complete dynamics curve for a section or song."""
    points: List[DynamicPoint]
    shape: DynamicShape
    start_marking: DynamicMarking
    end_marking: DynamicMarking
    peak_position: float = 0.5      # Where the peak occurs (0.0-1.0)
    peak_marking: Optional[DynamicMarking] = None

    # Expression parameters
    has_accents: bool = False
    accent_positions: List[float] = field(default_factory=list)
    accent_strength: float = 0.2    # How much louder accents are

    # Humanization
    velocity_variance: int = 5      # Random variance in velocity
    timing_variance: float = 0.02   # Random variance in timing


@dataclass
class BreathingPattern:
    """Defines natural breathing dynamics for a section."""
    inhale_bars: float              # Duration of "inhale" (building)
    hold_bars: float                # Duration at peak
    exhale_bars: float              # Duration of "exhale" (releasing)
    rest_bars: float                # Rest before next breath
    depth: float                    # 0.0-1.0, how deep the breath


@dataclass
class SilenceEvent:
    """A deliberate silence in the music."""
    position: float                 # When it occurs
    duration_beats: float           # How long
    silence_type: SilenceType
    instruments_affected: List[str] = field(default_factory=list)  # Empty = all
    has_reverb_tail: bool = True    # Let previous sound decay


@dataclass
class DynamicsProfile:
    """Complete dynamics specification for a section."""
    curve: DynamicCurve
    breathing: Optional[BreathingPattern]
    silences: List[SilenceEvent]

    # Per-instrument adjustments
    instrument_offsets: Dict[str, int] = field(default_factory=dict)

    # Expression
    use_expression_cc: bool = True
    use_volume_cc: bool = False     # Usually expression is preferred

    # Metadata
    emotion: str = ""
    section_type: str = ""


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_DYNAMICS_PROFILES: Dict[str, Dict] = {
    "grief": {
        "base_dynamic": DynamicMarking.P,
        "max_dynamic": DynamicMarking.MF,
        "min_dynamic": DynamicMarking.PPP,
        "preferred_shapes": [DynamicShape.BREATH, DynamicShape.SWELL, DynamicShape.WAVE],
        "breathing_depth": 0.6,
        "breath_cycle_bars": 8,
        "silence_frequency": 0.3,       # High - grief needs space
        "silence_types": [SilenceType.BREATH, SilenceType.PAUSE, SilenceType.DECAY],
        "accent_probability": 0.1,
        "velocity_variance": 8,
        "expression_range": (40, 90),
        "uses_subito": False,
        "fade_tendency": "out",
    },

    "sadness": {
        "base_dynamic": DynamicMarking.MP,
        "max_dynamic": DynamicMarking.F,
        "min_dynamic": DynamicMarking.PP,
        "preferred_shapes": [DynamicShape.SWELL, DynamicShape.DECRESCENDO, DynamicShape.WAVE],
        "breathing_depth": 0.5,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.2,
        "silence_types": [SilenceType.BREATH, SilenceType.DECAY],
        "accent_probability": 0.15,
        "velocity_variance": 6,
        "expression_range": (50, 100),
        "uses_subito": False,
        "fade_tendency": "out",
    },

    "melancholy": {
        "base_dynamic": DynamicMarking.P,
        "max_dynamic": DynamicMarking.MF,
        "min_dynamic": DynamicMarking.PPP,
        "preferred_shapes": [DynamicShape.WAVE, DynamicShape.BREATH, DynamicShape.FLAT],
        "breathing_depth": 0.4,
        "breath_cycle_bars": 8,
        "silence_frequency": 0.25,
        "silence_types": [SilenceType.DECAY, SilenceType.PAUSE],
        "accent_probability": 0.05,
        "velocity_variance": 5,
        "expression_range": (40, 85),
        "uses_subito": False,
        "fade_tendency": "out",
    },

    "rage": {
        "base_dynamic": DynamicMarking.FF,
        "max_dynamic": DynamicMarking.FFFF,
        "min_dynamic": DynamicMarking.MF,
        "preferred_shapes": [DynamicShape.ACCENT, DynamicShape.SFORZANDO, DynamicShape.CRESCENDO],
        "breathing_depth": 0.3,
        "breath_cycle_bars": 2,
        "silence_frequency": 0.15,      # Strategic silence before explosion
        "silence_types": [SilenceType.DRAMATIC, SilenceType.CUT],
        "accent_probability": 0.5,
        "velocity_variance": 10,
        "expression_range": (80, 127),
        "uses_subito": True,
        "fade_tendency": "none",
    },

    "anger": {
        "base_dynamic": DynamicMarking.F,
        "max_dynamic": DynamicMarking.FFF,
        "min_dynamic": DynamicMarking.MF,
        "preferred_shapes": [DynamicShape.ACCENT, DynamicShape.TERRACED, DynamicShape.CRESCENDO],
        "breathing_depth": 0.4,
        "breath_cycle_bars": 2,
        "silence_frequency": 0.1,
        "silence_types": [SilenceType.CUT, SilenceType.DRAMATIC],
        "accent_probability": 0.4,
        "velocity_variance": 12,
        "expression_range": (70, 120),
        "uses_subito": True,
        "fade_tendency": "none",
    },

    "fear": {
        "base_dynamic": DynamicMarking.PP,
        "max_dynamic": DynamicMarking.FF,
        "min_dynamic": DynamicMarking.PPPP,
        "preferred_shapes": [DynamicShape.CRESCENDO, DynamicShape.STUTTER, DynamicShape.SUBITO],
        "breathing_depth": 0.7,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.35,      # Fear uses silence effectively
        "silence_types": [SilenceType.DRAMATIC, SilenceType.PAUSE, SilenceType.BREATH],
        "accent_probability": 0.3,
        "velocity_variance": 15,
        "expression_range": (20, 110),
        "uses_subito": True,
        "fade_tendency": "in",          # Fear builds
    },

    "anxiety": {
        "base_dynamic": DynamicMarking.MP,
        "max_dynamic": DynamicMarking.F,
        "min_dynamic": DynamicMarking.P,
        "preferred_shapes": [DynamicShape.STUTTER, DynamicShape.WAVE, DynamicShape.HEARTBEAT],
        "breathing_depth": 0.5,
        "breath_cycle_bars": 2,         # Quick, shallow breaths
        "silence_frequency": 0.1,       # Anxiety doesn't rest
        "silence_types": [SilenceType.BREATH],
        "accent_probability": 0.25,
        "velocity_variance": 10,
        "expression_range": (60, 100),
        "uses_subito": True,
        "fade_tendency": "none",
    },

    "hope": {
        "base_dynamic": DynamicMarking.MF,
        "max_dynamic": DynamicMarking.FF,
        "min_dynamic": DynamicMarking.P,
        "preferred_shapes": [DynamicShape.CRESCENDO, DynamicShape.SWELL, DynamicShape.FADE_IN],
        "breathing_depth": 0.6,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.15,
        "silence_types": [SilenceType.BREATH, SilenceType.PAUSE],
        "accent_probability": 0.2,
        "velocity_variance": 6,
        "expression_range": (50, 110),
        "uses_subito": False,
        "fade_tendency": "in",          # Hope builds
    },

    "joy": {
        "base_dynamic": DynamicMarking.F,
        "max_dynamic": DynamicMarking.FFF,
        "min_dynamic": DynamicMarking.MF,
        "preferred_shapes": [DynamicShape.SWELL, DynamicShape.WAVE, DynamicShape.ACCENT],
        "breathing_depth": 0.5,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.1,
        "silence_types": [SilenceType.BREATH],
        "accent_probability": 0.3,
        "velocity_variance": 8,
        "expression_range": (70, 120),
        "uses_subito": False,
        "fade_tendency": "none",
    },

    "peace": {
        "base_dynamic": DynamicMarking.PP,
        "max_dynamic": DynamicMarking.MP,
        "min_dynamic": DynamicMarking.PPP,
        "preferred_shapes": [DynamicShape.FLAT, DynamicShape.BREATH, DynamicShape.WAVE],
        "breathing_depth": 0.3,
        "breath_cycle_bars": 8,         # Slow, deep breaths
        "silence_frequency": 0.2,
        "silence_types": [SilenceType.DECAY, SilenceType.REST],
        "accent_probability": 0.05,
        "velocity_variance": 4,
        "expression_range": (30, 70),
        "uses_subito": False,
        "fade_tendency": "out",
    },

    "nostalgia": {
        "base_dynamic": DynamicMarking.MP,
        "max_dynamic": DynamicMarking.MF,
        "min_dynamic": DynamicMarking.PP,
        "preferred_shapes": [DynamicShape.SWELL, DynamicShape.WAVE, DynamicShape.BREATH],
        "breathing_depth": 0.5,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.2,
        "silence_types": [SilenceType.PAUSE, SilenceType.DECAY],
        "accent_probability": 0.15,
        "velocity_variance": 6,
        "expression_range": (45, 90),
        "uses_subito": False,
        "fade_tendency": "out",
    },

    "longing": {
        "base_dynamic": DynamicMarking.MP,
        "max_dynamic": DynamicMarking.F,
        "min_dynamic": DynamicMarking.P,
        "preferred_shapes": [DynamicShape.CRESCENDO, DynamicShape.SWELL, DynamicShape.INVERSE_SWELL],  # noqa: E501

        "breathing_depth": 0.6,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.2,
        "silence_types": [SilenceType.PAUSE, SilenceType.BREATH],
        "accent_probability": 0.2,
        "velocity_variance": 7,
        "expression_range": (50, 100),
        "uses_subito": False,
        "fade_tendency": "in",
    },

    "tension": {
        "base_dynamic": DynamicMarking.MF,
        "max_dynamic": DynamicMarking.FF,
        "min_dynamic": DynamicMarking.PP,
        "preferred_shapes": [DynamicShape.CRESCENDO, DynamicShape.TERRACED, DynamicShape.HEARTBEAT],
        "breathing_depth": 0.7,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.25,
        "silence_types": [SilenceType.DRAMATIC, SilenceType.CUT],
        "accent_probability": 0.35,
        "velocity_variance": 12,
        "expression_range": (40, 115),
        "uses_subito": True,
        "fade_tendency": "in",
    },

    "defiance": {
        "base_dynamic": DynamicMarking.F,
        "max_dynamic": DynamicMarking.FFFF,
        "min_dynamic": DynamicMarking.MF,
        "preferred_shapes": [DynamicShape.ACCENT, DynamicShape.SFORZANDO, DynamicShape.CRESCENDO],
        "breathing_depth": 0.4,
        "breath_cycle_bars": 2,
        "silence_frequency": 0.15,
        "silence_types": [SilenceType.DRAMATIC, SilenceType.CUT],
        "accent_probability": 0.45,
        "velocity_variance": 10,
        "expression_range": (75, 127),
        "uses_subito": True,
        "fade_tendency": "none",
    },

    "vulnerability": {
        "base_dynamic": DynamicMarking.PP,
        "max_dynamic": DynamicMarking.MP,
        "min_dynamic": DynamicMarking.PPPP,
        "preferred_shapes": [DynamicShape.BREATH, DynamicShape.SWELL, DynamicShape.FADE_IN],
        "breathing_depth": 0.6,
        "breath_cycle_bars": 8,
        "silence_frequency": 0.3,
        "silence_types": [SilenceType.BREATH, SilenceType.PAUSE, SilenceType.DECAY],
        "accent_probability": 0.05,
        "velocity_variance": 5,
        "expression_range": (25, 70),
        "uses_subito": False,
        "fade_tendency": "out",
    },

    "euphoria": {
        "base_dynamic": DynamicMarking.FF,
        "max_dynamic": DynamicMarking.FFFF,
        "min_dynamic": DynamicMarking.F,
        "preferred_shapes": [DynamicShape.SWELL, DynamicShape.WAVE, DynamicShape.CRESCENDO],
        "breathing_depth": 0.5,
        "breath_cycle_bars": 4,
        "silence_frequency": 0.1,
        "silence_types": [SilenceType.BREATH],
        "accent_probability": 0.35,
        "velocity_variance": 10,
        "expression_range": (85, 127),
        "uses_subito": False,
        "fade_tendency": "none",
    },

    "emptiness": {
        "base_dynamic": DynamicMarking.PPP,
        "max_dynamic": DynamicMarking.P,
        "min_dynamic": DynamicMarking.NIENTE,
        "preferred_shapes": [DynamicShape.FLAT, DynamicShape.FADE_OUT, DynamicShape.DECRESCENDO],
        "breathing_depth": 0.2,
        "breath_cycle_bars": 16,
        "silence_frequency": 0.4,       # Emptiness is mostly silence
        "silence_types": [SilenceType.DECAY, SilenceType.REST, SilenceType.PAUSE],
        "accent_probability": 0.02,
        "velocity_variance": 3,
        "expression_range": (10, 50),
        "uses_subito": False,
        "fade_tendency": "out",
    },
}


# =============================================================================
# INSTRUMENT DYNAMIC OFFSETS
# =============================================================================

INSTRUMENT_DEFAULT_OFFSETS: Dict[str, int] = {
    # Rhythm section - foundation
    "drums": 5,
    "bass": 0,

    # Harmonic
    "piano": 0,
    "keys": 0,
    "guitar": -3,
    "chords": -5,

    # Melodic
    "melody": 10,
    "lead": 10,
    "vocal": 15,

    # Textural
    "pad": -10,
    "strings": -5,
    "synth": -5,

    # Decorative
    "arpeggio": -8,
    "counter_melody": 5,
    "fill": 8,
}


# =============================================================================
# CURVE GENERATORS
# =============================================================================

def generate_dynamic_curve(
    shape: DynamicShape,
    num_points: int,
    start_velocity: int,
    end_velocity: int,
    peak_velocity: Optional[int] = None,
    peak_position: float = 0.5,
) -> List[int]:
    """Generate velocity values for a dynamic shape."""

    if peak_velocity is None:
        peak_velocity = max(start_velocity, end_velocity)

    points = []

    if shape == DynamicShape.FLAT:
        avg = (start_velocity + end_velocity) // 2
        points = [avg] * num_points

    elif shape == DynamicShape.CRESCENDO:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Exponential curve for more natural crescendo
            t_exp = t ** 1.5
            points.append(int(start_velocity + t_exp * (end_velocity - start_velocity)))

    elif shape == DynamicShape.DECRESCENDO:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            t_exp = t ** 0.7  # Inverse curve
            points.append(int(start_velocity - t_exp * (start_velocity - end_velocity)))

    elif shape == DynamicShape.SWELL:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < peak_position:
                # Rising
                local_t = t / peak_position
                points.append(int(start_velocity + local_t * (peak_velocity - start_velocity)))
            else:
                # Falling
                local_t = (t - peak_position) / (1 - peak_position)
                points.append(int(peak_velocity - local_t * (peak_velocity - end_velocity)))

    elif shape == DynamicShape.INVERSE_SWELL:
        valley = min(start_velocity, end_velocity, peak_velocity) - 10
        valley = max(1, valley)
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < peak_position:
                local_t = t / peak_position
                points.append(int(start_velocity - local_t * (start_velocity - valley)))
            else:
                local_t = (t - peak_position) / (1 - peak_position)
                points.append(int(valley + local_t * (end_velocity - valley)))

    elif shape == DynamicShape.ACCENT:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.1:
                # Sharp attack
                points.append(peak_velocity)
            else:
                # Quick decay
                decay_t = (t - 0.1) / 0.9
                points.append(int(peak_velocity - decay_t * (peak_velocity - end_velocity)))

    elif shape == DynamicShape.SFORZANDO:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.05:
                points.append(peak_velocity)
            elif t < 0.15:
                # Quick drop
                drop_t = (t - 0.05) / 0.1
                points.append(int(peak_velocity - drop_t * (peak_velocity - start_velocity) * 0.7))
            else:
                # Sustain at lower level
                points.append(int(start_velocity * 0.8))

    elif shape == DynamicShape.SUBITO:
        mid_point = num_points // 2
        for i in range(num_points):
            if i < mid_point:
                points.append(start_velocity)
            else:
                points.append(end_velocity)

    elif shape == DynamicShape.TERRACED:
        num_steps = 4
        step_size = num_points // num_steps
        vel_step = (end_velocity - start_velocity) / (num_steps - 1)
        for i in range(num_points):
            step = min(i // step_size, num_steps - 1)
            points.append(int(start_velocity + step * vel_step))

    elif shape == DynamicShape.WAVE:
        num_waves = 2
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            wave = math.sin(t * math.pi * num_waves * 2) * 0.5 + 0.5
            mid = (start_velocity + end_velocity) / 2
            amplitude = (peak_velocity - min(start_velocity, end_velocity)) / 2
            points.append(int(mid + wave * amplitude - amplitude / 2))

    elif shape == DynamicShape.BREATH:
        # Natural breathing: inhale (build), hold, exhale (release), rest
        phases = [0.25, 0.1, 0.35, 0.3]  # inhale, hold, exhale, rest
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < phases[0]:
                # Inhale - gradual rise
                local_t = t / phases[0]
                points.append(int(start_velocity + local_t * (peak_velocity - start_velocity)))
            elif t < phases[0] + phases[1]:
                # Hold
                points.append(peak_velocity)
            elif t < phases[0] + phases[1] + phases[2]:
                # Exhale - gradual fall
                local_t = (t - phases[0] - phases[1]) / phases[2]
                points.append(int(peak_velocity - local_t * (peak_velocity - end_velocity)))
            else:
                # Rest
                points.append(end_velocity)

    elif shape == DynamicShape.HEARTBEAT:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Two beats per cycle
            cycle_pos = (t * 4) % 1.0
            if cycle_pos < 0.15:
                # First beat
                points.append(peak_velocity)
            elif cycle_pos < 0.3:
                # Gap
                points.append(start_velocity)
            elif cycle_pos < 0.45:
                # Second beat (slightly softer)
                points.append(int(peak_velocity * 0.85))
            else:
                # Rest
                points.append(start_velocity)

    elif shape == DynamicShape.STUTTER:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Random-ish pattern
            phase = int(t * 12) % 4
            if phase == 0:
                points.append(peak_velocity)
            elif phase == 1:
                points.append(start_velocity)
            elif phase == 2:
                points.append(int((start_velocity + peak_velocity) / 2))
            else:
                points.append(start_velocity)

    elif shape == DynamicShape.FADE_IN:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Logarithmic fade in
            if t > 0:
                log_t = math.log10(t * 9 + 1)  # 0 to 1
                points.append(int(1 + log_t * (end_velocity - 1)))
            else:
                points.append(1)

    elif shape == DynamicShape.FADE_OUT:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Logarithmic fade out
            rev_t = 1 - t
            if rev_t > 0:
                log_t = math.log10(rev_t * 9 + 1)
                points.append(int(1 + log_t * (start_velocity - 1)))
            else:
                points.append(1)

    else:
        points = [start_velocity] * num_points

    # Clamp all values
    return [max(1, min(127, p)) for p in points]


# =============================================================================
# DYNAMICS ENGINE
# =============================================================================

class DynamicsEngine:
    """
    Generates dynamic curves and breathing patterns for emotional expression.

    Coordinates volume, velocity, and expression across sections and instruments.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_section_dynamics(
        self,
        emotion: str,
        section_type: str,
        bars: int,
        beats_per_bar: int = 4,
        position_in_song: float = 0.5,
        is_climax: bool = False,
    ) -> DynamicsProfile:
        """
        Generate dynamics profile for a section.

        Args:
            emotion: Primary emotion
            section_type: Type of section (verse, chorus, etc.)
            bars: Number of bars in section
            beats_per_bar: Time signature numerator
            position_in_song: 0.0-1.0 position in song
            is_climax: Whether this is the emotional peak

        Returns:
            DynamicsProfile with curves and silences
        """
        emotion_lower = emotion.lower()
        profile = EMOTION_DYNAMICS_PROFILES.get(
            emotion_lower,
            EMOTION_DYNAMICS_PROFILES["sadness"]
        )

        # Determine dynamic range based on section type
        section_modifiers = {
            "intro": (-10, 0.7),      # Quieter, less range
            "verse": (0, 0.8),
            "pre_chorus": (5, 0.9),
            "chorus": (10, 1.0),      # Loudest
            "post_chorus": (5, 0.9),
            "bridge": (-5, 0.8),
            "breakdown": (-15, 0.6),
            "build": (0, 1.0),        # Full range for builds
            "drop": (15, 1.0),
            "outro": (-10, 0.7),
            "interlude": (-10, 0.7),
            "solo": (5, 0.9),
            "silence": (-30, 0.3),
        }

        offset, range_mult = section_modifiers.get(section_type.lower(), (0, 0.8))

        # Calculate velocities
        base_vel = profile["base_dynamic"].value + offset
        if is_climax:
            max_vel = profile["max_dynamic"].value
        else:
            max_vel = int(
                profile["base_dynamic"].value +
                (profile["max_dynamic"].value - profile["base_dynamic"].value) * range_mult)
        min_vel = int(profile["min_dynamic"].value + offset * 0.5)

        # Clamp
        base_vel = max(1, min(127, base_vel))
        max_vel = max(1, min(127, max_vel))
        min_vel = max(1, min(127, min_vel))

        # Select shape
        if section_type.lower() == "build":
            shape = DynamicShape.CRESCENDO
        elif section_type.lower() == "breakdown":
            shape = DynamicShape.DECRESCENDO
        elif section_type.lower() == "drop":
            shape = DynamicShape.SFORZANDO
        elif section_type.lower() in ["outro"] and profile["fade_tendency"] == "out":
            shape = DynamicShape.FADE_OUT
        elif section_type.lower() in ["intro"] and profile["fade_tendency"] == "in":
            shape = DynamicShape.FADE_IN
        else:
            shape = self.rng.choice(profile["preferred_shapes"])

        # Generate curve
        num_points = bars * beats_per_bar

        # Determine start/end based on shape
        if shape in [DynamicShape.CRESCENDO, DynamicShape.FADE_IN]:
            start_vel, end_vel = min_vel, max_vel
        elif shape in [DynamicShape.DECRESCENDO, DynamicShape.FADE_OUT]:
            start_vel, end_vel = max_vel, min_vel
        else:
            start_vel, end_vel = base_vel, base_vel

        velocities = generate_dynamic_curve(
            shape=shape,
            num_points=num_points,
            start_velocity=start_vel,
            end_velocity=end_vel,
            peak_velocity=max_vel,
        )

        # Build curve object
        curve_points = []
        for i, vel in enumerate(velocities):
            pos = i / (len(velocities) - 1) if len(velocities) > 1 else 0

            # Add variance
            vel_with_variance = vel + self.rng.randint(
                -profile["velocity_variance"],
                profile["velocity_variance"]
            )
            vel_with_variance = max(1, min(127, vel_with_variance))

            # Calculate expression
            expr_min, expr_max = profile["expression_range"]
            expr = int(expr_min + (vel / 127) * (expr_max - expr_min))

            curve_points.append(DynamicPoint(
                position=pos,
                velocity=vel_with_variance,
                expression=expr,
                volume=100,
            ))

        # Generate accents
        accent_positions = []
        if profile["accent_probability"] > 0:
            for i in range(bars):
                if self.rng.random() < profile["accent_probability"]:
                    # Accent on downbeat
                    accent_positions.append(i / bars)
                if self.rng.random() < profile["accent_probability"] * 0.5:
                    # Occasional accent on beat 3
                    accent_positions.append((i + 0.5) / bars)

        curve = DynamicCurve(
            points=curve_points,
            shape=shape,
            start_marking=self._velocity_to_marking(start_vel),
            end_marking=self._velocity_to_marking(end_vel),
            peak_position=0.5,
            peak_marking=self._velocity_to_marking(max_vel),
            has_accents=len(accent_positions) > 0,
            accent_positions=accent_positions,
            accent_strength=0.2,
            velocity_variance=profile["velocity_variance"],
        )

        # Generate breathing pattern
        breathing = None
        if shape == DynamicShape.BREATH or profile["breathing_depth"] > 0.4:
            cycle_bars = profile["breath_cycle_bars"]
            breathing = BreathingPattern(
                inhale_bars=cycle_bars * 0.3,
                hold_bars=cycle_bars * 0.1,
                exhale_bars=cycle_bars * 0.4,
                rest_bars=cycle_bars * 0.2,
                depth=profile["breathing_depth"],
            )

        # Generate silences
        silences = []
        if profile["silence_frequency"] > 0:
            num_potential_silences = int(bars * profile["silence_frequency"])
            for _ in range(num_potential_silences):
                if self.rng.random() < profile["silence_frequency"]:
                    silence_type = self.rng.choice(profile["silence_types"])

                    # Duration based on type
                    if silence_type == SilenceType.BREATH:
                        duration = 0.5
                    elif silence_type == SilenceType.PAUSE:
                        duration = 1.0
                    elif silence_type == SilenceType.DRAMATIC:
                        duration = 2.0
                    elif silence_type == SilenceType.DECAY:
                        duration = 1.5
                    else:
                        duration = 0.25

                    silences.append(SilenceEvent(
                        position=self.rng.random(),
                        duration_beats=duration,
                        silence_type=silence_type,
                        has_reverb_tail=silence_type in [SilenceType.DECAY, SilenceType.PAUSE],
                    ))

        return DynamicsProfile(
            curve=curve,
            breathing=breathing,
            silences=silences,
            instrument_offsets=INSTRUMENT_DEFAULT_OFFSETS.copy(),
            use_expression_cc=True,
            emotion=emotion_lower,
            section_type=section_type,
        )

    def get_velocity_at_position(
        self,
        profile: DynamicsProfile,
        position: float,
        instrument: str = "",
    ) -> int:
        """Get velocity at a specific position in the section."""

        if not profile.curve.points:
            return 80

        # Find surrounding points
        prev_point = profile.curve.points[0]
        next_point = profile.curve.points[-1]

        for i, point in enumerate(profile.curve.points):
            if point.position >= position:
                next_point = point
                prev_point = profile.curve.points[max(0, i - 1)]
                break

        # Interpolate
        if prev_point.position == next_point.position:
            base_vel = prev_point.velocity
        else:
            t = (position - prev_point.position) / (next_point.position - prev_point.position)
            base_vel = int(prev_point.velocity + t * (next_point.velocity - prev_point.velocity))

        # Apply instrument offset
        if instrument:
            offset = profile.instrument_offsets.get(instrument.lower(), 0)
            base_vel += offset

        # Check for accent
        if profile.curve.has_accents:
            for accent_pos in profile.curve.accent_positions:
                if abs(position - accent_pos) < 0.02:
                    base_vel = int(base_vel * (1 + profile.curve.accent_strength))
                    break

        return max(1, min(127, base_vel))

    def get_expression_at_position(
        self,
        profile: DynamicsProfile,
        position: float,
    ) -> int:
        """Get expression CC value at position."""

        if not profile.curve.points:
            return 100

        # Find surrounding points
        prev_point = profile.curve.points[0]
        next_point = profile.curve.points[-1]

        for i, point in enumerate(profile.curve.points):
            if point.position >= position:
                next_point = point
                prev_point = profile.curve.points[max(0, i - 1)]
                break

        # Interpolate
        if prev_point.position == next_point.position:
            return prev_point.expression

        t = (position - prev_point.position) / (next_point.position - prev_point.position)
        return int(prev_point.expression + t * (next_point.expression - prev_point.expression))

    def is_silence_at_position(
        self,
        profile: DynamicsProfile,
        position: float,
        beats_per_bar: int = 4,
    ) -> Optional[SilenceEvent]:
        """Check if there's a silence event at this position."""

        for silence in profile.silences:
            # Calculate silence window
            silence_duration_normalized = silence.duration_beats / (beats_per_bar * 4)  # Rough
            if silence.position <= position < silence.position + silence_duration_normalized:
                return silence

        return None

    def _velocity_to_marking(self, velocity: int) -> DynamicMarking:
        """Convert velocity to traditional marking."""

        if velocity <= 10:
            return DynamicMarking.NIENTE
        elif velocity <= 20:
            return DynamicMarking.PPPP
        elif velocity <= 30:
            return DynamicMarking.PPP
        elif velocity <= 45:
            return DynamicMarking.PP
        elif velocity <= 60:
            return DynamicMarking.P
        elif velocity <= 75:
            return DynamicMarking.MP
        elif velocity <= 90:
            return DynamicMarking.MF
        elif velocity <= 105:
            return DynamicMarking.F
        elif velocity <= 115:
            return DynamicMarking.FF
        elif velocity <= 123:
            return DynamicMarking.FFF
        else:
            return DynamicMarking.FFFF

    def print_dynamics(self, profile: DynamicsProfile) -> str:
        """Generate visual representation of dynamics."""

        lines = []
        lines.append(
            f"═══ DYNAMICS: {profile.emotion.upper()} / {profile.section_type.upper()} ═══")
        lines.append(f"Shape: {profile.curve.shape.value}")
        lines.append(
            f"Range: {profile.curve.start_marking.name} → {profile.curve.end_marking.name}")
        lines.append("")

        # Visual curve
        width = 50
        lines.append("Velocity Curve:")

        # Sample 20 points
        for i in range(20):
            pos = i / 19
            vel = self.get_velocity_at_position(profile, pos)
            bar_len = int((vel / 127) * width)
            bar = "█" * bar_len + "░" * (width - bar_len)

            # Mark accents
            is_accent = any(abs(pos - ap) < 0.03 for ap in profile.curve.accent_positions)
            accent_mark = ">" if is_accent else " "

            lines.append(f"  {pos:.1f} [{bar}] {vel:3d} {accent_mark}")

        if profile.silences:
            lines.append("")
            lines.append(f"Silences: {len(profile.silences)}")
            for s in profile.silences[:3]:
                lines.append(
                    f"  @{s.position:.2f}: {s.silence_type.value} ({s.duration_beats} beats)")

        if profile.breathing:
            lines.append("")
            lines.append(f"Breathing: depth={profile.breathing.depth:.1f}")
            lines.append(
                f"  Cycle: inhale {profile.breathing.inhale_bars:.1f}b → hold {profile.breathing.hold_bars:.1f}b → exhale {profile.breathing.exhale_bars:.1f}b → rest {profile.breathing.rest_bars:.1f}b")  # noqa: E501

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_dynamics(
    emotion: str,
    section_type: str = "verse",
    bars: int = 8,
    seed: Optional[int] = None,
) -> DynamicsProfile:
    """Quick dynamics generation."""
    engine = DynamicsEngine(seed=seed)
    return engine.generate_section_dynamics(
        emotion=emotion,
        section_type=section_type,
        bars=bars,
    )


def get_available_shapes() -> List[str]:
    """List available dynamic shapes."""
    return [s.value for s in DynamicShape]


def get_dynamic_markings() -> Dict[str, int]:
    """Get mapping of dynamic markings to velocity."""
    return {m.name: m.value for m in DynamicMarking}


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║      KELLY DYNAMICS ENGINE - TIER 3      ║")
    print("╚══════════════════════════════════════════╝\n")

    engine = DynamicsEngine(seed=42)

    # Test different emotions and sections
    test_cases = [
        ("grief", "verse"),
        ("rage", "chorus"),
        ("hope", "build"),
        ("fear", "breakdown"),
        ("peace", "outro"),
    ]

    for emotion, section in test_cases:
        print(f"\n{'='*60}")

        profile = engine.generate_section_dynamics(
            emotion=emotion,
            section_type=section,
            bars=8,
        )

        print(engine.print_dynamics(profile))
