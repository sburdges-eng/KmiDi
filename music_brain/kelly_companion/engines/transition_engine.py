"""
Transition Engine - Section-to-section bridges and mood transitions.

Part of Kelly MIDI Companion Tier 3.

Philosophy: The moment between sections is where magic happens - or where
songs fall apart. A clumsy transition breaks the spell. A masterful one
feels inevitable. The drop that takes your breath. The bridge that lifts
you somewhere new. The fade that lets go gently. Transitions are the
connective tissue of emotional narrative.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple
import random
import math

# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================


class TransitionType(Enum):
    """Types of transitions between sections."""

    # Energy-based
    HARD_CUT = "hard_cut"  # Immediate switch
    SOFT_CUT = "soft_cut"  # Brief gap then new section
    CROSSFADE = "crossfade"  # Overlap outgoing/incoming
    FADE_OUT = "fade_out"  # Gradual decrease
    FADE_IN = "fade_in"  # Gradual increase

    # Rhythmic
    FILL = "fill"  # Drum/melodic fill
    BREAK = "break"  # Stop then restart
    STOP_TIME = "stop_time"  # Rhythmic hits
    ACCELERANDO = "accelerando"  # Speed up
    RITARDANDO = "ritardando"  # Slow down

    # Harmonic
    PIVOT = "pivot"  # Pivot chord
    TURNAROUND = "turnaround"  # Harmonic turnaround
    MODULATION = "modulation"  # Key change
    CADENCE = "cadence"  # Strong cadential motion
    DECEPTIVE = "deceptive"  # Deceptive resolution

    # Textural
    BUILD = "build"  # Add layers progressively
    STRIP = "strip"  # Remove layers progressively
    SWELL = "swell"  # Crescendo then release
    COLLAPSE = "collapse"  # Sudden thinning

    # Structural
    RISER = "riser"  # Synth/noise riser
    REVERSE = "reverse"  # Reversed sound effect
    IMPACT = "impact"  # Crash/hit
    SWEEP = "sweep"  # Filter sweep
    SILENCE = "silence"  # Dramatic pause

    # Organic
    BREATH = "breath"  # Natural breathing space
    ANTICIPATION = "anticipation"  # Early elements of next section
    OVERLAP = "overlap"  # Elements from both sections


class TransitionLength(Enum):
    """Duration of transition."""

    INSTANT = 0  # No transition time
    BRIEF = 1  # 1 beat
    SHORT = 2  # 2 beats / half bar
    MEDIUM = 4  # 1 bar
    LONG = 8  # 2 bars
    EXTENDED = 16  # 4 bars


class EnergyDirection(Enum):
    """Direction of energy change."""

    UP = "up"  # Building energy
    DOWN = "down"  # Releasing energy
    SAME = "same"  # Maintaining level
    SPIKE_UP = "spike_up"  # Sudden increase
    SPIKE_DOWN = "spike_down"  # Sudden decrease


class MoodShift(Enum):
    """Type of mood change."""

    SAME = "same"  # No mood change
    GRADUAL = "gradual"  # Smooth transition
    CONTRAST = "contrast"  # Clear change
    SURPRISE = "surprise"  # Unexpected shift


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TransitionPoint:
    """A point within a transition."""

    position: float  # 0.0 - 1.0 through transition
    energy_level: float  # 0.0 - 1.0
    density: float  # 0.0 - 1.0 instrument density
    tension: float  # 0.0 - 1.0 harmonic tension

    # Active elements
    drums_active: bool = True
    bass_active: bool = True
    chords_active: bool = True
    melody_active: bool = True
    effects_active: bool = False  # Risers, impacts, etc.


@dataclass
class TransitionSpec:
    """Complete specification for a transition."""

    transition_type: TransitionType
    length: TransitionLength
    energy_direction: EnergyDirection
    mood_shift: MoodShift

    # Curve
    points: List[TransitionPoint] = field(default_factory=list)

    # Source and destination
    from_section: str = ""
    to_section: str = ""
    from_energy: float = 0.5
    to_energy: float = 0.5

    # Musical elements
    has_fill: bool = False
    fill_type: str = ""  # "drum", "melodic", "bass", etc.
    has_riser: bool = False
    has_impact: bool = False
    has_reverse: bool = False

    # Harmonic
    pivot_chord: Optional[str] = None
    uses_turnaround: bool = False

    # Timing
    beat_offset: float = 0.0  # Where transition starts relative to bar
    tempo_change: float = 1.0  # Tempo ratio

    # Metadata
    emotion: str = ""


@dataclass
class TransitionPlan:
    """Plan for all transitions in a song."""

    transitions: List[TransitionSpec]
    section_names: List[str]

    # Global settings
    default_type: TransitionType = TransitionType.FILL
    use_production_elements: bool = True  # Risers, impacts

    # Metadata
    emotion: str = ""
    total_sections: int = 0


# =============================================================================
# SECTION TRANSITION MATRICES
# =============================================================================

# What transitions work well between section types
SECTION_TRANSITION_PREFERENCES: Dict[Tuple[str, str], List[TransitionType]] = {
    # Into verse
    ("intro", "verse"): [TransitionType.FILL, TransitionType.BUILD, TransitionType.SOFT_CUT],
    ("chorus", "verse"): [TransitionType.STRIP, TransitionType.FILL, TransitionType.BREATH],
    ("bridge", "verse"): [TransitionType.FILL, TransitionType.SOFT_CUT, TransitionType.BREATH],
    # Into chorus
    ("verse", "chorus"): [TransitionType.BUILD, TransitionType.FILL, TransitionType.SWELL],
    ("pre_chorus", "chorus"): [TransitionType.RISER, TransitionType.BUILD, TransitionType.IMPACT],
    ("bridge", "chorus"): [TransitionType.BUILD, TransitionType.SWELL, TransitionType.IMPACT],
    # Into bridge
    ("chorus", "bridge"): [TransitionType.STRIP, TransitionType.COLLAPSE, TransitionType.BREATH],
    ("verse", "bridge"): [TransitionType.FILL, TransitionType.PIVOT, TransitionType.BREATH],
    # Into breakdown
    ("chorus", "breakdown"): [
        TransitionType.COLLAPSE,
        TransitionType.SILENCE,
        TransitionType.STRIP,
    ],  # noqa: E501
    ("verse", "breakdown"): [TransitionType.STRIP, TransitionType.SILENCE, TransitionType.BREATH],
    ("drop", "breakdown"): [TransitionType.COLLAPSE, TransitionType.SILENCE, TransitionType.STRIP],
    # Into build
    ("breakdown", "build"): [
        TransitionType.BUILD,
        TransitionType.RISER,
        TransitionType.ANTICIPATION,
    ],  # noqa: E501
    ("verse", "build"): [TransitionType.BUILD, TransitionType.FILL, TransitionType.SWELL],
    # Into drop
    ("build", "drop"): [TransitionType.IMPACT, TransitionType.HARD_CUT, TransitionType.SILENCE],
    ("breakdown", "drop"): [TransitionType.BUILD, TransitionType.RISER, TransitionType.IMPACT],
    # Into outro
    ("chorus", "outro"): [TransitionType.FADE_OUT, TransitionType.STRIP, TransitionType.BREATH],
    ("verse", "outro"): [TransitionType.FADE_OUT, TransitionType.SOFT_CUT, TransitionType.BREATH],
    ("bridge", "outro"): [TransitionType.FADE_OUT, TransitionType.STRIP, TransitionType.CADENCE],
}

# Energy change implications
ENERGY_TRANSITIONS: Dict[EnergyDirection, List[TransitionType]] = {
    EnergyDirection.UP: [
        TransitionType.BUILD,
        TransitionType.FILL,
        TransitionType.RISER,
        TransitionType.SWELL,
        TransitionType.ACCELERANDO,
    ],
    EnergyDirection.DOWN: [
        TransitionType.STRIP,
        TransitionType.FADE_OUT,
        TransitionType.COLLAPSE,
        TransitionType.BREATH,
        TransitionType.RITARDANDO,
    ],
    EnergyDirection.SAME: [
        TransitionType.FILL,
        TransitionType.SOFT_CUT,
        TransitionType.TURNAROUND,
        TransitionType.OVERLAP,
    ],
    EnergyDirection.SPIKE_UP: [
        TransitionType.IMPACT,
        TransitionType.HARD_CUT,
        TransitionType.STOP_TIME,
    ],
    EnergyDirection.SPIKE_DOWN: [
        TransitionType.SILENCE,
        TransitionType.COLLAPSE,
        TransitionType.BREAK,
    ],
}


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_TRANSITION_PROFILES: Dict[str, Dict] = {
    "grief": {
        "preferred_types": [
            TransitionType.BREATH,
            TransitionType.FADE_OUT,
            TransitionType.SOFT_CUT,
            TransitionType.SILENCE,
        ],
        "avoid_types": [
            TransitionType.IMPACT,
            TransitionType.HARD_CUT,
            TransitionType.RISER,
        ],
        "preferred_length": TransitionLength.LONG,
        "use_production_elements": False,
        "energy_tendency": EnergyDirection.DOWN,
        "fill_probability": 0.2,
        "impact_probability": 0.0,
        "riser_probability": 0.0,
    },
    "sadness": {
        "preferred_types": [
            TransitionType.BREATH,
            TransitionType.SOFT_CUT,
            TransitionType.FADE_OUT,
            TransitionType.STRIP,
        ],
        "avoid_types": [
            TransitionType.IMPACT,
            TransitionType.HARD_CUT,
        ],
        "preferred_length": TransitionLength.MEDIUM,
        "use_production_elements": False,
        "energy_tendency": EnergyDirection.DOWN,
        "fill_probability": 0.3,
        "impact_probability": 0.1,
        "riser_probability": 0.1,
    },
    "melancholy": {
        "preferred_types": [
            TransitionType.CROSSFADE,
            TransitionType.BREATH,
            TransitionType.SOFT_CUT,
        ],
        "avoid_types": [
            TransitionType.IMPACT,
            TransitionType.HARD_CUT,
            TransitionType.RISER,
        ],
        "preferred_length": TransitionLength.MEDIUM,
        "use_production_elements": False,
        "energy_tendency": EnergyDirection.SAME,
        "fill_probability": 0.25,
        "impact_probability": 0.05,
        "riser_probability": 0.05,
    },
    "rage": {
        "preferred_types": [
            TransitionType.HARD_CUT,
            TransitionType.IMPACT,
            TransitionType.BREAK,
            TransitionType.STOP_TIME,
        ],
        "avoid_types": [
            TransitionType.FADE_OUT,
            TransitionType.CROSSFADE,
            TransitionType.BREATH,
        ],
        "preferred_length": TransitionLength.BRIEF,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.SPIKE_UP,
        "fill_probability": 0.7,
        "impact_probability": 0.6,
        "riser_probability": 0.4,
    },
    "anger": {
        "preferred_types": [
            TransitionType.HARD_CUT,
            TransitionType.FILL,
            TransitionType.IMPACT,
            TransitionType.BUILD,
        ],
        "avoid_types": [
            TransitionType.FADE_OUT,
            TransitionType.BREATH,
        ],
        "preferred_length": TransitionLength.SHORT,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.6,
        "impact_probability": 0.5,
        "riser_probability": 0.3,
    },
    "fear": {
        "preferred_types": [
            TransitionType.BUILD,
            TransitionType.SILENCE,
            TransitionType.ANTICIPATION,
            TransitionType.RISER,
        ],
        "avoid_types": [
            TransitionType.HARD_CUT,
        ],
        "preferred_length": TransitionLength.LONG,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.3,
        "impact_probability": 0.4,
        "riser_probability": 0.6,
    },
    "anxiety": {
        "preferred_types": [
            TransitionType.BUILD,
            TransitionType.ACCELERANDO,
            TransitionType.ANTICIPATION,
            TransitionType.OVERLAP,
        ],
        "avoid_types": [
            TransitionType.SILENCE,
            TransitionType.BREATH,
        ],
        "preferred_length": TransitionLength.SHORT,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.5,
        "impact_probability": 0.3,
        "riser_probability": 0.5,
    },
    "hope": {
        "preferred_types": [
            TransitionType.BUILD,
            TransitionType.SWELL,
            TransitionType.FILL,
            TransitionType.ANTICIPATION,
        ],
        "avoid_types": [
            TransitionType.COLLAPSE,
            TransitionType.SILENCE,
        ],
        "preferred_length": TransitionLength.MEDIUM,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.5,
        "impact_probability": 0.4,
        "riser_probability": 0.5,
    },
    "joy": {
        "preferred_types": [
            TransitionType.FILL,
            TransitionType.BUILD,
            TransitionType.IMPACT,
            TransitionType.SWELL,
        ],
        "avoid_types": [
            TransitionType.COLLAPSE,
            TransitionType.SILENCE,
        ],
        "preferred_length": TransitionLength.SHORT,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.6,
        "impact_probability": 0.5,
        "riser_probability": 0.4,
    },
    "peace": {
        "preferred_types": [
            TransitionType.CROSSFADE,
            TransitionType.BREATH,
            TransitionType.FADE_OUT,
            TransitionType.SOFT_CUT,
        ],
        "avoid_types": [
            TransitionType.IMPACT,
            TransitionType.HARD_CUT,
            TransitionType.RISER,
        ],
        "preferred_length": TransitionLength.LONG,
        "use_production_elements": False,
        "energy_tendency": EnergyDirection.SAME,
        "fill_probability": 0.2,
        "impact_probability": 0.0,
        "riser_probability": 0.0,
    },
    "nostalgia": {
        "preferred_types": [
            TransitionType.CROSSFADE,
            TransitionType.BREATH,
            TransitionType.SOFT_CUT,
            TransitionType.TURNAROUND,
        ],
        "avoid_types": [
            TransitionType.IMPACT,
            TransitionType.HARD_CUT,
        ],
        "preferred_length": TransitionLength.MEDIUM,
        "use_production_elements": False,
        "energy_tendency": EnergyDirection.SAME,
        "fill_probability": 0.3,
        "impact_probability": 0.1,
        "riser_probability": 0.1,
    },
    "longing": {
        "preferred_types": [
            TransitionType.SWELL,
            TransitionType.BUILD,
            TransitionType.BREATH,
            TransitionType.ANTICIPATION,
        ],
        "avoid_types": [
            TransitionType.HARD_CUT,
            TransitionType.COLLAPSE,
        ],
        "preferred_length": TransitionLength.MEDIUM,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.4,
        "impact_probability": 0.2,
        "riser_probability": 0.4,
    },
    "tension": {
        "preferred_types": [
            TransitionType.BUILD,
            TransitionType.RISER,
            TransitionType.ANTICIPATION,
            TransitionType.SILENCE,
        ],
        "avoid_types": [
            TransitionType.BREATH,
            TransitionType.FADE_OUT,
        ],
        "preferred_length": TransitionLength.LONG,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.4,
        "impact_probability": 0.5,
        "riser_probability": 0.7,
    },
    "defiance": {
        "preferred_types": [
            TransitionType.IMPACT,
            TransitionType.HARD_CUT,
            TransitionType.BUILD,
            TransitionType.STOP_TIME,
        ],
        "avoid_types": [
            TransitionType.FADE_OUT,
            TransitionType.BREATH,
        ],
        "preferred_length": TransitionLength.SHORT,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.SPIKE_UP,
        "fill_probability": 0.6,
        "impact_probability": 0.6,
        "riser_probability": 0.4,
    },
    "vulnerability": {
        "preferred_types": [
            TransitionType.BREATH,
            TransitionType.SOFT_CUT,
            TransitionType.FADE_OUT,
            TransitionType.SILENCE,
        ],
        "avoid_types": [
            TransitionType.IMPACT,
            TransitionType.HARD_CUT,
            TransitionType.RISER,
        ],
        "preferred_length": TransitionLength.MEDIUM,
        "use_production_elements": False,
        "energy_tendency": EnergyDirection.DOWN,
        "fill_probability": 0.2,
        "impact_probability": 0.0,
        "riser_probability": 0.0,
    },
    "euphoria": {
        "preferred_types": [
            TransitionType.BUILD,
            TransitionType.RISER,
            TransitionType.IMPACT,
            TransitionType.SWELL,
        ],
        "avoid_types": [
            TransitionType.FADE_OUT,
            TransitionType.COLLAPSE,
        ],
        "preferred_length": TransitionLength.MEDIUM,
        "use_production_elements": True,
        "energy_tendency": EnergyDirection.UP,
        "fill_probability": 0.5,
        "impact_probability": 0.7,
        "riser_probability": 0.7,
    },
    "emptiness": {
        "preferred_types": [
            TransitionType.FADE_OUT,
            TransitionType.SILENCE,
            TransitionType.STRIP,
            TransitionType.BREATH,
        ],
        "avoid_types": [
            TransitionType.BUILD,
            TransitionType.IMPACT,
            TransitionType.RISER,
        ],
        "preferred_length": TransitionLength.EXTENDED,
        "use_production_elements": False,
        "energy_tendency": EnergyDirection.DOWN,
        "fill_probability": 0.1,
        "impact_probability": 0.0,
        "riser_probability": 0.0,
    },
}


# =============================================================================
# CURVE GENERATORS
# =============================================================================


def generate_transition_curve(
    transition_type: TransitionType,
    num_points: int,
    from_energy: float,
    to_energy: float,
) -> List[TransitionPoint]:
    """Generate a curve of transition points."""

    points = []

    for i in range(num_points):
        pos = i / (num_points - 1) if num_points > 1 else 0.5

        # Calculate energy based on transition type
        if transition_type == TransitionType.HARD_CUT:
            if pos < 0.5:
                energy = from_energy
            else:
                energy = to_energy

        elif transition_type == TransitionType.SOFT_CUT:
            if pos < 0.4:
                energy = from_energy
            elif pos < 0.6:
                energy = from_energy * 0.3  # Dip
            else:
                energy = to_energy

        elif transition_type in [TransitionType.CROSSFADE, TransitionType.OVERLAP]:
            # Linear interpolation
            energy = from_energy + pos * (to_energy - from_energy)

        elif transition_type == TransitionType.FADE_OUT:
            energy = from_energy * (1 - pos)

        elif transition_type == TransitionType.FADE_IN:
            energy = to_energy * pos

        elif transition_type in [TransitionType.BUILD, TransitionType.RISER]:
            # Exponential build
            energy = from_energy + (pos**2) * (to_energy - from_energy)

        elif transition_type in [TransitionType.STRIP, TransitionType.COLLAPSE]:
            # Inverse exponential
            energy = from_energy + ((1 - (1 - pos) ** 2)) * (to_energy - from_energy)

        elif transition_type == TransitionType.SWELL:
            # Up then down
            if pos < 0.7:
                energy = from_energy + (pos / 0.7) * (
                    max(from_energy, to_energy) * 1.2 - from_energy
                )
            else:
                peak = max(from_energy, to_energy) * 1.2
                energy = peak - ((pos - 0.7) / 0.3) * (peak - to_energy)

        elif transition_type == TransitionType.SILENCE:
            if 0.3 < pos < 0.7:
                energy = 0.0
            elif pos <= 0.3:
                energy = from_energy * (1 - pos / 0.3)
            else:
                energy = to_energy * ((pos - 0.7) / 0.3)

        elif transition_type in [TransitionType.IMPACT, TransitionType.STOP_TIME]:
            if pos < 0.1:
                energy = from_energy
            elif pos < 0.15:
                energy = max(from_energy, to_energy) * 1.3  # Spike
            else:
                energy = to_energy

        elif transition_type == TransitionType.BREAK:
            if pos < 0.4:
                energy = from_energy
            elif pos < 0.6:
                energy = 0.0
            else:
                energy = to_energy

        else:
            energy = from_energy + pos * (to_energy - from_energy)

        energy = max(0.0, min(1.0, energy))

        # Calculate density (generally follows energy)
        density = energy * 0.8 + 0.2

        # Calculate tension (peaks at midpoint for most transitions)
        tension_curve = math.sin(pos * math.pi)
        base_tension = 0.3
        if transition_type in [
            TransitionType.BUILD,
            TransitionType.RISER,
            TransitionType.ANTICIPATION,
        ]:
            base_tension = 0.5
        tension = base_tension + tension_curve * 0.4

        # Determine active elements
        drums_active = energy > 0.2
        bass_active = energy > 0.15
        chords_active = energy > 0.1
        melody_active = energy > 0.3
        effects_active = transition_type in [
            TransitionType.RISER,
            TransitionType.IMPACT,
            TransitionType.REVERSE,
            TransitionType.SWEEP,
        ]

        points.append(
            TransitionPoint(
                position=pos,
                energy_level=energy,
                density=density,
                tension=tension,
                drums_active=drums_active,
                bass_active=bass_active,
                chords_active=chords_active,
                melody_active=melody_active,
                effects_active=effects_active,
            )
        )

    return points


# =============================================================================
# TRANSITION ENGINE
# =============================================================================


class TransitionEngine:
    """
    Generates transitions between song sections.

    Ensures smooth flow between sections with appropriate energy changes
    and emotional coherence.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_transition(
        self,
        emotion: str,
        from_section: str,
        to_section: str,
        from_energy: float = 0.5,
        to_energy: float = 0.5,
        length_override: Optional[TransitionLength] = None,
        type_override: Optional[TransitionType] = None,
    ) -> TransitionSpec:
        """
        Generate a transition between two sections.

        Args:
            emotion: Primary emotion
            from_section: Source section type
            to_section: Destination section type
            from_energy: Energy level of source (0.0-1.0)
            to_energy: Energy level of destination (0.0-1.0)
            length_override: Force specific length
            type_override: Force specific type

        Returns:
            TransitionSpec with all details
        """
        emotion_lower = emotion.lower()
        profile = EMOTION_TRANSITION_PROFILES.get(
            emotion_lower, EMOTION_TRANSITION_PROFILES["sadness"]
        )

        # Determine energy direction
        energy_diff = to_energy - from_energy
        if abs(energy_diff) < 0.1:
            energy_direction = EnergyDirection.SAME
        elif energy_diff > 0.4:
            energy_direction = EnergyDirection.SPIKE_UP
        elif energy_diff < -0.4:
            energy_direction = EnergyDirection.SPIKE_DOWN
        elif energy_diff > 0:
            energy_direction = EnergyDirection.UP
        else:
            energy_direction = EnergyDirection.DOWN

        # Override with emotion tendency if same level
        if energy_direction == EnergyDirection.SAME:
            energy_direction = profile["energy_tendency"]

        # Select transition type
        if type_override:
            trans_type = type_override
        else:
            # Check section-specific preferences
            section_key = (from_section.lower(), to_section.lower())
            if section_key in SECTION_TRANSITION_PREFERENCES:
                candidates = SECTION_TRANSITION_PREFERENCES[section_key]
                # Filter by emotion preferences
                filtered = [t for t in candidates if t in profile["preferred_types"]]
                if filtered:
                    trans_type = self.rng.choice(filtered)
                else:
                    trans_type = self.rng.choice(candidates)
            else:
                # Use energy-based selection
                energy_candidates = ENERGY_TRANSITIONS.get(energy_direction, [])
                filtered = [t for t in energy_candidates if t in profile["preferred_types"]]
                if filtered:
                    trans_type = self.rng.choice(filtered)
                elif profile["preferred_types"]:
                    trans_type = self.rng.choice(profile["preferred_types"])
                else:
                    trans_type = TransitionType.FILL

        # Avoid disallowed types
        if trans_type in profile["avoid_types"]:
            alternatives = [
                t for t in profile["preferred_types"] if t not in profile["avoid_types"]
            ]
            if alternatives:
                trans_type = self.rng.choice(alternatives)

        # Select length
        if length_override:
            length = length_override
        else:
            length = profile["preferred_length"]
            # Adjust for energy difference
            if abs(energy_diff) > 0.5:
                # Big changes need more time (usually)
                if length.value < TransitionLength.MEDIUM.value:
                    length = TransitionLength.MEDIUM

        # Determine mood shift
        if abs(energy_diff) < 0.15:
            mood_shift = MoodShift.SAME
        elif trans_type in [TransitionType.HARD_CUT, TransitionType.IMPACT]:
            mood_shift = MoodShift.CONTRAST
        elif trans_type in [TransitionType.SILENCE, TransitionType.BREAK]:
            mood_shift = MoodShift.SURPRISE
        else:
            mood_shift = MoodShift.GRADUAL

        # Generate transition curve
        num_points = max(4, length.value * 2)
        points = generate_transition_curve(
            trans_type,
            num_points,
            from_energy,
            to_energy,
        )

        # Determine production elements
        has_fill = self.rng.random() < profile["fill_probability"]
        has_riser = (
            profile["use_production_elements"]
            and self.rng.random() < profile["riser_probability"]
            and trans_type
            in [TransitionType.BUILD, TransitionType.RISER, TransitionType.ANTICIPATION]
        )
        has_impact = (
            profile["use_production_elements"]
            and self.rng.random() < profile["impact_probability"]
            and trans_type in [TransitionType.IMPACT, TransitionType.HARD_CUT, TransitionType.BUILD]
        )
        has_reverse = (
            profile["use_production_elements"]
            and self.rng.random() < 0.2
            and trans_type == TransitionType.REVERSE
        )

        # Fill type
        fill_type = ""
        if has_fill:
            if to_section.lower() in ["chorus", "drop"]:
                fill_type = "drum"
            elif from_section.lower() in ["breakdown", "bridge"]:
                fill_type = "melodic"
            else:
                fill_type = self.rng.choice(["drum", "melodic", "bass"])

        return TransitionSpec(
            transition_type=trans_type,
            length=length,
            energy_direction=energy_direction,
            mood_shift=mood_shift,
            points=points,
            from_section=from_section,
            to_section=to_section,
            from_energy=from_energy,
            to_energy=to_energy,
            has_fill=has_fill,
            fill_type=fill_type,
            has_riser=has_riser,
            has_impact=has_impact,
            has_reverse=has_reverse,
            emotion=emotion_lower,
        )

    def generate_transition_plan(
        self,
        emotion: str,
        sections: List[Tuple[str, float]],  # (section_type, energy_level)
    ) -> TransitionPlan:
        """
        Generate transitions for an entire song.

        Args:
            emotion: Primary emotion
            sections: List of (section_type, energy) tuples

        Returns:
            TransitionPlan with all transitions
        """
        transitions = []
        section_names = [s[0] for s in sections]

        for i in range(len(sections) - 1):
            from_section, from_energy = sections[i]
            to_section, to_energy = sections[i + 1]

            trans = self.generate_transition(
                emotion=emotion,
                from_section=from_section,
                to_section=to_section,
                from_energy=from_energy,
                to_energy=to_energy,
            )
            transitions.append(trans)

        return TransitionPlan(
            transitions=transitions,
            section_names=section_names,
            emotion=emotion,
            total_sections=len(sections),
        )

    def get_state_at_position(
        self,
        spec: TransitionSpec,
        position: float,
    ) -> TransitionPoint:
        """Get interpolated transition state at any position."""

        if not spec.points:
            return TransitionPoint(
                position=position,
                energy_level=0.5,
                density=0.5,
                tension=0.5,
            )

        # Find surrounding points
        prev_point = spec.points[0]
        next_point = spec.points[-1]

        for i, point in enumerate(spec.points):
            if point.position >= position:
                next_point = point
                prev_point = spec.points[max(0, i - 1)]
                break

        # Interpolate
        if prev_point.position == next_point.position:
            return prev_point

        t = (position - prev_point.position) / (next_point.position - prev_point.position)

        return TransitionPoint(
            position=position,
            energy_level=prev_point.energy_level
            + t * (next_point.energy_level - prev_point.energy_level),
            density=prev_point.density + t * (next_point.density - prev_point.density),
            tension=prev_point.tension + t * (next_point.tension - prev_point.tension),
            drums_active=prev_point.drums_active if t < 0.5 else next_point.drums_active,
            bass_active=prev_point.bass_active if t < 0.5 else next_point.bass_active,
            chords_active=prev_point.chords_active if t < 0.5 else next_point.chords_active,
            melody_active=prev_point.melody_active if t < 0.5 else next_point.melody_active,
            effects_active=prev_point.effects_active or next_point.effects_active,
        )

    def print_transition(self, spec: TransitionSpec) -> str:
        """Generate visual representation of transition."""

        lines = []
        lines.append(f"═══ TRANSITION: {spec.from_section} → {spec.to_section} ═══")
        lines.append(f"Type: {spec.transition_type.value} | Length: {spec.length.name}")
        lines.append(
            f"Energy: {spec.from_energy:.2f} → {spec.to_energy:.2f} ({spec.energy_direction.value})"
        )  # noqa: E501

        lines.append(f"Mood: {spec.mood_shift.value}")
        lines.append("")

        # Elements
        elements = []
        if spec.has_fill:
            elements.append(f"Fill({spec.fill_type})")
        if spec.has_riser:
            elements.append("Riser")
        if spec.has_impact:
            elements.append("Impact")
        if spec.has_reverse:
            elements.append("Reverse")
        if elements:
            lines.append(f"Elements: {', '.join(elements)}")
        else:
            lines.append("Elements: None")
        lines.append("")

        # Curve visualization
        width = 40
        lines.append("Energy Curve:")

        for point in spec.points:
            bar_len = int(point.energy_level * width)
            bar = "█" * bar_len + "░" * (width - bar_len)

            active = ""
            if point.drums_active:
                active += "D"
            if point.bass_active:
                active += "B"
            if point.chords_active:
                active += "C"
            if point.melody_active:
                active += "M"
            if point.effects_active:
                active += "X"

            lines.append(f"  {point.position:.2f} [{bar}] {active}")

        return "\n".join(lines)

    def print_transition_plan(self, plan: TransitionPlan) -> str:
        """Generate visual representation of transition plan."""

        lines = []
        lines.append(f"═══ TRANSITION PLAN: {plan.emotion.upper()} ═══")
        lines.append(f"Sections: {plan.total_sections} | Transitions: {len(plan.transitions)}")
        lines.append("")

        # Section flow
        flow = " → ".join(plan.section_names)
        lines.append(f"Flow: {flow}")
        lines.append("")

        # Each transition
        for i, trans in enumerate(plan.transitions):
            type_icon = {
                TransitionType.HARD_CUT: "⚡",
                TransitionType.SOFT_CUT: "～",
                TransitionType.CROSSFADE: "≋",
                TransitionType.FADE_OUT: "↘",
                TransitionType.FADE_IN: "↗",
                TransitionType.BUILD: "▲",
                TransitionType.STRIP: "▼",
                TransitionType.FILL: "♪",
                TransitionType.IMPACT: "💥",
                TransitionType.RISER: "⤴",
                TransitionType.SILENCE: "○",
                TransitionType.BREATH: "◦",
                TransitionType.SWELL: "◆",
            }.get(trans.transition_type, "·")

            energy_arrow = (
                "→"
                if abs(trans.to_energy - trans.from_energy) < 0.1
                else "↑" if trans.to_energy > trans.from_energy else "↓"
            )  # noqa: E501

            elements = []
            if trans.has_fill:
                elements.append("F")
            if trans.has_riser:
                elements.append("R")
            if trans.has_impact:
                elements.append("I")
            elem_str = "".join(elements) if elements else "-"

            lines.append(
                f"  {i+1}. {trans.from_section:12s} {type_icon} {trans.to_section:12s} | "
                f"{trans.transition_type.value:12s} | "
                f"{trans.from_energy:.1f}{energy_arrow}{trans.to_energy:.1f} | "
                f"[{elem_str}]"
            )

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_transition(
    emotion: str,
    from_section: str,
    to_section: str,
    from_energy: float = 0.5,
    to_energy: float = 0.5,
    seed: Optional[int] = None,
) -> TransitionSpec:
    """Quick transition generation."""
    engine = TransitionEngine(seed=seed)
    return engine.generate_transition(
        emotion=emotion,
        from_section=from_section,
        to_section=to_section,
        from_energy=from_energy,
        to_energy=to_energy,
    )


def create_transition_plan(
    emotion: str,
    sections: List[Tuple[str, float]],
    seed: Optional[int] = None,
) -> TransitionPlan:
    """Quick transition plan generation."""
    engine = TransitionEngine(seed=seed)
    return engine.generate_transition_plan(emotion=emotion, sections=sections)


def get_transition_types() -> List[str]:
    """List available transition types."""
    return [t.value for t in TransitionType]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║    KELLY TRANSITION ENGINE - TIER 3      ║")
    print("╚══════════════════════════════════════════╝\n")

    engine = TransitionEngine(seed=42)

    # Test individual transitions
    test_cases = [
        ("grief", "verse", "chorus", 0.4, 0.5),
        ("rage", "build", "drop", 0.7, 1.0),
        ("hope", "verse", "chorus", 0.5, 0.8),
        ("fear", "breakdown", "build", 0.3, 0.7),
        ("peace", "verse", "outro", 0.4, 0.2),
    ]

    for emotion, from_s, to_s, from_e, to_e in test_cases:
        print(f"\n{'='*60}")

        trans = engine.generate_transition(
            emotion=emotion,
            from_section=from_s,
            to_section=to_s,
            from_energy=from_e,
            to_energy=to_e,
        )

        print(engine.print_transition(trans))

    # Test full plan
    print(f"\n{'='*60}")
    print("FULL TRANSITION PLAN")
    print("=" * 60)

    sections = [
        ("intro", 0.3),
        ("verse", 0.5),
        ("chorus", 0.8),
        ("verse", 0.5),
        ("chorus", 0.85),
        ("bridge", 0.6),
        ("chorus", 0.9),
        ("outro", 0.3),
    ]

    plan = engine.generate_transition_plan(
        emotion="hope",
        sections=sections,
    )

    print(engine.print_transition_plan(plan))
