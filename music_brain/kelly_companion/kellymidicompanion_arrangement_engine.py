"""
Arrangement Engine - Song structure and emotional arc coordination.

Part of Kelly MIDI Companion Tier 3.

Philosophy: A song is a journey, not a destination. The arrangement engine
maps the emotional terrain - where we start, where we climb, where we rest,
where we break. Every section serves the emotional narrative. A grief song
might never reach a triumphant chorus. A rage song might skip the gentle intro.
The structure IS the story.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Tuple
import random

# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================


class SectionType(Enum):
    """Song section types."""

    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    POST_CHORUS = "post_chorus"
    BRIDGE = "bridge"
    BREAKDOWN = "breakdown"
    BUILD = "build"
    DROP = "drop"
    OUTRO = "outro"
    INTERLUDE = "interlude"
    SOLO = "solo"
    REFRAIN = "refrain"
    CODA = "coda"
    SILENCE = "silence"


class ArcShape(Enum):
    """Emotional arc shapes across the song."""

    LINEAR_RISE = "linear_rise"  # Steady climb
    LINEAR_FALL = "linear_fall"  # Steady descent
    WAVE = "wave"  # Up and down cycles
    PLATEAU = "plateau"  # Rise then sustain
    VALLEY = "valley"  # Fall then rise
    PEAK = "peak"  # Rise, peak, fall
    DOUBLE_PEAK = "double_peak"  # Two climaxes
    FLAT = "flat"  # Consistent intensity
    SPIRAL = "spiral"  # Each cycle higher
    COLLAPSE = "collapse"  # Sudden drop
    EXPLOSION = "explosion"  # Sudden rise
    TENSION_RELEASE = "tension_release"  # Build then release cycles


class StructureTemplate(Enum):
    """Common song structure templates."""

    VERSE_CHORUS = "verse_chorus"  # V-C-V-C
    VERSE_CHORUS_BRIDGE = "verse_chorus_bridge"  # V-C-V-C-B-C
    AABA = "aaba"  # Classic 32-bar
    ABABCB = "ababcb"  # Pop standard
    THROUGH_COMPOSED = "through_composed"  # No repeats
    RONDO = "rondo"  # A-B-A-C-A
    BINARY = "binary"  # A-B
    TERNARY = "ternary"  # A-B-A
    EDM_BUILD_DROP = "edm_build_drop"  # Build-Drop cycles
    AMBIENT_DRIFT = "ambient_drift"  # Gradual evolution
    MINIMAL = "minimal"  # Single section exploration
    GRIEF_ARC = "grief_arc"  # Custom: denial-anger-bargaining-depression-acceptance  # noqa: E501

    RAGE_SPIRAL = "rage_spiral"  # Custom: escalating intensity


class EnergyLevel(Enum):
    """Section energy levels."""

    SILENT = 0
    MINIMAL = 1
    LOW = 2
    MEDIUM_LOW = 3
    MEDIUM = 4
    MEDIUM_HIGH = 5
    HIGH = 6
    INTENSE = 7
    EXPLOSIVE = 8


class InstrumentDensity(Enum):
    """How many instruments active in section."""

    SOLO = "solo"  # 1 instrument
    DUO = "duo"  # 2 instruments
    TRIO = "trio"  # 3 instruments
    SMALL = "small"  # 4-5 instruments
    MEDIUM = "medium"  # 6-8 instruments
    FULL = "full"  # All instruments
    MASSIVE = "massive"  # All + layers


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SectionProfile:
    """Defines characteristics of a song section."""

    section_type: SectionType
    bars: int
    energy: EnergyLevel
    density: InstrumentDensity
    emotion_intensity: float  # 0.0 - 1.0

    # Instrument activation flags
    drums_active: bool = True
    bass_active: bool = True
    chords_active: bool = True
    melody_active: bool = True
    pads_active: bool = False
    strings_active: bool = False
    arpeggio_active: bool = False
    counter_melody_active: bool = False

    # Section-specific modifiers
    tempo_modifier: float = 1.0  # Relative to base tempo
    velocity_modifier: float = 1.0  # Relative to base velocity
    humanization: float = 0.5  # 0.0 = rigid, 1.0 = loose

    # Transitions
    has_fill_in: bool = False  # Fill leading into this section
    has_fill_out: bool = False  # Fill leading out of this section
    fade_in_bars: int = 0
    fade_out_bars: int = 0

    # Variation
    variation_index: int = 0  # For repeated sections (Verse 1, Verse 2)
    variation_intensity: float = 0.0  # How different from base (0.0 = identical)


@dataclass
class EmotionArcPoint:
    """A point on the emotional arc."""

    position: float  # 0.0 - 1.0 through song
    primary_emotion: str
    intensity: float  # 0.0 - 1.0
    secondary_emotion: Optional[str] = None
    tension: float = 0.5  # 0.0 - 1.0


@dataclass
class ArrangementPlan:
    """Complete arrangement plan for a song."""

    sections: List[SectionProfile]
    emotional_arc: List[EmotionArcPoint]
    total_bars: int
    structure_template: StructureTemplate
    arc_shape: ArcShape

    # Global parameters
    base_tempo: int = 120
    key: str = "C"
    mode: str = "major"
    time_signature: Tuple[int, int] = (4, 4)

    # Metadata
    estimated_duration_seconds: float = 0.0
    section_count: int = 0
    unique_sections: int = 0


@dataclass
class SectionTransition:
    """Defines how to transition between sections."""

    from_section: SectionType
    to_section: SectionType
    transition_type: str  # "cut", "fill", "fade", "build", "breakdown"
    bars: int = 1
    energy_curve: str = "linear"  # "linear", "exponential", "sudden"


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_ARRANGEMENT_PROFILES: Dict[str, Dict] = {
    "grief": {
        "preferred_structures": [StructureTemplate.THROUGH_COMPOSED, StructureTemplate.GRIEF_ARC],
        "preferred_arcs": [ArcShape.VALLEY, ArcShape.PLATEAU, ArcShape.FLAT],
        "max_energy": EnergyLevel.MEDIUM,
        "typical_density": InstrumentDensity.SMALL,
        "section_preferences": {
            SectionType.INTRO: 0.9,
            SectionType.VERSE: 1.0,
            SectionType.CHORUS: 0.3,  # Grief often avoids triumphant chorus
            SectionType.BRIDGE: 0.7,
            SectionType.BREAKDOWN: 0.8,
            SectionType.OUTRO: 0.9,
            SectionType.SILENCE: 0.6,
        },
        "tempo_range": (60, 90),
        "builds_to_release": False,
        "allows_climax": False,
        "preferred_ending": "fade",
    },
    "sadness": {
        "preferred_structures": [StructureTemplate.VERSE_CHORUS, StructureTemplate.TERNARY],
        "preferred_arcs": [ArcShape.VALLEY, ArcShape.WAVE],
        "max_energy": EnergyLevel.MEDIUM_HIGH,
        "typical_density": InstrumentDensity.MEDIUM,
        "section_preferences": {
            SectionType.INTRO: 0.8,
            SectionType.VERSE: 1.0,
            SectionType.CHORUS: 0.7,
            SectionType.BRIDGE: 0.8,
            SectionType.OUTRO: 0.9,
        },
        "tempo_range": (70, 100),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "fade",
    },
    "melancholy": {
        "preferred_structures": [StructureTemplate.AABA, StructureTemplate.AMBIENT_DRIFT],
        "preferred_arcs": [ArcShape.WAVE, ArcShape.PLATEAU],
        "max_energy": EnergyLevel.MEDIUM,
        "typical_density": InstrumentDensity.SMALL,
        "section_preferences": {
            SectionType.INTRO: 0.9,
            SectionType.VERSE: 1.0,
            SectionType.INTERLUDE: 0.8,
            SectionType.OUTRO: 0.9,
        },
        "tempo_range": (65, 95),
        "builds_to_release": False,
        "allows_climax": False,
        "preferred_ending": "fade",
    },
    "rage": {
        "preferred_structures": [StructureTemplate.RAGE_SPIRAL, StructureTemplate.EDM_BUILD_DROP],
        "preferred_arcs": [ArcShape.SPIRAL, ArcShape.EXPLOSION, ArcShape.LINEAR_RISE],
        "max_energy": EnergyLevel.EXPLOSIVE,
        "typical_density": InstrumentDensity.FULL,
        "section_preferences": {
            SectionType.INTRO: 0.5,  # Rage often skips gentle intros
            SectionType.VERSE: 0.8,
            SectionType.CHORUS: 1.0,
            SectionType.BREAKDOWN: 0.9,
            SectionType.DROP: 1.0,
            SectionType.BUILD: 0.9,
        },
        "tempo_range": (120, 180),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "sudden",
    },
    "anger": {
        "preferred_structures": [StructureTemplate.VERSE_CHORUS, StructureTemplate.BINARY],
        "preferred_arcs": [ArcShape.PEAK, ArcShape.LINEAR_RISE],
        "max_energy": EnergyLevel.INTENSE,
        "typical_density": InstrumentDensity.FULL,
        "section_preferences": {
            SectionType.VERSE: 1.0,
            SectionType.CHORUS: 1.0,
            SectionType.BRIDGE: 0.6,
            SectionType.BREAKDOWN: 0.7,
        },
        "tempo_range": (110, 160),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "hard",
    },
    "fear": {
        "preferred_structures": [
            StructureTemplate.THROUGH_COMPOSED,
            StructureTemplate.AMBIENT_DRIFT,
        ],  # noqa: E501
        "preferred_arcs": [ArcShape.TENSION_RELEASE, ArcShape.SPIRAL],
        "max_energy": EnergyLevel.HIGH,
        "typical_density": InstrumentDensity.MEDIUM,
        "section_preferences": {
            SectionType.INTRO: 1.0,
            SectionType.VERSE: 0.9,
            SectionType.BUILD: 1.0,
            SectionType.BREAKDOWN: 0.8,
            SectionType.SILENCE: 0.7,
        },
        "tempo_range": (90, 140),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "unresolved",
    },
    "anxiety": {
        "preferred_structures": [StructureTemplate.RONDO, StructureTemplate.THROUGH_COMPOSED],
        "preferred_arcs": [ArcShape.WAVE, ArcShape.SPIRAL],
        "max_energy": EnergyLevel.HIGH,
        "typical_density": InstrumentDensity.MEDIUM,
        "section_preferences": {
            SectionType.VERSE: 1.0,
            SectionType.BUILD: 0.9,
            SectionType.INTERLUDE: 0.7,
        },
        "tempo_range": (100, 150),
        "builds_to_release": False,  # Anxiety rarely resolves
        "allows_climax": False,
        "preferred_ending": "unresolved",
    },
    "hope": {
        "preferred_structures": [StructureTemplate.VERSE_CHORUS_BRIDGE, StructureTemplate.ABABCB],
        "preferred_arcs": [ArcShape.LINEAR_RISE, ArcShape.PEAK, ArcShape.SPIRAL],
        "max_energy": EnergyLevel.HIGH,
        "typical_density": InstrumentDensity.FULL,
        "section_preferences": {
            SectionType.INTRO: 0.7,
            SectionType.VERSE: 1.0,
            SectionType.PRE_CHORUS: 0.8,
            SectionType.CHORUS: 1.0,
            SectionType.BRIDGE: 0.9,
            SectionType.OUTRO: 0.8,
        },
        "tempo_range": (100, 140),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "triumphant",
    },
    "joy": {
        "preferred_structures": [StructureTemplate.VERSE_CHORUS, StructureTemplate.ABABCB],
        "preferred_arcs": [ArcShape.PLATEAU, ArcShape.WAVE],
        "max_energy": EnergyLevel.INTENSE,
        "typical_density": InstrumentDensity.FULL,
        "section_preferences": {
            SectionType.INTRO: 0.6,
            SectionType.VERSE: 1.0,
            SectionType.CHORUS: 1.0,
            SectionType.POST_CHORUS: 0.7,
            SectionType.BRIDGE: 0.6,
            SectionType.OUTRO: 0.8,
        },
        "tempo_range": (110, 150),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "big",
    },
    "peace": {
        "preferred_structures": [StructureTemplate.AMBIENT_DRIFT, StructureTemplate.MINIMAL],
        "preferred_arcs": [ArcShape.FLAT, ArcShape.WAVE],
        "max_energy": EnergyLevel.MEDIUM_LOW,
        "typical_density": InstrumentDensity.SMALL,
        "section_preferences": {
            SectionType.INTRO: 0.9,
            SectionType.VERSE: 0.8,
            SectionType.INTERLUDE: 1.0,
            SectionType.OUTRO: 0.9,
        },
        "tempo_range": (60, 90),
        "builds_to_release": False,
        "allows_climax": False,
        "preferred_ending": "fade",
    },
    "nostalgia": {
        "preferred_structures": [StructureTemplate.AABA, StructureTemplate.VERSE_CHORUS],
        "preferred_arcs": [ArcShape.WAVE, ArcShape.VALLEY],
        "max_energy": EnergyLevel.MEDIUM,
        "typical_density": InstrumentDensity.MEDIUM,
        "section_preferences": {
            SectionType.INTRO: 1.0,
            SectionType.VERSE: 1.0,
            SectionType.CHORUS: 0.8,
            SectionType.BRIDGE: 0.9,
            SectionType.OUTRO: 1.0,
        },
        "tempo_range": (80, 110),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "fade",
    },
    "longing": {
        "preferred_structures": [StructureTemplate.VERSE_CHORUS_BRIDGE, StructureTemplate.TERNARY],
        "preferred_arcs": [ArcShape.PEAK, ArcShape.TENSION_RELEASE],
        "max_energy": EnergyLevel.HIGH,
        "typical_density": InstrumentDensity.MEDIUM,
        "section_preferences": {
            SectionType.INTRO: 0.9,
            SectionType.VERSE: 1.0,
            SectionType.PRE_CHORUS: 0.9,
            SectionType.CHORUS: 1.0,
            SectionType.BRIDGE: 0.8,
            SectionType.OUTRO: 0.9,
        },
        "tempo_range": (85, 120),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "unresolved",
    },
    "tension": {
        "preferred_structures": [
            StructureTemplate.THROUGH_COMPOSED,
            StructureTemplate.EDM_BUILD_DROP,
        ],  # noqa: E501
        "preferred_arcs": [ArcShape.TENSION_RELEASE, ArcShape.SPIRAL],
        "max_energy": EnergyLevel.INTENSE,
        "typical_density": InstrumentDensity.MEDIUM,
        "section_preferences": {
            SectionType.BUILD: 1.0,
            SectionType.BREAKDOWN: 0.9,
            SectionType.DROP: 0.8,
            SectionType.SILENCE: 0.6,
        },
        "tempo_range": (100, 150),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "sudden",
    },
    "defiance": {
        "preferred_structures": [StructureTemplate.VERSE_CHORUS, StructureTemplate.RAGE_SPIRAL],
        "preferred_arcs": [ArcShape.LINEAR_RISE, ArcShape.DOUBLE_PEAK],
        "max_energy": EnergyLevel.EXPLOSIVE,
        "typical_density": InstrumentDensity.FULL,
        "section_preferences": {
            SectionType.VERSE: 1.0,
            SectionType.CHORUS: 1.0,
            SectionType.BRIDGE: 0.7,
            SectionType.BREAKDOWN: 0.8,
            SectionType.DROP: 0.9,
        },
        "tempo_range": (110, 160),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "triumphant",
    },
    "vulnerability": {
        "preferred_structures": [StructureTemplate.MINIMAL, StructureTemplate.TERNARY],
        "preferred_arcs": [ArcShape.VALLEY, ArcShape.FLAT],
        "max_energy": EnergyLevel.MEDIUM_LOW,
        "typical_density": InstrumentDensity.SOLO,
        "section_preferences": {
            SectionType.INTRO: 1.0,
            SectionType.VERSE: 1.0,
            SectionType.BRIDGE: 0.7,
            SectionType.OUTRO: 0.9,
            SectionType.SILENCE: 0.8,
        },
        "tempo_range": (60, 90),
        "builds_to_release": False,
        "allows_climax": False,
        "preferred_ending": "fade",
    },
    "euphoria": {
        "preferred_structures": [StructureTemplate.EDM_BUILD_DROP, StructureTemplate.ABABCB],
        "preferred_arcs": [ArcShape.EXPLOSION, ArcShape.DOUBLE_PEAK],
        "max_energy": EnergyLevel.EXPLOSIVE,
        "typical_density": InstrumentDensity.MASSIVE,
        "section_preferences": {
            SectionType.BUILD: 1.0,
            SectionType.DROP: 1.0,
            SectionType.CHORUS: 1.0,
            SectionType.POST_CHORUS: 0.8,
            SectionType.BREAKDOWN: 0.7,
        },
        "tempo_range": (120, 150),
        "builds_to_release": True,
        "allows_climax": True,
        "preferred_ending": "big",
    },
    "emptiness": {
        "preferred_structures": [StructureTemplate.AMBIENT_DRIFT, StructureTemplate.MINIMAL],
        "preferred_arcs": [ArcShape.FLAT, ArcShape.LINEAR_FALL],
        "max_energy": EnergyLevel.LOW,
        "typical_density": InstrumentDensity.SOLO,
        "section_preferences": {
            SectionType.INTRO: 0.9,
            SectionType.VERSE: 0.8,
            SectionType.SILENCE: 1.0,
            SectionType.OUTRO: 0.9,
        },
        "tempo_range": (50, 80),
        "builds_to_release": False,
        "allows_climax": False,
        "preferred_ending": "fade",
    },
}


# =============================================================================
# STRUCTURE TEMPLATES
# =============================================================================

STRUCTURE_DEFINITIONS: Dict[StructureTemplate, List[Tuple[SectionType, int]]] = {
    StructureTemplate.VERSE_CHORUS: [
        (SectionType.INTRO, 4),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.OUTRO, 4),
    ],
    StructureTemplate.VERSE_CHORUS_BRIDGE: [
        (SectionType.INTRO, 4),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.BRIDGE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.OUTRO, 4),
    ],
    StructureTemplate.AABA: [
        (SectionType.VERSE, 8),
        (SectionType.VERSE, 8),
        (SectionType.BRIDGE, 8),
        (SectionType.VERSE, 8),
    ],
    StructureTemplate.ABABCB: [
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.VERSE, 8),
        (SectionType.CHORUS, 8),
        (SectionType.BRIDGE, 8),
        (SectionType.CHORUS, 8),
    ],
    StructureTemplate.THROUGH_COMPOSED: [
        (SectionType.INTRO, 4),
        (SectionType.VERSE, 8),
        (SectionType.INTERLUDE, 4),
        (SectionType.VERSE, 8),
        (SectionType.BREAKDOWN, 8),
        (SectionType.BUILD, 4),
        (SectionType.OUTRO, 8),
    ],
    StructureTemplate.RONDO: [
        (SectionType.VERSE, 8),
        (SectionType.INTERLUDE, 4),
        (SectionType.VERSE, 8),
        (SectionType.BRIDGE, 8),
        (SectionType.VERSE, 8),
    ],
    StructureTemplate.BINARY: [
        (SectionType.VERSE, 16),
        (SectionType.CHORUS, 16),
    ],
    StructureTemplate.TERNARY: [
        (SectionType.VERSE, 8),
        (SectionType.BRIDGE, 8),
        (SectionType.VERSE, 8),
    ],
    StructureTemplate.EDM_BUILD_DROP: [
        (SectionType.INTRO, 8),
        (SectionType.BUILD, 8),
        (SectionType.DROP, 8),
        (SectionType.BREAKDOWN, 8),
        (SectionType.BUILD, 8),
        (SectionType.DROP, 8),
        (SectionType.OUTRO, 4),
    ],
    StructureTemplate.AMBIENT_DRIFT: [
        (SectionType.INTRO, 8),
        (SectionType.VERSE, 16),
        (SectionType.INTERLUDE, 8),
        (SectionType.VERSE, 16),
        (SectionType.OUTRO, 8),
    ],
    StructureTemplate.MINIMAL: [
        (SectionType.INTRO, 4),
        (SectionType.VERSE, 16),
        (SectionType.OUTRO, 4),
    ],
    StructureTemplate.GRIEF_ARC: [
        (SectionType.INTRO, 4),  # Shock/numbness
        (SectionType.VERSE, 8),  # Denial
        (SectionType.INTERLUDE, 4),
        (SectionType.VERSE, 8),  # Anger
        (SectionType.BREAKDOWN, 8),  # Bargaining
        (SectionType.SILENCE, 2),
        (SectionType.VERSE, 8),  # Depression
        (SectionType.BUILD, 4),
        (SectionType.OUTRO, 8),  # Acceptance (tentative)
    ],
    StructureTemplate.RAGE_SPIRAL: [
        (SectionType.INTRO, 2),
        (SectionType.BUILD, 4),
        (SectionType.DROP, 8),
        (SectionType.BUILD, 4),
        (SectionType.DROP, 8),
        (SectionType.BREAKDOWN, 4),
        (SectionType.BUILD, 8),
        (SectionType.DROP, 8),
    ],
}


# =============================================================================
# SECTION CHARACTERISTICS
# =============================================================================

SECTION_DEFAULTS: Dict[SectionType, Dict] = {
    SectionType.INTRO: {
        "energy_range": (EnergyLevel.MINIMAL, EnergyLevel.MEDIUM),
        "density": InstrumentDensity.SMALL,
        "drums_active": False,
        "bass_active": False,
        "chords_active": True,
        "melody_active": False,
        "pads_active": True,
        "has_fill_out": True,
    },
    SectionType.VERSE: {
        "energy_range": (EnergyLevel.LOW, EnergyLevel.MEDIUM_HIGH),
        "density": InstrumentDensity.MEDIUM,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "melody_active": True,
        "pads_active": False,
    },
    SectionType.PRE_CHORUS: {
        "energy_range": (EnergyLevel.MEDIUM, EnergyLevel.HIGH),
        "density": InstrumentDensity.MEDIUM,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "melody_active": True,
        "arpeggio_active": True,
        "has_fill_out": True,
    },
    SectionType.CHORUS: {
        "energy_range": (EnergyLevel.MEDIUM_HIGH, EnergyLevel.EXPLOSIVE),
        "density": InstrumentDensity.FULL,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "melody_active": True,
        "strings_active": True,
        "counter_melody_active": True,
        "has_fill_in": True,
    },
    SectionType.POST_CHORUS: {
        "energy_range": (EnergyLevel.MEDIUM, EnergyLevel.HIGH),
        "density": InstrumentDensity.MEDIUM,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "arpeggio_active": True,
    },
    SectionType.BRIDGE: {
        "energy_range": (EnergyLevel.LOW, EnergyLevel.HIGH),
        "density": InstrumentDensity.SMALL,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "melody_active": True,
        "pads_active": True,
        "has_fill_out": True,
    },
    SectionType.BREAKDOWN: {
        "energy_range": (EnergyLevel.MINIMAL, EnergyLevel.MEDIUM),
        "density": InstrumentDensity.DUO,
        "drums_active": False,
        "bass_active": False,
        "chords_active": True,
        "melody_active": True,
        "pads_active": True,
    },
    SectionType.BUILD: {
        "energy_range": (EnergyLevel.LOW, EnergyLevel.INTENSE),
        "density": InstrumentDensity.MEDIUM,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "arpeggio_active": True,
        "has_fill_out": True,
    },
    SectionType.DROP: {
        "energy_range": (EnergyLevel.HIGH, EnergyLevel.EXPLOSIVE),
        "density": InstrumentDensity.MASSIVE,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "melody_active": True,
        "strings_active": True,
        "arpeggio_active": True,
        "has_fill_in": True,
    },
    SectionType.OUTRO: {
        "energy_range": (EnergyLevel.MINIMAL, EnergyLevel.MEDIUM),
        "density": InstrumentDensity.SMALL,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "pads_active": True,
        "fade_out_bars": 4,
    },
    SectionType.INTERLUDE: {
        "energy_range": (EnergyLevel.LOW, EnergyLevel.MEDIUM),
        "density": InstrumentDensity.TRIO,
        "drums_active": False,
        "bass_active": True,
        "chords_active": True,
        "pads_active": True,
    },
    SectionType.SOLO: {
        "energy_range": (EnergyLevel.MEDIUM, EnergyLevel.HIGH),
        "density": InstrumentDensity.MEDIUM,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "melody_active": True,  # Solo instrument
    },
    SectionType.REFRAIN: {
        "energy_range": (EnergyLevel.MEDIUM, EnergyLevel.HIGH),
        "density": InstrumentDensity.MEDIUM,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "melody_active": True,
    },
    SectionType.CODA: {
        "energy_range": (EnergyLevel.LOW, EnergyLevel.MEDIUM),
        "density": InstrumentDensity.SMALL,
        "drums_active": True,
        "bass_active": True,
        "chords_active": True,
        "pads_active": True,
        "fade_out_bars": 2,
    },
    SectionType.SILENCE: {
        "energy_range": (EnergyLevel.SILENT, EnergyLevel.MINIMAL),
        "density": InstrumentDensity.SOLO,
        "drums_active": False,
        "bass_active": False,
        "chords_active": False,
        "melody_active": False,
        "pads_active": False,
    },
}


# =============================================================================
# ARC GENERATORS
# =============================================================================


def generate_arc_curve(
    shape: ArcShape,
    num_points: int,
    start_intensity: float = 0.3,
    peak_intensity: float = 0.9,
) -> List[float]:
    """Generate intensity values for an emotional arc shape."""

    points = []

    if shape == ArcShape.LINEAR_RISE:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            points.append(start_intensity + t * (peak_intensity - start_intensity))

    elif shape == ArcShape.LINEAR_FALL:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            points.append(peak_intensity - t * (peak_intensity - start_intensity))

    elif shape == ArcShape.WAVE:
        import math

        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            wave = math.sin(t * math.pi * 2) * 0.5 + 0.5
            mid = (start_intensity + peak_intensity) / 2
            amplitude = (peak_intensity - start_intensity) / 2
            points.append(mid + wave * amplitude - amplitude / 2)

    elif shape == ArcShape.PLATEAU:
        rise_portion = 0.3
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < rise_portion:
                points.append(
                    start_intensity + (t / rise_portion) * (peak_intensity - start_intensity)
                )
            else:
                points.append(peak_intensity)

    elif shape == ArcShape.VALLEY:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.5:
                points.append(start_intensity - t * 2 * (start_intensity * 0.5))
            else:
                points.append(start_intensity * 0.5 + (t - 0.5) * 2 * (start_intensity * 0.5))

    elif shape == ArcShape.PEAK:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.6:
                points.append(start_intensity + (t / 0.6) * (peak_intensity - start_intensity))
            else:
                points.append(
                    peak_intensity - ((t - 0.6) / 0.4) * (peak_intensity - start_intensity)
                )

    elif shape == ArcShape.DOUBLE_PEAK:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.3:
                points.append(
                    start_intensity + (t / 0.3) * (peak_intensity * 0.8 - start_intensity)
                )
            elif t < 0.5:
                points.append(peak_intensity * 0.8 - ((t - 0.3) / 0.2) * (peak_intensity * 0.3))
            elif t < 0.8:
                points.append(peak_intensity * 0.5 + ((t - 0.5) / 0.3) * (peak_intensity * 0.5))
            else:
                points.append(
                    peak_intensity - ((t - 0.8) / 0.2) * (peak_intensity - start_intensity)
                )

    elif shape == ArcShape.FLAT:
        mid = (start_intensity + peak_intensity) / 2
        points = [mid] * num_points

    elif shape == ArcShape.SPIRAL:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Each cycle goes higher
            import math

            cycle = math.sin(t * math.pi * 3) * 0.2
            base = start_intensity + t * (peak_intensity - start_intensity)
            points.append(max(0.1, min(1.0, base + cycle)))

    elif shape == ArcShape.COLLAPSE:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.7:
                points.append(start_intensity + (t / 0.7) * (peak_intensity - start_intensity))
            else:
                # Sudden drop
                drop_t = (t - 0.7) / 0.3
                points.append(peak_intensity - drop_t * peak_intensity * 0.7)

    elif shape == ArcShape.EXPLOSION:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.3:
                points.append(start_intensity)
            else:
                # Sudden rise
                rise_t = (t - 0.3) / 0.7
                points.append(start_intensity + rise_t * (peak_intensity - start_intensity))

    elif shape == ArcShape.TENSION_RELEASE:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Multiple tension-release cycles
            import math

            cycle_pos = (t * 3) % 1.0
            if cycle_pos < 0.7:
                # Building tension
                points.append(
                    start_intensity
                    + (cycle_pos / 0.7) * (peak_intensity - start_intensity) * (0.5 + t * 0.5)
                )
            else:
                # Release
                points.append(start_intensity + 0.2)

    else:
        points = [0.5] * num_points

    # Clamp all values
    return [max(0.0, min(1.0, p)) for p in points]


# =============================================================================
# ARRANGEMENT ENGINE
# =============================================================================


class ArrangementEngine:
    """
    Generates complete song arrangements based on emotional intent.

    Coordinates all other engines by providing:
    - Song structure (sections and their order)
    - Emotional arc across the song
    - Per-section instrument activation
    - Energy and intensity curves
    - Transition points
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_arrangement(
        self,
        emotion: str,
        duration_minutes: float = 3.5,
        tempo_bpm: int = 120,
        key: str = "C",
        mode: str = "major",
        structure_override: Optional[StructureTemplate] = None,
        arc_override: Optional[ArcShape] = None,
        time_signature: Tuple[int, int] = (4, 4),
    ) -> ArrangementPlan:
        """
        Generate a complete arrangement plan.

        Args:
            emotion: Primary emotion for the arrangement
            duration_minutes: Target song duration
            tempo_bpm: Base tempo
            key: Musical key
            mode: Scale mode
            structure_override: Force specific structure template
            arc_override: Force specific arc shape
            time_signature: Time signature tuple

        Returns:
            ArrangementPlan with all sections and emotional arc
        """
        emotion_lower = emotion.lower()
        profile = EMOTION_ARRANGEMENT_PROFILES.get(
            emotion_lower, EMOTION_ARRANGEMENT_PROFILES["sadness"]  # Default
        )

        # Select structure
        if structure_override:
            structure = structure_override
        else:
            structure = self.rng.choice(profile["preferred_structures"])

        # Select arc shape
        if arc_override:
            arc_shape = arc_override
        else:
            arc_shape = self.rng.choice(profile["preferred_arcs"])

        # Get structure template
        structure_sections = STRUCTURE_DEFINITIONS.get(
            structure, STRUCTURE_DEFINITIONS[StructureTemplate.VERSE_CHORUS]
        )

        # Calculate timing
        beats_per_bar = time_signature[0]
        seconds_per_beat = 60.0 / tempo_bpm
        seconds_per_bar = seconds_per_beat * beats_per_bar
        target_bars = int((duration_minutes * 60) / seconds_per_bar)

        # Scale structure to target duration
        template_bars = sum(bars for _, bars in structure_sections)
        scale_factor = target_bars / template_bars if template_bars > 0 else 1.0

        # Build sections
        sections: List[SectionProfile] = []
        section_counts: Dict[SectionType, int] = {}
        current_bar = 0

        for section_type, base_bars in structure_sections:
            # Scale bars
            scaled_bars = max(2, round(base_bars * scale_factor))

            # Track variation index
            section_counts[section_type] = section_counts.get(section_type, 0) + 1
            variation_idx = section_counts[section_type] - 1

            # Get section defaults
            defaults = SECTION_DEFAULTS.get(section_type, {})

            # Calculate position-based intensity
            position = current_bar / target_bars if target_bars > 0 else 0.5
            arc_curve = generate_arc_curve(arc_shape, 10)
            arc_index = min(int(position * len(arc_curve)), len(arc_curve) - 1)
            arc_intensity = arc_curve[arc_index]

            # Determine energy based on emotion profile and arc
            energy_range = defaults.get("energy_range", (EnergyLevel.LOW, EnergyLevel.HIGH))
            max_energy_value = min(profile["max_energy"].value, energy_range[1].value)
            min_energy_value = energy_range[0].value

            energy_value = int(
                min_energy_value + arc_intensity * (max_energy_value - min_energy_value)
            )
            energy = EnergyLevel(min(max(energy_value, 0), 8))

            # Build section profile
            section = SectionProfile(
                section_type=section_type,
                bars=scaled_bars,
                energy=energy,
                density=defaults.get("density", profile["typical_density"]),
                emotion_intensity=arc_intensity,
                drums_active=defaults.get("drums_active", True),
                bass_active=defaults.get("bass_active", True),
                chords_active=defaults.get("chords_active", True),
                melody_active=defaults.get("melody_active", True),
                pads_active=defaults.get("pads_active", False),
                strings_active=defaults.get("strings_active", False),
                arpeggio_active=defaults.get("arpeggio_active", False),
                counter_melody_active=defaults.get("counter_melody_active", False),
                has_fill_in=defaults.get("has_fill_in", False),
                has_fill_out=defaults.get("has_fill_out", False),
                fade_in_bars=defaults.get("fade_in_bars", 0),
                fade_out_bars=defaults.get("fade_out_bars", 0),
                variation_index=variation_idx,
                variation_intensity=variation_idx * 0.15,  # Each repeat slightly different
            )

            # Apply emotion-specific overrides
            section = self._apply_emotion_overrides(section, emotion_lower, profile)

            sections.append(section)
            current_bar += scaled_bars

        # Generate emotional arc points
        emotional_arc = self._generate_emotional_arc(
            emotion=emotion_lower,
            arc_shape=arc_shape,
            num_sections=len(sections),
        )

        # Calculate totals
        total_bars = sum(s.bars for s in sections)
        duration_seconds = total_bars * seconds_per_bar

        return ArrangementPlan(
            sections=sections,
            emotional_arc=emotional_arc,
            total_bars=total_bars,
            structure_template=structure,
            arc_shape=arc_shape,
            base_tempo=tempo_bpm,
            key=key,
            mode=mode,
            time_signature=time_signature,
            estimated_duration_seconds=duration_seconds,
            section_count=len(sections),
            unique_sections=len(set(s.section_type for s in sections)),
        )

    def _apply_emotion_overrides(
        self,
        section: SectionProfile,
        emotion: str,
        profile: Dict,
    ) -> SectionProfile:
        """Apply emotion-specific modifications to section."""

        # Grief: Strip back density, add pads
        if emotion in ["grief", "emptiness", "vulnerability"]:
            if section.section_type == SectionType.CHORUS:
                section.drums_active = False
                section.energy = EnergyLevel(max(0, section.energy.value - 2))
            section.pads_active = True
            section.strings_active = section.emotion_intensity > 0.5

        # Rage/Anger: Maximize energy, add all instruments
        elif emotion in ["rage", "anger", "defiance"]:
            if section.section_type in [SectionType.CHORUS, SectionType.DROP]:
                section.drums_active = True
                section.bass_active = True
                section.strings_active = True
                section.counter_melody_active = True
                section.density = InstrumentDensity.FULL

        # Anxiety: Keep things restless, use arpeggios
        elif emotion == "anxiety":
            section.arpeggio_active = True
            section.humanization = 0.7  # More unstable
            if section.section_type == SectionType.OUTRO:
                section.fade_out_bars = 0  # Anxiety doesn't resolve

        # Hope/Joy: Full instrumentation on climax
        elif emotion in ["hope", "joy", "euphoria"]:
            if section.emotion_intensity > 0.7:
                section.strings_active = True
                section.counter_melody_active = True

        # Fear: Use silence and builds
        elif emotion == "fear":
            if section.section_type == SectionType.BREAKDOWN:
                section.pads_active = True
                section.drums_active = False

        # Peace: Minimal, pad-heavy
        elif emotion == "peace":
            section.drums_active = section.emotion_intensity > 0.3
            section.pads_active = True
            section.arpeggio_active = False

        return section

    def _generate_emotional_arc(
        self,
        emotion: str,
        arc_shape: ArcShape,
        num_sections: int,
    ) -> List[EmotionArcPoint]:
        """Generate emotional arc points for the song."""

        arc_curve = generate_arc_curve(arc_shape, num_sections)
        arc_points = []

        # Get related emotions for secondary emotion assignment
        emotion_relatives = {
            "grief": ["sadness", "emptiness", "longing"],
            "sadness": ["melancholy", "grief", "longing"],
            "rage": ["anger", "defiance", "tension"],
            "anger": ["rage", "defiance", "tension"],
            "fear": ["anxiety", "tension", "dread"],
            "anxiety": ["fear", "tension", "restlessness"],
            "hope": ["joy", "longing", "peace"],
            "joy": ["hope", "euphoria", "peace"],
            "peace": ["hope", "contentment", "nostalgia"],
            "nostalgia": ["melancholy", "longing", "peace"],
            "longing": ["nostalgia", "hope", "sadness"],
            "tension": ["fear", "anxiety", "anticipation"],
            "defiance": ["rage", "hope", "anger"],
            "vulnerability": ["fear", "sadness", "hope"],
            "euphoria": ["joy", "hope", "peace"],
            "emptiness": ["grief", "sadness", "numbness"],
            "melancholy": ["sadness", "nostalgia", "peace"],
        }

        relatives = emotion_relatives.get(emotion, [emotion])

        for i, intensity in enumerate(arc_curve):
            position = i / (num_sections - 1) if num_sections > 1 else 0.5

            # Determine secondary emotion based on position
            if intensity > 0.7:
                secondary = relatives[0] if relatives else None
            elif intensity < 0.3:
                secondary = relatives[-1] if relatives else None
            else:
                secondary = self.rng.choice(relatives) if relatives else None

            # Calculate tension (generally higher in middle, lower at ends)
            tension = 0.3 + abs(position - 0.5) * 0.4 + intensity * 0.3
            tension = min(1.0, tension)

            arc_points.append(
                EmotionArcPoint(
                    position=position,
                    primary_emotion=emotion,
                    intensity=intensity,
                    secondary_emotion=secondary if secondary != emotion else None,
                    tension=tension,
                )
            )

        return arc_points

    def get_section_at_bar(
        self,
        arrangement: ArrangementPlan,
        bar: int,
    ) -> Optional[Tuple[SectionProfile, int]]:
        """
        Get the section and local bar number for a given global bar.

        Returns:
            Tuple of (SectionProfile, local_bar) or None if bar out of range
        """
        current = 0
        for section in arrangement.sections:
            if current <= bar < current + section.bars:
                return (section, bar - current)
            current += section.bars
        return None

    def get_arc_at_position(
        self,
        arrangement: ArrangementPlan,
        position: float,
    ) -> EmotionArcPoint:
        """
        Get interpolated emotional arc point at any position (0.0-1.0).
        """
        if not arrangement.emotional_arc:
            return EmotionArcPoint(
                position=position,
                primary_emotion="neutral",
                intensity=0.5,
            )

        # Find surrounding arc points
        prev_point = arrangement.emotional_arc[0]
        next_point = arrangement.emotional_arc[-1]

        for i, point in enumerate(arrangement.emotional_arc):
            if point.position >= position:
                next_point = point
                prev_point = arrangement.emotional_arc[max(0, i - 1)]
                break

        # Interpolate
        if prev_point.position == next_point.position:
            return prev_point

        t = (position - prev_point.position) / (next_point.position - prev_point.position)

        return EmotionArcPoint(
            position=position,
            primary_emotion=prev_point.primary_emotion,
            intensity=prev_point.intensity + t * (next_point.intensity - prev_point.intensity),
            secondary_emotion=(
                prev_point.secondary_emotion if t < 0.5 else next_point.secondary_emotion
            ),
            tension=prev_point.tension + t * (next_point.tension - prev_point.tension),
        )

    def suggest_transitions(
        self,
        arrangement: ArrangementPlan,
    ) -> List[SectionTransition]:
        """Generate suggested transitions between sections."""

        transitions = []

        for i in range(len(arrangement.sections) - 1):
            current = arrangement.sections[i]
            next_section = arrangement.sections[i + 1]

            energy_diff = next_section.energy.value - current.energy.value

            # Determine transition type
            if energy_diff > 2:
                trans_type = "build"
                bars = 2
                curve = "exponential"
            elif energy_diff < -2:
                trans_type = "breakdown"
                bars = 2
                curve = "sudden"
            elif current.has_fill_out or next_section.has_fill_in:
                trans_type = "fill"
                bars = 1
                curve = "linear"
            elif next_section.section_type == SectionType.SILENCE:
                trans_type = "fade"
                bars = 2
                curve = "linear"
            else:
                trans_type = "cut"
                bars = 0
                curve = "sudden"

            transitions.append(
                SectionTransition(
                    from_section=current.section_type,
                    to_section=next_section.section_type,
                    transition_type=trans_type,
                    bars=bars,
                    energy_curve=curve,
                )
            )

        return transitions

    def print_arrangement(self, arrangement: ArrangementPlan) -> str:
        """Generate a visual representation of the arrangement."""

        lines = []
        lines.append(f"═══ ARRANGEMENT: {arrangement.structure_template.value.upper()} ═══")
        lines.append(
            f"Key: {arrangement.key} {arrangement.mode} | Tempo: {arrangement.base_tempo} BPM"
        )
        lines.append(
            f"Duration: {arrangement.estimated_duration_seconds / 60:.1f} min | Bars: "
            f"{arrangement.total_bars}"
        )  # noqa: E501

        lines.append(f"Arc: {arrangement.arc_shape.value}")
        lines.append("")

        # Section timeline
        current_bar = 0
        for i, section in enumerate(arrangement.sections):
            energy_bar = "█" * section.energy.value + "░" * (8 - section.energy.value)

            instruments = []
            if section.drums_active:
                instruments.append("D")
            if section.bass_active:
                instruments.append("B")
            if section.chords_active:
                instruments.append("C")
            if section.melody_active:
                instruments.append("M")
            if section.pads_active:
                instruments.append("P")
            if section.strings_active:
                instruments.append("S")
            if section.arpeggio_active:
                instruments.append("A")
            if section.counter_melody_active:
                instruments.append("X")

            inst_str = "".join(instruments) if instruments else "-"

            var_str = f" (v{section.variation_index + 1})" if section.variation_index > 0 else ""

            lines.append(
                f"  {current_bar:3d}-{current_bar + section.bars:3d} │ "
                f"{section.section_type.value:12s}{var_str:5s} │ "
                f"[{energy_bar}] │ {inst_str:8s} │ "
                f"Int: {section.emotion_intensity:.2f}"
            )

            current_bar += section.bars

        lines.append("")

        # Emotional arc visualization
        lines.append("Emotional Arc:")
        arc_width = 40
        for point in arrangement.emotional_arc:
            bar_pos = int(point.intensity * arc_width)
            bar = "─" * bar_pos + "●" + "─" * (arc_width - bar_pos - 1)
            lines.append(f"  {point.position:.1f} [{bar}] {point.intensity:.2f}")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_arrangement(
    emotion: str,
    duration_minutes: float = 3.5,
    tempo_bpm: int = 120,
    key: str = "C",
    mode: str = "major",
    seed: Optional[int] = None,
) -> ArrangementPlan:
    """Quick arrangement generation."""
    engine = ArrangementEngine(seed=seed)
    return engine.generate_arrangement(
        emotion=emotion,
        duration_minutes=duration_minutes,
        tempo_bpm=tempo_bpm,
        key=key,
        mode=mode,
    )


def get_available_structures() -> List[str]:
    """List available structure templates."""
    return [s.value for s in StructureTemplate]


def get_available_arcs() -> List[str]:
    """List available arc shapes."""
    return [a.value for a in ArcShape]


def get_section_types() -> List[str]:
    """List available section types."""
    return [s.value for s in SectionType]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║     KELLY ARRANGEMENT ENGINE - TIER 3    ║")
    print("╚══════════════════════════════════════════╝\n")

    engine = ArrangementEngine(seed=42)

    # Test different emotions
    test_emotions = ["grief", "rage", "hope", "anxiety", "peace"]

    for emotion in test_emotions:
        print(f"\n{'='*60}")
        print(f"EMOTION: {emotion.upper()}")
        print("=" * 60)

        arrangement = engine.generate_arrangement(
            emotion=emotion,
            duration_minutes=3.5,
            tempo_bpm=100 if emotion in ["grief", "peace"] else 130,
            key="F" if emotion == "grief" else "C",
            mode="minor" if emotion in ["grief", "rage", "anxiety"] else "major",
        )

        print(engine.print_arrangement(arrangement))

        # Show transitions
        print("\nTransitions:")
        for trans in engine.suggest_transitions(arrangement):
            print(
                f"  {trans.from_section.value} → {trans.to_section.value}: {trans.transition_type} "
                f"({trans.bars} bars)"
            )  # noqa: E501
