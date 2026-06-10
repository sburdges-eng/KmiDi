"""
String Engine - Generates emotion-driven orchestral string arrangements.

Part of Kelly MIDI Companion Tier 2.

Philosophy: Strings are the voice of the soul. They can whisper intimate
grief or roar with collective rage. A tremolo string speaks fear. A
pizzicato speaks playfulness or unease. A sustained legato line is
pure emotional breath. The articulation IS the emotion.

GM String Programs:
    40 = Violin
    41 = Viola
    42 = Cello
    43 = Contrabass
    44 = Tremolo Strings
    45 = Pizzicato Strings
    48 = String Ensemble 1
    49 = String Ensemble 2
    50 = Synth Strings 1
    51 = Synth Strings 2

Integration:
    from kellymidicompanion_string_engine import StringEngine, StringConfig

    engine = StringEngine()
    strings = engine.generate(
        emotion="grief",
        chord_progression=["F", "C", "Dm", "Bbm"],
        key="F",
        bars=4,
        tempo_bpm=82
    )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import math

# =============================================================================
# CONSTANTS
# =============================================================================

CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

TICKS_PER_BEAT = 480

CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "m": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "m7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 14],
}

# GM String instruments
GM_STRINGS = {
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "tremolo": 44,
    "pizzicato": 45,
    "harp": 46,
    "timpani": 47,
    "ensemble_1": 48,
    "ensemble_2": 49,
    "synth_1": 50,
    "synth_2": 51,
}

# String ranges (MIDI note numbers)
STRING_RANGES = {
    "violin": (55, 96),  # G3 to C7
    "viola": (48, 84),  # C3 to C6
    "cello": (36, 72),  # C2 to C5
    "contrabass": (28, 60),  # E1 to C4
    "ensemble": (36, 84),  # Full range
}


# =============================================================================
# ENUMS
# =============================================================================


class StringArticulation(Enum):
    """String playing techniques."""

    LEGATO = "legato"  # Smooth, connected
    STACCATO = "staccato"  # Short, detached
    SPICCATO = "spiccato"  # Bouncing bow
    PIZZICATO = "pizzicato"  # Plucked
    TREMOLO = "tremolo"  # Rapid bow trembling
    TREMOLO_MEASURED = "tremolo_measured"  # Rhythmic tremolo
    MARCATO = "marcato"  # Accented
    TENUTO = "tenuto"  # Full value, slight accent
    COL_LEGNO = "col_legno"  # Wood of bow
    SUL_PONT = "sul_pont"  # Near bridge (glassy)
    SUL_TASTO = "sul_tasto"  # Over fingerboard (soft)
    HARMONIC = "harmonic"  # Flageolet tones


class StringDynamic(Enum):
    """Dynamic expression patterns."""

    FLAT = "flat"  # Constant level
    SWELL = "swell"  # Crescendo then diminuendo
    CRESCENDO = "crescendo"  # Building
    DIMINUENDO = "diminuendo"  # Fading
    SFORZANDO = "sforzando"  # Sudden accent
    FP = "fp"  # Forte-piano (loud then soft)
    HAIRPIN = "hairpin"  # Quick swell
    PULSING = "pulsing"  # Rhythmic dynamics


class StringVoicing(Enum):
    """How string parts are arranged."""

    UNISON = "unison"  # All same notes
    OCTAVES = "octaves"  # Doubled at octave
    DIVISI = "divisi"  # Split into parts
    FULL_SECTION = "full_section"  # 4-part harmony
    SOLO = "solo"  # Single voice
    CLUSTER = "cluster"  # Close dissonant voicing
    OPEN = "open"  # Wide spread


class StringRole(Enum):
    """Function in arrangement."""

    PAD = "pad"  # Sustained background
    MELODY = "melody"  # Lead line
    COUNTER = "counter"  # Countermelody
    OSTINATO = "ostinato"  # Repeating figure
    PUNCTUATION = "punctuation"  # Rhythmic hits
    SWELL = "swell"  # Dynamic builds
    TEXTURE = "texture"  # Atmospheric


class StringSection(Enum):
    """Which string section."""

    VIOLIN_1 = "violin_1"
    VIOLIN_2 = "violin_2"
    VIOLA = "viola"
    CELLO = "cello"
    BASS = "bass"
    FULL = "full"


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_PROFILES = {
    "grief": {
        "articulation": StringArticulation.LEGATO,
        "dynamic": StringDynamic.SWELL,
        "voicing": StringVoicing.DIVISI,
        "role": StringRole.PAD,
        "velocity_range": (40, 75),
        "attack_time": 0.3,  # Slow attack (0-1)
        "release_time": 0.8,  # Long release
        "vibrato_intensity": 0.6,
        "expression_depth": 0.8,
        "sustain_ratio": 0.95,
        "humanize_timing": 20,
        "humanize_velocity": 10,
        "preferred_section": StringSection.CELLO,
        "gm_instrument": GM_STRINGS["ensemble_1"],
    },
    "sadness": {
        "articulation": StringArticulation.LEGATO,
        "dynamic": StringDynamic.DIMINUENDO,
        "voicing": StringVoicing.SOLO,
        "role": StringRole.MELODY,
        "velocity_range": (35, 65),
        "attack_time": 0.25,
        "release_time": 0.7,
        "vibrato_intensity": 0.5,
        "expression_depth": 0.7,
        "sustain_ratio": 0.9,
        "humanize_timing": 15,
        "humanize_velocity": 8,
        "preferred_section": StringSection.VIOLIN_1,
        "gm_instrument": GM_STRINGS["violin"],
    },
    "hope": {
        "articulation": StringArticulation.LEGATO,
        "dynamic": StringDynamic.CRESCENDO,
        "voicing": StringVoicing.OCTAVES,
        "role": StringRole.SWELL,
        "velocity_range": (50, 90),
        "attack_time": 0.2,
        "release_time": 0.5,
        "vibrato_intensity": 0.4,
        "expression_depth": 0.7,
        "sustain_ratio": 0.9,
        "humanize_timing": 10,
        "humanize_velocity": 8,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["ensemble_1"],
    },
    "joy": {
        "articulation": StringArticulation.SPICCATO,
        "dynamic": StringDynamic.PULSING,
        "voicing": StringVoicing.FULL_SECTION,
        "role": StringRole.OSTINATO,
        "velocity_range": (70, 110),
        "attack_time": 0.1,
        "release_time": 0.3,
        "vibrato_intensity": 0.3,
        "expression_depth": 0.5,
        "sustain_ratio": 0.5,
        "humanize_timing": 5,
        "humanize_velocity": 10,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["ensemble_1"],
    },
    "rage": {
        "articulation": StringArticulation.MARCATO,
        "dynamic": StringDynamic.SFORZANDO,
        "voicing": StringVoicing.UNISON,
        "role": StringRole.PUNCTUATION,
        "velocity_range": (100, 127),
        "attack_time": 0.05,
        "release_time": 0.2,
        "vibrato_intensity": 0.1,
        "expression_depth": 0.3,
        "sustain_ratio": 0.4,
        "humanize_timing": 3,
        "humanize_velocity": 5,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["ensemble_1"],
    },
    "anger": {
        "articulation": StringArticulation.TREMOLO,
        "dynamic": StringDynamic.CRESCENDO,
        "voicing": StringVoicing.OCTAVES,
        "role": StringRole.TEXTURE,
        "velocity_range": (80, 115),
        "attack_time": 0.1,
        "release_time": 0.3,
        "vibrato_intensity": 0.2,
        "expression_depth": 0.6,
        "sustain_ratio": 0.95,
        "humanize_timing": 5,
        "humanize_velocity": 8,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["tremolo"],
    },
    "fear": {
        "articulation": StringArticulation.TREMOLO,
        "dynamic": StringDynamic.HAIRPIN,
        "voicing": StringVoicing.CLUSTER,
        "role": StringRole.TEXTURE,
        "velocity_range": (30, 80),
        "attack_time": 0.15,
        "release_time": 0.6,
        "vibrato_intensity": 0.8,
        "expression_depth": 0.9,
        "sustain_ratio": 0.85,
        "humanize_timing": 25,
        "humanize_velocity": 15,
        "preferred_section": StringSection.VIOLA,
        "gm_instrument": GM_STRINGS["tremolo"],
    },
    "anxiety": {
        "articulation": StringArticulation.SUL_PONT,
        "dynamic": StringDynamic.PULSING,
        "voicing": StringVoicing.DIVISI,
        "role": StringRole.OSTINATO,
        "velocity_range": (45, 85),
        "attack_time": 0.1,
        "release_time": 0.4,
        "vibrato_intensity": 0.7,
        "expression_depth": 0.8,
        "sustain_ratio": 0.6,
        "humanize_timing": 30,
        "humanize_velocity": 15,
        "preferred_section": StringSection.VIOLIN_1,
        "gm_instrument": GM_STRINGS["ensemble_2"],
    },
    "tension": {
        "articulation": StringArticulation.TREMOLO,
        "dynamic": StringDynamic.CRESCENDO,
        "voicing": StringVoicing.CLUSTER,
        "role": StringRole.SWELL,
        "velocity_range": (40, 100),
        "attack_time": 0.2,
        "release_time": 0.5,
        "vibrato_intensity": 0.5,
        "expression_depth": 0.9,
        "sustain_ratio": 0.95,
        "humanize_timing": 15,
        "humanize_velocity": 10,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["tremolo"],
    },
    "longing": {
        "articulation": StringArticulation.LEGATO,
        "dynamic": StringDynamic.SWELL,
        "voicing": StringVoicing.SOLO,
        "role": StringRole.MELODY,
        "velocity_range": (45, 80),
        "attack_time": 0.3,
        "release_time": 0.7,
        "vibrato_intensity": 0.7,
        "expression_depth": 0.85,
        "sustain_ratio": 0.9,
        "humanize_timing": 18,
        "humanize_velocity": 10,
        "preferred_section": StringSection.CELLO,
        "gm_instrument": GM_STRINGS["cello"],
    },
    "nostalgia": {
        "articulation": StringArticulation.SUL_TASTO,
        "dynamic": StringDynamic.SWELL,
        "voicing": StringVoicing.DIVISI,
        "role": StringRole.PAD,
        "velocity_range": (40, 70),
        "attack_time": 0.35,
        "release_time": 0.8,
        "vibrato_intensity": 0.5,
        "expression_depth": 0.7,
        "sustain_ratio": 0.95,
        "humanize_timing": 20,
        "humanize_velocity": 10,
        "preferred_section": StringSection.VIOLA,
        "gm_instrument": GM_STRINGS["ensemble_2"],
    },
    "peace": {
        "articulation": StringArticulation.LEGATO,
        "dynamic": StringDynamic.FLAT,
        "voicing": StringVoicing.OPEN,
        "role": StringRole.PAD,
        "velocity_range": (35, 55),
        "attack_time": 0.4,
        "release_time": 0.9,
        "vibrato_intensity": 0.3,
        "expression_depth": 0.4,
        "sustain_ratio": 1.0,
        "humanize_timing": 10,
        "humanize_velocity": 5,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["ensemble_2"],
    },
    "euphoria": {
        "articulation": StringArticulation.LEGATO,
        "dynamic": StringDynamic.CRESCENDO,
        "voicing": StringVoicing.FULL_SECTION,
        "role": StringRole.SWELL,
        "velocity_range": (60, 115),
        "attack_time": 0.15,
        "release_time": 0.4,
        "vibrato_intensity": 0.5,
        "expression_depth": 0.8,
        "sustain_ratio": 0.95,
        "humanize_timing": 8,
        "humanize_velocity": 10,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["ensemble_1"],
    },
    "wonder": {
        "articulation": StringArticulation.HARMONIC,
        "dynamic": StringDynamic.SWELL,
        "voicing": StringVoicing.OPEN,
        "role": StringRole.TEXTURE,
        "velocity_range": (40, 75),
        "attack_time": 0.3,
        "release_time": 0.7,
        "vibrato_intensity": 0.2,
        "expression_depth": 0.6,
        "sustain_ratio": 0.85,
        "humanize_timing": 15,
        "humanize_velocity": 10,
        "preferred_section": StringSection.VIOLIN_1,
        "gm_instrument": GM_STRINGS["ensemble_2"],
    },
    "playful": {
        "articulation": StringArticulation.PIZZICATO,
        "dynamic": StringDynamic.PULSING,
        "voicing": StringVoicing.DIVISI,
        "role": StringRole.OSTINATO,
        "velocity_range": (60, 95),
        "attack_time": 0.05,
        "release_time": 0.2,
        "vibrato_intensity": 0.0,
        "expression_depth": 0.3,
        "sustain_ratio": 0.3,
        "humanize_timing": 8,
        "humanize_velocity": 12,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["pizzicato"],
    },
    "mystery": {
        "articulation": StringArticulation.SUL_PONT,
        "dynamic": StringDynamic.HAIRPIN,
        "voicing": StringVoicing.CLUSTER,
        "role": StringRole.TEXTURE,
        "velocity_range": (25, 60),
        "attack_time": 0.25,
        "release_time": 0.8,
        "vibrato_intensity": 0.4,
        "expression_depth": 0.7,
        "sustain_ratio": 0.9,
        "humanize_timing": 25,
        "humanize_velocity": 12,
        "preferred_section": StringSection.VIOLA,
        "gm_instrument": GM_STRINGS["ensemble_2"],
    },
    "dread": {
        "articulation": StringArticulation.COL_LEGNO,
        "dynamic": StringDynamic.CRESCENDO,
        "voicing": StringVoicing.UNISON,
        "role": StringRole.PUNCTUATION,
        "velocity_range": (30, 90),
        "attack_time": 0.1,
        "release_time": 0.5,
        "vibrato_intensity": 0.2,
        "expression_depth": 0.8,
        "sustain_ratio": 0.5,
        "humanize_timing": 20,
        "humanize_velocity": 15,
        "preferred_section": StringSection.CELLO,
        "gm_instrument": GM_STRINGS["ensemble_2"],
    },
    "neutral": {
        "articulation": StringArticulation.LEGATO,
        "dynamic": StringDynamic.FLAT,
        "voicing": StringVoicing.DIVISI,
        "role": StringRole.PAD,
        "velocity_range": (55, 75),
        "attack_time": 0.2,
        "release_time": 0.5,
        "vibrato_intensity": 0.4,
        "expression_depth": 0.5,
        "sustain_ratio": 0.85,
        "humanize_timing": 10,
        "humanize_velocity": 8,
        "preferred_section": StringSection.FULL,
        "gm_instrument": GM_STRINGS["ensemble_1"],
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class StringNote:
    """A single string note."""

    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    section: StringSection
    articulation: StringArticulation
    voice_function: str  # "root", "third", "fifth", etc.
    velocity_curve: List[int] = field(default_factory=list)


@dataclass
class StringVoice:
    """A complete string voice/part."""

    section: StringSection
    notes: List[StringNote]
    gm_instrument: int
    channel: int


@dataclass
class StringConfig:
    """Configuration for string generation."""

    emotion: str = "neutral"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    articulation_override: Optional[StringArticulation] = None
    voicing_override: Optional[StringVoicing] = None
    role_override: Optional[StringRole] = None


@dataclass
class StringOutput:
    """Complete string output."""

    voices: List[StringVoice]
    notes: List[StringNote]  # All notes flattened
    emotion: str
    articulation_used: StringArticulation
    dynamic_used: StringDynamic
    voicing_used: StringVoicing
    role_used: StringRole
    gm_instrument: int
    total_ticks: int
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def note_name_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.strip()
    if len(note) > 1 and note[1] in ["#", "b"]:
        base = note[0].upper()
        modifier = note[1]
    else:
        base = note[0].upper()
        modifier = ""

    base_idx = CHROMATIC.index(base) if base in CHROMATIC else 0

    if modifier == "#":
        base_idx += 1
    elif modifier == "b":
        base_idx -= 1

    base_idx = base_idx % 12
    return base_idx + (octave + 1) * 12


def parse_chord(chord_symbol: str) -> Tuple[str, str, Optional[str]]:
    """Parse chord symbol."""
    chord_symbol = chord_symbol.strip()

    bass_note = None
    if "/" in chord_symbol:
        chord_symbol, bass_note = chord_symbol.split("/")

    if len(chord_symbol) > 1 and chord_symbol[1] in ["#", "b"]:
        root = chord_symbol[:2]
        quality = chord_symbol[2:] or "maj"
    else:
        root = chord_symbol[0]
        quality = chord_symbol[1:] or "maj"

    quality = quality.lower()
    if quality == "":
        quality = "maj"
    elif quality in ["m", "-"]:
        quality = "min"

    return root, quality, bass_note


def get_chord_pitches(root: str, quality: str, base_octave: int = 4) -> List[int]:
    """Get MIDI pitches for a chord."""
    root_midi = note_name_to_midi(root, base_octave)
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])
    return [root_midi + i for i in intervals]


def get_section_range(section: StringSection) -> Tuple[int, int]:
    """Get MIDI note range for section."""
    if section == StringSection.VIOLIN_1:
        return STRING_RANGES["violin"]
    elif section == StringSection.VIOLIN_2:
        return (52, 88)  # Slightly lower than V1
    elif section == StringSection.VIOLA:
        return STRING_RANGES["viola"]
    elif section == StringSection.CELLO:
        return STRING_RANGES["cello"]
    elif section == StringSection.BASS:
        return STRING_RANGES["contrabass"]
    else:
        return STRING_RANGES["ensemble"]


def get_section_instrument(section: StringSection) -> int:
    """Get GM instrument for section."""
    mapping = {
        StringSection.VIOLIN_1: GM_STRINGS["violin"],
        StringSection.VIOLIN_2: GM_STRINGS["violin"],
        StringSection.VIOLA: GM_STRINGS["viola"],
        StringSection.CELLO: GM_STRINGS["cello"],
        StringSection.BASS: GM_STRINGS["contrabass"],
        StringSection.FULL: GM_STRINGS["ensemble_1"],
    }
    return mapping.get(section, GM_STRINGS["ensemble_1"])


def apply_voicing_to_pitches(
    pitches: List[int], voicing: StringVoicing, profile: Dict
) -> List[Tuple[int, str, StringSection]]:
    """Apply voicing style. Returns (pitch, function, section) tuples."""
    if not pitches:
        return []

    root = pitches[0]
    result = []

    if voicing == StringVoicing.UNISON:
        result.append((root, "root", StringSection.FULL))

    elif voicing == StringVoicing.OCTAVES:
        result.append((root, "root", StringSection.CELLO))
        result.append((root + 12, "octave", StringSection.VIOLA))
        result.append((root + 24, "high_octave", StringSection.VIOLIN_1))

    elif voicing == StringVoicing.SOLO:
        result.append((root, "root", profile["preferred_section"]))
        if len(pitches) > 2:
            result.append((pitches[2], "fifth", profile["preferred_section"]))

    elif voicing == StringVoicing.DIVISI:
        result.append((root, "root", StringSection.CELLO))
        if len(pitches) > 1:
            result.append((pitches[1] + 12, "third", StringSection.VIOLA))
        if len(pitches) > 2:
            result.append((pitches[2] + 12, "fifth", StringSection.VIOLIN_2))

    elif voicing == StringVoicing.FULL_SECTION:
        result.append((root - 12, "bass", StringSection.BASS))
        result.append((root, "root", StringSection.CELLO))
        if len(pitches) > 1:
            result.append((pitches[1] + 12, "third", StringSection.VIOLA))
        if len(pitches) > 2:
            result.append((pitches[2] + 12, "fifth", StringSection.VIOLIN_2))
            result.append((pitches[2] + 24, "high_fifth", StringSection.VIOLIN_1))

    elif voicing == StringVoicing.CLUSTER:
        # Close voicing with possible dissonance
        result.append((root, "root", StringSection.VIOLA))
        for i, p in enumerate(pitches[1:], 1):
            # Keep close together
            adj_p = p
            while adj_p > root + 7:
                adj_p -= 12
            func = ["root", "third", "fifth", "seventh"][min(i, 3)]
            sect = [StringSection.CELLO, StringSection.VIOLA, StringSection.VIOLIN_2][i % 3]
            result.append((adj_p, func, sect))

    elif voicing == StringVoicing.OPEN:
        # Wide spread
        result.append((root - 12, "bass", StringSection.BASS))
        result.append((root, "root", StringSection.CELLO))
        if len(pitches) > 2:
            result.append((pitches[2] + 12, "fifth", StringSection.VIOLA))
        if len(pitches) > 1:
            result.append((pitches[1] + 24, "third", StringSection.VIOLIN_1))

    return result


def generate_velocity_curve(
    base_velocity: int, duration_ticks: int, dynamic: StringDynamic, expression_depth: float
) -> List[int]:
    """Generate expression curve for string note."""
    num_points = max(4, duration_ticks // (TICKS_PER_BEAT // 2))
    depth = expression_depth * 0.4  # Max 40% variation

    if dynamic == StringDynamic.FLAT:
        return [base_velocity] * num_points

    elif dynamic == StringDynamic.SWELL:
        curve = []
        for i in range(num_points):
            progress = i / num_points
            # Bell curve
            mod = math.sin(progress * math.pi) * depth
            curve.append(max(1, min(127, int(base_velocity * (1 + mod)))))
        return curve

    elif dynamic == StringDynamic.CRESCENDO:
        curve = []
        for i in range(num_points):
            progress = i / num_points
            mod = progress * depth
            curve.append(max(1, min(127, int(base_velocity * (1 + mod)))))
        return curve

    elif dynamic == StringDynamic.DIMINUENDO:
        curve = []
        for i in range(num_points):
            progress = i / num_points
            mod = (1 - progress) * depth
            curve.append(max(1, min(127, int(base_velocity * (1 + mod)))))
        return curve

    elif dynamic == StringDynamic.SFORZANDO:
        curve = [int(base_velocity * 1.3)]
        for i in range(1, num_points):
            curve.append(int(base_velocity * 0.7))
        return [max(1, min(127, v)) for v in curve]

    elif dynamic == StringDynamic.FP:
        curve = [int(base_velocity * 1.2)]
        for i in range(1, num_points):
            progress = i / num_points
            curve.append(int(base_velocity * (0.5 + 0.3 * progress)))
        return [max(1, min(127, v)) for v in curve]

    elif dynamic == StringDynamic.HAIRPIN:
        curve = []
        for i in range(num_points):
            progress = i / num_points
            if progress < 0.3:
                mod = (progress / 0.3) * depth
            else:
                mod = ((1 - progress) / 0.7) * depth
            curve.append(max(1, min(127, int(base_velocity * (1 + mod)))))
        return curve

    elif dynamic == StringDynamic.PULSING:
        curve = []
        for i in range(num_points):
            mod = math.sin(i * math.pi) * depth * 0.5
            curve.append(max(1, min(127, int(base_velocity * (1 + mod)))))
        return curve

    return [base_velocity] * num_points


def get_articulation_duration_modifier(articulation: StringArticulation) -> float:
    """Get duration modifier based on articulation."""
    modifiers = {
        StringArticulation.LEGATO: 0.98,
        StringArticulation.STACCATO: 0.3,
        StringArticulation.SPICCATO: 0.25,
        StringArticulation.PIZZICATO: 0.2,
        StringArticulation.TREMOLO: 1.0,
        StringArticulation.TREMOLO_MEASURED: 1.0,
        StringArticulation.MARCATO: 0.7,
        StringArticulation.TENUTO: 1.0,
        StringArticulation.COL_LEGNO: 0.4,
        StringArticulation.SUL_PONT: 0.9,
        StringArticulation.SUL_TASTO: 0.95,
        StringArticulation.HARMONIC: 0.8,
    }
    return modifiers.get(articulation, 0.85)


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================


class StringEngine:
    """
    Generates emotion-driven orchestral string arrangements.

    Strings provide the emotional voice - from intimate solo lines
    to powerful tutti swells.
    """

    def __init__(self):
        self.profiles = EMOTION_PROFILES

    def generate(
        self,
        emotion: str,
        chord_progression: List[str],
        key: str = "C",
        bars: int = 4,
        tempo_bpm: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        config: Optional[StringConfig] = None,
        articulation_override: Optional[StringArticulation] = None,
        voicing_override: Optional[StringVoicing] = None,
        role_override: Optional[StringRole] = None,
    ) -> StringOutput:
        """Generate string arrangement for chord progression."""
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])

        articulation = articulation_override or profile["articulation"]
        voicing = voicing_override or profile["voicing"]
        role = role_override or profile["role"]
        dynamic = profile["dynamic"]

        beats_per_bar = time_signature[0]
        ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
        total_ticks = ticks_per_bar * bars
        ticks_per_chord = total_ticks // len(chord_progression)

        all_notes = []
        voices_dict = {}  # section -> list of notes

        current_tick = 0

        for chord_idx, chord_symbol in enumerate(chord_progression):
            root, quality, bass = parse_chord(chord_symbol)

            # Determine base octave based on preferred section
            section = profile["preferred_section"]
            range_low, range_high = get_section_range(section)
            base_octave = (range_low // 12) + 1

            pitches = get_chord_pitches(root, quality, base_octave)

            # Apply voicing
            voiced = apply_voicing_to_pitches(pitches, voicing, profile)

            # Calculate duration
            base_duration = ticks_per_chord
            dur_mod = get_articulation_duration_modifier(articulation)
            duration = int(base_duration * profile["sustain_ratio"] * dur_mod)

            # Generate notes for each voice
            vel_min, vel_max = profile["velocity_range"]

            for pitch, func, note_section in voiced:
                # Clamp to section range
                sec_low, sec_high = get_section_range(note_section)
                while pitch < sec_low:
                    pitch += 12
                while pitch > sec_high:
                    pitch -= 12

                # Velocity
                base_vel = random.randint(vel_min, vel_max)
                vel_curve = generate_velocity_curve(
                    base_vel, duration, dynamic, profile["expression_depth"]
                )

                # Humanize timing
                timing_offset = random.randint(
                    -profile["humanize_timing"], profile["humanize_timing"]
                )
                final_tick = max(0, current_tick + timing_offset)

                # Humanize velocity
                base_vel += random.randint(
                    -profile["humanize_velocity"], profile["humanize_velocity"]
                )
                base_vel = max(1, min(127, base_vel))

                note = StringNote(
                    pitch=pitch,
                    start_tick=final_tick,
                    duration_ticks=duration,
                    velocity=base_vel,
                    section=note_section,
                    articulation=articulation,
                    voice_function=func,
                    velocity_curve=vel_curve,
                )

                all_notes.append(note)

                if note_section not in voices_dict:
                    voices_dict[note_section] = []
                voices_dict[note_section].append(note)

            current_tick += ticks_per_chord

        # Build voice objects
        voices = []
        channel = 4  # Start at channel 4 for strings
        for section, notes in voices_dict.items():
            voice = StringVoice(
                section=section,
                notes=notes,
                gm_instrument=get_section_instrument(section),
                channel=channel,
            )
            voices.append(voice)
            channel += 1

        all_notes.sort(key=lambda n: n.start_tick)

        return StringOutput(
            voices=voices,
            notes=all_notes,
            emotion=emotion,
            articulation_used=articulation,
            dynamic_used=dynamic,
            voicing_used=voicing,
            role_used=role,
            gm_instrument=profile["gm_instrument"],
            total_ticks=total_ticks,
            metadata={
                "tempo_bpm": tempo_bpm,
                "time_signature": time_signature,
                "key": key,
                "chord_progression": chord_progression,
            },
        )

    def generate_for_section(
        self,
        emotion: str,
        chord_progression: List[str],
        section_type: str,
        key: str = "C",
        bars: int = 4,
        tempo_bpm: int = 120,
    ) -> StringOutput:
        """Generate strings adjusted for song section."""
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])

        if section_type == "intro":
            voicing = StringVoicing.SOLO
            articulation = StringArticulation.SUL_TASTO
        elif section_type == "verse":
            voicing = StringVoicing.DIVISI
            articulation = profile["articulation"]
        elif section_type == "pre_chorus":
            voicing = StringVoicing.DIVISI
            articulation = StringArticulation.TREMOLO
        elif section_type == "chorus":
            voicing = StringVoicing.FULL_SECTION
            articulation = profile["articulation"]
        elif section_type == "bridge":
            voicing = StringVoicing.OCTAVES
            articulation = StringArticulation.LEGATO
        elif section_type == "outro":
            voicing = StringVoicing.SOLO
            articulation = StringArticulation.HARMONIC
        else:
            voicing = profile["voicing"]
            articulation = profile["articulation"]

        return self.generate(
            emotion=emotion,
            chord_progression=chord_progression,
            key=key,
            bars=bars,
            tempo_bpm=tempo_bpm,
            voicing_override=voicing,
            articulation_override=articulation,
        )

    def get_available_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return list(self.profiles.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_grief_strings(chords: List[str], key: str = "F", bars: int = 4) -> StringOutput:
    """Quick grief strings generation."""
    return StringEngine().generate("grief", chords, key, bars, 72)


def generate_epic_strings(chords: List[str], key: str = "C", bars: int = 4) -> StringOutput:
    """Epic, powerful strings."""
    engine = StringEngine()
    return engine.generate(
        "euphoria",
        chords,
        key,
        bars,
        100,
        voicing_override=StringVoicing.FULL_SECTION,
    )


def strings_to_midi_events(string_output: StringOutput) -> List[Dict]:
    """Convert StringOutput to MIDI event list with multiple channels."""
    events = []

    for voice in string_output.voices:
        events.append(
            {
                "type": "program_change",
                "tick": 0,
                "channel": voice.channel,
                "program": voice.gm_instrument,
            }
        )

        for note in voice.notes:
            events.append(
                {
                    "type": "note_on",
                    "tick": note.start_tick,
                    "channel": voice.channel,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                }
            )
            events.append(
                {
                    "type": "note_off",
                    "tick": note.start_tick + note.duration_ticks,
                    "channel": voice.channel,
                    "pitch": note.pitch,
                    "velocity": 0,
                }
            )

    events.sort(key=lambda e: (e["tick"], 0 if e["type"] == "note_on" else 1))
    return events


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    engine = StringEngine()

    print("=== STRING ENGINE DEMO ===\n")

    chords = ["F", "C", "Dm", "Bbm"]

    emotions = ["grief", "hope", "fear", "joy", "peace"]

    for emotion in emotions:
        strings = engine.generate(
            emotion=emotion,
            chord_progression=chords,
            key="F",
            bars=4,
            tempo_bpm=82,
        )

        print(f"--- {emotion.upper()} ---")
        print(f"Articulation: {strings.articulation_used.value}")
        print(f"Dynamic: {strings.dynamic_used.value}")
        print(f"Voicing: {strings.voicing_used.value}")
        print(f"Role: {strings.role_used.value}")
        print(f"Voices: {len(strings.voices)}")
        print(f"Total notes: {len(strings.notes)}")
        sections = set(n.section.value for n in strings.notes)
        print(f"Sections: {', '.join(sections)}")
        print()
