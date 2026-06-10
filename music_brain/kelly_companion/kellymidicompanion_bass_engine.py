"""
Bass Engine - Generates emotion-driven bass lines.

Part of Kelly MIDI Companion Tier 1 MVP.

Philosophy: Bass isn't just chord roots. A grief bass line breathes
and sighs. A defiant bass line pushes against the beat. An anxious
bass line never settles. The bass IS the emotional foundation.

Integration:
    from kellymidicompanion_bass_engine import BassEngine, BassConfig

    engine = BassEngine()
    bassline = engine.generate(
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

# =============================================================================
# CONSTANTS
# =============================================================================

CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

TICKS_PER_BEAT = 480

# Chord quality to intervals
CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 14],
}


# =============================================================================
# ENUMS
# =============================================================================


class BassPattern(Enum):
    """Bass line pattern archetypes."""

    ROOT_ONLY = "root_only"  # Just roots on beat 1
    ROOT_FIFTH = "root_fifth"  # Root and 5th
    WALKING = "walking"  # Chromatic/stepwise motion
    PEDAL = "pedal"  # Sustained single note
    ARPEGGIATED = "arpeggiated"  # Chord tones broken up
    SYNCOPATED = "syncopated"  # Off-beat emphasis
    DRIVING = "driving"  # Steady 8ths
    PULSING = "pulsing"  # Repeated note patterns
    BREATHING = "breathing"  # Long notes with rests
    DESCENDING = "descending"  # Chromatic descent
    CLIMBING = "climbing"  # Ascending motion
    GHOST = "ghost"  # Sparse, barely there


class BassArticulation(Enum):
    """Bass note articulation."""

    SUSTAINED = "sustained"  # Full length, legato
    STACCATO = "staccato"  # Short, punchy
    MUTED = "muted"  # Dampened
    SLIDE = "slide"  # Glide between notes
    DEAD = "dead"  # Ghost notes, no pitch


class BassRegister(Enum):
    """Bass register/octave preference."""

    SUB = "sub"  # Very low (octave 1)
    LOW = "low"  # Standard bass (octave 2)
    MID = "mid"  # Higher bass (octave 3)
    FLEXIBLE = "flexible"  # Moves between registers


# =============================================================================
# EMOTION TO BASS MAPPING
# =============================================================================

EMOTION_BASS_PROFILES = {
    "grief": {
        "patterns": [BassPattern.BREATHING, BassPattern.PEDAL, BassPattern.DESCENDING],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.LOW,
        "note_density": 0.3,  # Notes per beat average
        "syncopation": 0.1,
        "chromatic_tendency": 0.2,
        "approach_notes": True,  # Use leading tones
        "velocity_range": (40, 70),
        "swing": 0.0,
        "push_pull": -0.05,  # Slightly behind beat
        "fifth_probability": 0.3,
        "octave_jump_prob": 0.1,
        "rest_probability": 0.3,
    },
    "hope": {
        "patterns": [BassPattern.ROOT_FIFTH, BassPattern.CLIMBING, BassPattern.ARPEGGIATED],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.MID,
        "note_density": 0.5,
        "syncopation": 0.2,
        "chromatic_tendency": 0.1,
        "approach_notes": True,
        "velocity_range": (55, 85),
        "swing": 0.1,
        "push_pull": 0.02,  # Slightly ahead
        "fifth_probability": 0.5,
        "octave_jump_prob": 0.2,
        "rest_probability": 0.15,
    },
    "rage": {
        "patterns": [BassPattern.DRIVING, BassPattern.SYNCOPATED, BassPattern.PULSING],
        "articulation": BassArticulation.STACCATO,
        "register": BassRegister.LOW,
        "note_density": 1.0,  # Busy
        "syncopation": 0.4,
        "chromatic_tendency": 0.3,
        "approach_notes": False,
        "velocity_range": (90, 127),
        "swing": 0.0,
        "push_pull": 0.03,  # Pushing hard
        "fifth_probability": 0.3,
        "octave_jump_prob": 0.3,
        "rest_probability": 0.05,
    },
    "fear": {
        "patterns": [BassPattern.GHOST, BassPattern.PEDAL, BassPattern.SYNCOPATED],
        "articulation": BassArticulation.MUTED,
        "register": BassRegister.SUB,
        "note_density": 0.4,
        "syncopation": 0.5,
        "chromatic_tendency": 0.4,
        "approach_notes": False,
        "velocity_range": (30, 80),
        "swing": 0.0,
        "push_pull": -0.02,
        "fifth_probability": 0.2,
        "octave_jump_prob": 0.4,  # Startling jumps
        "rest_probability": 0.4,
    },
    "joy": {
        "patterns": [BassPattern.WALKING, BassPattern.ARPEGGIATED, BassPattern.DRIVING],
        "articulation": BassArticulation.STACCATO,
        "register": BassRegister.MID,
        "note_density": 0.75,
        "syncopation": 0.3,
        "chromatic_tendency": 0.15,
        "approach_notes": True,
        "velocity_range": (70, 100),
        "swing": 0.2,
        "push_pull": 0.01,
        "fifth_probability": 0.5,
        "octave_jump_prob": 0.2,
        "rest_probability": 0.1,
    },
    "longing": {
        "patterns": [BassPattern.BREATHING, BassPattern.CLIMBING, BassPattern.PEDAL],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.LOW,
        "note_density": 0.35,
        "syncopation": 0.15,
        "chromatic_tendency": 0.25,
        "approach_notes": True,
        "velocity_range": (45, 75),
        "swing": 0.05,
        "push_pull": -0.03,  # Behind, yearning
        "fifth_probability": 0.4,
        "octave_jump_prob": 0.25,
        "rest_probability": 0.25,
    },
    "anxiety": {
        "patterns": [BassPattern.PULSING, BassPattern.SYNCOPATED, BassPattern.DRIVING],
        "articulation": BassArticulation.STACCATO,
        "register": BassRegister.FLEXIBLE,
        "note_density": 0.9,
        "syncopation": 0.6,
        "chromatic_tendency": 0.35,
        "approach_notes": False,
        "velocity_range": (50, 95),
        "swing": 0.0,
        "push_pull": 0.02,
        "fifth_probability": 0.2,
        "octave_jump_prob": 0.3,
        "rest_probability": 0.1,
    },
    "tenderness": {
        "patterns": [BassPattern.ROOT_ONLY, BassPattern.BREATHING, BassPattern.PEDAL],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.LOW,
        "note_density": 0.25,
        "syncopation": 0.05,
        "chromatic_tendency": 0.1,
        "approach_notes": True,
        "velocity_range": (35, 60),
        "swing": 0.0,
        "push_pull": 0.0,
        "fifth_probability": 0.3,
        "octave_jump_prob": 0.05,
        "rest_probability": 0.35,
    },
    "defiance": {
        "patterns": [BassPattern.SYNCOPATED, BassPattern.DRIVING, BassPattern.CLIMBING],
        "articulation": BassArticulation.STACCATO,
        "register": BassRegister.MID,
        "note_density": 0.8,
        "syncopation": 0.5,
        "chromatic_tendency": 0.2,
        "approach_notes": True,
        "velocity_range": (75, 110),
        "swing": 0.0,
        "push_pull": 0.04,  # Pushing against the beat
        "fifth_probability": 0.4,
        "octave_jump_prob": 0.25,
        "rest_probability": 0.1,
    },
    "melancholy": {
        "patterns": [BassPattern.DESCENDING, BassPattern.BREATHING, BassPattern.WALKING],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.LOW,
        "note_density": 0.4,
        "syncopation": 0.1,
        "chromatic_tendency": 0.3,
        "approach_notes": True,
        "velocity_range": (40, 70),
        "swing": 0.05,
        "push_pull": -0.04,
        "fifth_probability": 0.35,
        "octave_jump_prob": 0.1,
        "rest_probability": 0.25,
    },
    "nostalgia": {
        "patterns": [BassPattern.ROOT_FIFTH, BassPattern.WALKING, BassPattern.ARPEGGIATED],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.LOW,
        "note_density": 0.5,
        "syncopation": 0.15,
        "chromatic_tendency": 0.15,
        "approach_notes": True,
        "velocity_range": (50, 80),
        "swing": 0.15,
        "push_pull": -0.02,
        "fifth_probability": 0.5,
        "octave_jump_prob": 0.15,
        "rest_probability": 0.2,
    },
    "euphoria": {
        "patterns": [BassPattern.ARPEGGIATED, BassPattern.CLIMBING, BassPattern.DRIVING],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.FLEXIBLE,
        "note_density": 0.7,
        "syncopation": 0.25,
        "chromatic_tendency": 0.1,
        "approach_notes": True,
        "velocity_range": (80, 115),
        "swing": 0.1,
        "push_pull": 0.02,
        "fifth_probability": 0.6,
        "octave_jump_prob": 0.3,
        "rest_probability": 0.05,
    },
    "dissociation": {
        "patterns": [BassPattern.GHOST, BassPattern.PEDAL, BassPattern.ROOT_ONLY],
        "articulation": BassArticulation.MUTED,
        "register": BassRegister.SUB,
        "note_density": 0.2,
        "syncopation": 0.3,
        "chromatic_tendency": 0.2,
        "approach_notes": False,
        "velocity_range": (25, 50),
        "swing": 0.0,
        "push_pull": 0.0,
        "fifth_probability": 0.2,
        "octave_jump_prob": 0.5,
        "rest_probability": 0.5,
    },
    "neutral": {
        "patterns": [BassPattern.ROOT_FIFTH, BassPattern.WALKING, BassPattern.ARPEGGIATED],
        "articulation": BassArticulation.SUSTAINED,
        "register": BassRegister.LOW,
        "note_density": 0.5,
        "syncopation": 0.2,
        "chromatic_tendency": 0.1,
        "approach_notes": True,
        "velocity_range": (60, 90),
        "swing": 0.0,
        "push_pull": 0.0,
        "fifth_probability": 0.4,
        "octave_jump_prob": 0.15,
        "rest_probability": 0.15,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BassNote:
    """A single bass note."""

    pitch: int  # MIDI pitch
    start_tick: int  # Start time in ticks
    duration_ticks: int  # Duration
    velocity: int  # MIDI velocity
    chord_function: str  # "root", "third", "fifth", "seventh", "approach", "chromatic"
    articulation: BassArticulation
    timing_offset: int  # Ticks offset from grid (push/pull)


@dataclass
class BassConfig:
    """Configuration for bass generation."""

    emotion: str = "neutral"
    chord_progression: List[str] = field(default_factory=lambda: ["C"])
    chord_durations: Optional[List[int]] = None  # Bars per chord, defaults to equal
    key: str = "C"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    seed: Optional[int] = None
    pattern_override: Optional[BassPattern] = None
    register_override: Optional[BassRegister] = None
    force_root_on_one: bool = True  # Always play root on beat 1


@dataclass
class BassOutput:
    """Output from bass generation."""

    notes: List[BassNote]
    config: BassConfig
    pattern_used: BassPattern
    register_used: BassRegister
    total_ticks: int

    def to_midi_events(self) -> List[Dict]:
        """Convert to MIDI event list."""
        events = []
        for note in self.notes:
            events.append(
                {
                    "type": "note_on",
                    "note": note.pitch,
                    "velocity": note.velocity,
                    "time": note.start_tick + note.timing_offset,
                }
            )
            events.append(
                {
                    "type": "note_off",
                    "note": note.pitch,
                    "velocity": 0,
                    "time": note.start_tick + note.timing_offset + note.duration_ticks,
                }
            )
        return sorted(events, key=lambda e: (e["time"], e["type"] == "note_off"))

    def to_note_list(self) -> List[Tuple[int, int, int, int]]:
        """Return as (pitch, start, duration, velocity)."""
        return [
            (n.pitch, n.start_tick + n.timing_offset, n.duration_ticks, n.velocity)
            for n in self.notes
        ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def note_name_to_midi(note: str, octave: int = 2) -> int:
    """Convert note name to MIDI pitch."""
    note = note.upper().strip()

    # Handle flats
    flat_map = {"DB": "C#", "EB": "D#", "FB": "E", "GB": "F#", "AB": "G#", "BB": "A#", "CB": "B"}

    # Extract note name (may have accidental)
    if len(note) >= 2 and note[1] in "#B":
        note_part = note[:2]
    else:
        note_part = note[0]

    if note_part in flat_map:
        note_part = flat_map[note_part]

    if note_part in CHROMATIC:
        return CHROMATIC.index(note_part) + (octave + 1) * 12

    return 36  # Default to C2


def parse_chord(chord_str: str) -> Tuple[str, str]:
    """
    Parse chord string into root and quality.

    Examples:
        "C" -> ("C", "maj")
        "Am" -> ("A", "min")
        "F#m7" -> ("F#", "min7")
        "Bbmaj7" -> ("Bb", "maj7")
    """
    chord_str = chord_str.strip()

    # Extract root
    if len(chord_str) >= 2 and chord_str[1] in "#b":
        root = chord_str[:2]
        remainder = chord_str[2:]
    else:
        root = chord_str[0]
        remainder = chord_str[1:]

    # Determine quality
    remainder_lower = remainder.lower()

    if remainder_lower.startswith("maj7"):
        quality = "maj7"
    elif remainder_lower.startswith("min7") or remainder_lower.startswith("m7"):
        quality = "min7"
    elif remainder_lower.startswith("dim7"):
        quality = "dim7"
    elif remainder_lower.startswith("dim") or remainder_lower == "°":
        quality = "dim"
    elif remainder_lower.startswith("aug") or remainder_lower == "+":
        quality = "aug"
    elif remainder_lower.startswith("sus2"):
        quality = "sus2"
    elif remainder_lower.startswith("sus4") or remainder_lower.startswith("sus"):
        quality = "sus4"
    elif remainder_lower.startswith("7"):
        quality = "7"
    elif remainder_lower.startswith("m") or remainder_lower.startswith("min"):
        quality = "min"
    elif remainder_lower.startswith("add9"):
        quality = "add9"
    else:
        quality = "maj"

    return root, quality


def get_chord_tones(root: str, quality: str, octave: int) -> Dict[str, int]:
    """Get chord tones as MIDI pitches with function labels."""
    root_midi = note_name_to_midi(root, octave)
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])

    functions = ["root", "third", "fifth", "seventh", "ninth"]
    tones = {}

    for i, interval in enumerate(intervals):
        func = functions[i] if i < len(functions) else f"ext{i}"
        tones[func] = root_midi + interval

    return tones


def get_register_octave(register: BassRegister) -> int:
    """Get base octave for register."""
    if register == BassRegister.SUB:
        return 1
    elif register == BassRegister.LOW:
        return 2
    elif register == BassRegister.MID:
        return 3
    else:  # FLEXIBLE
        return 2


# =============================================================================
# PATTERN GENERATORS
# =============================================================================


def generate_root_only(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Generate root-only pattern: (start_tick, duration, pitch, function)"""
    notes = []
    root = chord_tones["root"]

    # One root per bar, on beat 1
    duration = bar_ticks - TICKS_PER_BEAT // 2  # Slight gap at end
    notes.append((0, duration, root, "root"))

    return notes


def generate_root_fifth(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Root on 1, fifth on 3."""
    notes = []
    root = chord_tones["root"]
    fifth = chord_tones.get("fifth", root + 7)

    beat_ticks = TICKS_PER_BEAT

    # Root on beat 1
    notes.append((0, beat_ticks * 2 - 20, root, "root"))

    # Fifth on beat 3
    notes.append((beat_ticks * 2, beat_ticks * 2 - 20, fifth, "fifth"))

    return notes


def generate_walking(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Walking bass - stepwise motion through chord and approach tones."""
    notes = []
    root = chord_tones["root"]
    third = chord_tones.get("third", root + 4)
    fifth = chord_tones.get("fifth", root + 7)

    beat_ticks = TICKS_PER_BEAT
    beats = time_sig[0]

    # Standard walking pattern
    targets = [root, third, fifth, third][:beats]

    for i, target in enumerate(targets):
        start = i * beat_ticks
        duration = beat_ticks - 20

        # Add chromatic approach on last beat
        if i == beats - 1 and profile.get("approach_notes", True):
            # Approach note
            approach = target + (1 if random.random() > 0.5 else -1)
            notes.append((start, duration // 2, approach, "approach"))
            notes.append((start + duration // 2, duration // 2, target, "root"))
        else:
            func = "root" if target == root else ("fifth" if target == fifth else "third")
            notes.append((start, duration, target, func))

    return notes


def generate_pedal(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Sustained pedal tone."""
    root = chord_tones["root"]
    duration = bar_ticks - 40  # Full bar minus tiny gap
    return [(0, duration, root, "root")]


def generate_arpeggiated(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Arpeggiate through chord tones."""
    notes = []
    tones = list(chord_tones.items())

    eighth = TICKS_PER_BEAT // 2
    _beat_ticks = TICKS_PER_BEAT  # noqa: F841
    total_eighths = bar_ticks // eighth

    for i in range(total_eighths):
        func, pitch = tones[i % len(tones)]
        start = i * eighth
        duration = eighth - 10
        notes.append((start, duration, pitch, func))

    return notes


def generate_syncopated(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Off-beat emphasis."""
    notes = []
    root = chord_tones["root"]
    fifth = chord_tones.get("fifth", root + 7)

    eighth = TICKS_PER_BEAT // 2

    # Syncopated positions: and-of-1, beat-2, and-of-3, beat-4
    positions = [
        (eighth, root, "root"),  # And of 1
        (TICKS_PER_BEAT * 2, fifth, "fifth"),  # Beat 3
        (TICKS_PER_BEAT * 2 + eighth, root, "root"),  # And of 3
        (TICKS_PER_BEAT * 3 + eighth, fifth, "fifth"),  # And of 4
    ]

    for start, pitch, func in positions:
        if start < bar_ticks:
            notes.append((start, eighth - 10, pitch, func))

    return notes


def generate_driving(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Steady eighth notes."""
    notes = []
    root = chord_tones["root"]
    fifth = chord_tones.get("fifth", root + 7)

    eighth = TICKS_PER_BEAT // 2
    num_eighths = bar_ticks // eighth

    for i in range(num_eighths):
        start = i * eighth
        pitch = root if i % 4 in [0, 1] else fifth
        func = "root" if pitch == root else "fifth"
        notes.append((start, eighth - 10, pitch, func))

    return notes


def generate_pulsing(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Repeated note pattern."""
    notes = []
    root = chord_tones["root"]

    sixteenth = TICKS_PER_BEAT // 4
    num_sixteenths = bar_ticks // sixteenth

    # Pattern: note-note-rest-note repeated
    for i in range(num_sixteenths):
        if i % 4 != 2:  # Skip every 3rd in group of 4
            start = i * sixteenth
            notes.append((start, sixteenth - 5, root, "root"))

    return notes


def generate_breathing(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Long notes with natural pauses."""
    notes = []
    root = chord_tones["root"]
    fifth = chord_tones.get("fifth", root + 7)

    # Note, then rest, then note
    half_bar = bar_ticks // 2

    notes.append((0, half_bar - TICKS_PER_BEAT, root, "root"))
    # Gap for breath
    if random.random() > profile.get("rest_probability", 0.3):
        notes.append((half_bar + TICKS_PER_BEAT // 2, half_bar - TICKS_PER_BEAT, fifth, "fifth"))

    return notes


def generate_descending(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Chromatic descent."""
    notes = []
    root = chord_tones["root"]

    beat_ticks = TICKS_PER_BEAT
    beats = time_sig[0]

    current_pitch = root + 7  # Start from fifth
    for i in range(beats):
        start = i * beat_ticks
        notes.append((start, beat_ticks - 20, current_pitch, "chromatic"))
        current_pitch -= random.choice([1, 2])  # Chromatic or whole step down

    return notes


def generate_climbing(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Ascending motion."""
    notes = []
    root = chord_tones["root"]

    beat_ticks = TICKS_PER_BEAT
    beats = time_sig[0]

    current_pitch = root - 5  # Start below root
    for i in range(beats):
        start = i * beat_ticks
        notes.append((start, beat_ticks - 20, current_pitch, "chromatic"))
        current_pitch += random.choice([1, 2, 3])

    return notes


def generate_ghost(
    chord_tones: Dict[str, int], bar_ticks: int, profile: Dict, time_sig: Tuple[int, int]
) -> List[Tuple[int, int, int, str]]:
    """Sparse, barely-there notes."""
    notes = []
    root = chord_tones["root"]

    # Just one or two quiet notes
    if random.random() > 0.5:
        notes.append((0, TICKS_PER_BEAT // 2, root, "root"))

    if random.random() > 0.7:
        pos = random.randint(1, 3) * TICKS_PER_BEAT
        if pos < bar_ticks:
            notes.append((pos, TICKS_PER_BEAT // 2, root, "root"))

    return notes


PATTERN_GENERATORS = {
    BassPattern.ROOT_ONLY: generate_root_only,
    BassPattern.ROOT_FIFTH: generate_root_fifth,
    BassPattern.WALKING: generate_walking,
    BassPattern.PEDAL: generate_pedal,
    BassPattern.ARPEGGIATED: generate_arpeggiated,
    BassPattern.SYNCOPATED: generate_syncopated,
    BassPattern.DRIVING: generate_driving,
    BassPattern.PULSING: generate_pulsing,
    BassPattern.BREATHING: generate_breathing,
    BassPattern.DESCENDING: generate_descending,
    BassPattern.CLIMBING: generate_climbing,
    BassPattern.GHOST: generate_ghost,
}


# =============================================================================
# MAIN ENGINE
# =============================================================================


class BassEngine:
    """
    Generates emotion-driven bass lines.

    Usage:
        engine = BassEngine()
        bassline = engine.generate(
            emotion="grief",
            chord_progression=["F", "C", "Dm", "Bbm"],
            key="F",
            bars=4,
            tempo_bpm=82
        )
    """

    def __init__(self, custom_profiles: Optional[Dict] = None):
        self.profiles = EMOTION_BASS_PROFILES.copy()
        if custom_profiles:
            self.profiles.update(custom_profiles)

    def generate(
        self,
        emotion: str = "neutral",
        chord_progression: List[str] = None,
        key: str = "C",
        bars: int = 4,
        tempo_bpm: int = 120,
        **kwargs,
    ) -> BassOutput:
        """
        Generate bass line.

        Args:
            emotion: Emotional state
            chord_progression: List of chord symbols ["F", "C", "Dm", "Bbm"]
            key: Musical key
            bars: Number of bars
            tempo_bpm: Tempo
        """
        if chord_progression is None:
            chord_progression = [key]

        config = BassConfig(
            emotion=emotion.lower(),
            chord_progression=chord_progression,
            key=key,
            bars=bars,
            tempo_bpm=tempo_bpm,
            **kwargs,
        )

        return self._generate_from_config(config)

    def _generate_from_config(self, config: BassConfig) -> BassOutput:
        """Internal generation."""

        if config.seed is not None:
            random.seed(config.seed)

        profile = self.profiles.get(config.emotion, self.profiles["neutral"])

        # Select pattern
        pattern = config.pattern_override or random.choice(profile["patterns"])

        # Select register
        register = config.register_override or profile["register"]
        base_octave = get_register_octave(register)

        # Calculate timing
        beats_per_bar = config.time_signature[0]
        ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
        total_ticks = ticks_per_bar * config.bars

        # Distribute chords across bars
        num_chords = len(config.chord_progression)
        if config.chord_durations:
            chord_bars = config.chord_durations
        else:
            # Equal distribution
            bars_per_chord = config.bars / num_chords
            chord_bars = [bars_per_chord] * num_chords

        # Generate notes for each chord
        all_notes = []
        current_tick = 0

        for chord_idx, (chord_str, chord_dur) in enumerate(
            zip(config.chord_progression, chord_bars)
        ):
            root, quality = parse_chord(chord_str)
            chord_tones = get_chord_tones(root, quality, base_octave)

            chord_ticks = int(ticks_per_bar * chord_dur)

            # Get pattern generator
            generator = PATTERN_GENERATORS.get(pattern, generate_root_fifth)

            # Generate pattern for this chord
            pattern_notes = generator(chord_tones, chord_ticks, profile, config.time_signature)

            # Convert to BassNote objects
            for start, duration, pitch, func in pattern_notes:
                # Apply timing offset (push/pull)
                timing_offset = int(profile["push_pull"] * TICKS_PER_BEAT)

                # Apply syncopation variation
                if random.random() < profile["syncopation"]:
                    timing_offset += random.choice([-TICKS_PER_BEAT // 8, TICKS_PER_BEAT // 8])

                # Velocity
                vel_min, vel_max = profile["velocity_range"]
                velocity = random.randint(vel_min, vel_max)

                # Accent on beat 1
                if start == 0:
                    velocity = min(127, velocity + 15)

                # Apply articulation to duration
                articulation = profile["articulation"]
                if articulation == BassArticulation.STACCATO:
                    duration = max(TICKS_PER_BEAT // 8, int(duration * 0.5))
                elif articulation == BassArticulation.MUTED:
                    duration = max(TICKS_PER_BEAT // 4, int(duration * 0.6))
                    velocity = max(20, velocity - 20)

                # Octave jumps
                if random.random() < profile["octave_jump_prob"]:
                    pitch += random.choice([-12, 12])
                    pitch = max(24, min(72, pitch))  # Clamp to bass range

                # Rest probability
                if random.random() < profile["rest_probability"] and start > 0:
                    continue  # Skip this note

                note = BassNote(
                    pitch=pitch,
                    start_tick=current_tick + start,
                    duration_ticks=duration,
                    velocity=velocity,
                    chord_function=func,
                    articulation=articulation,
                    timing_offset=timing_offset,
                )

                all_notes.append(note)

            current_tick += chord_ticks

        # Force root on beat 1 if configured
        if config.force_root_on_one and all_notes:
            first_chord = config.chord_progression[0]
            root, quality = parse_chord(first_chord)
            root_pitch = note_name_to_midi(root, base_octave)

            # Check if first note is root on beat 1
            if all_notes[0].start_tick != 0 or all_notes[0].pitch != root_pitch:
                # Insert root
                vel_min, vel_max = profile["velocity_range"]
                root_note = BassNote(
                    pitch=root_pitch,
                    start_tick=0,
                    duration_ticks=TICKS_PER_BEAT,
                    velocity=min(127, vel_max + 10),
                    chord_function="root",
                    articulation=profile["articulation"],
                    timing_offset=0,
                )
                all_notes.insert(0, root_note)

        return BassOutput(
            notes=all_notes,
            config=config,
            pattern_used=pattern,
            register_used=register,
            total_ticks=total_ticks,
        )

    def generate_for_section(
        self,
        emotion: str,
        chord_progression: List[str],
        section_type: str = "verse",  # "verse", "chorus", "bridge"
        key: str = "C",
        bars: int = 8,
        tempo_bpm: int = 120,
    ) -> BassOutput:
        """
        Generate bass appropriate for a song section.

        Sections have different energy levels:
        - verse: more sparse, breathing
        - chorus: more driving, full
        - bridge: transitional, building
        """
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])

        if section_type == "verse":
            patterns = [BassPattern.ROOT_FIFTH, BassPattern.BREATHING, BassPattern.WALKING]
        elif section_type == "chorus":
            patterns = [BassPattern.DRIVING, BassPattern.ARPEGGIATED, BassPattern.ROOT_FIFTH]
        elif section_type == "bridge":
            patterns = [BassPattern.PEDAL, BassPattern.CLIMBING, BassPattern.DESCENDING]
        else:
            patterns = profile["patterns"]

        return self.generate(
            emotion=emotion,
            chord_progression=chord_progression,
            key=key,
            bars=bars,
            tempo_bpm=tempo_bpm,
            pattern_override=random.choice(patterns),
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_grief_bass(chords: List[str], key: str = "F", bars: int = 4) -> BassOutput:
    """Quick grief bass generation."""
    return BassEngine().generate("grief", chords, key, bars, 72)


def generate_driving_bass(chords: List[str], key: str = "E", bars: int = 4) -> BassOutput:
    """Quick driving bass for energy."""
    engine = BassEngine()
    return engine.generate("rage", chords, key, bars, 140, pattern_override=BassPattern.DRIVING)


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    engine = BassEngine()

    print("=== BASS ENGINE DEMO ===\n")

    chords = ["F", "C", "Dm", "Bbm"]

    emotions = ["grief", "hope", "rage", "anxiety"]

    for emotion in emotions:
        bass = engine.generate(
            emotion=emotion,
            chord_progression=chords,
            key="F",
            bars=4,
            tempo_bpm=82,
        )

        print(f"--- {emotion.upper()} ---")
        print(f"Pattern: {bass.pattern_used.value}")
        print(f"Register: {bass.register_used.value}")
        print(f"Notes: {len(bass.notes)}")
        print(f"First 4 pitches: {[n.pitch for n in bass.notes[:4]]}")
        print()
