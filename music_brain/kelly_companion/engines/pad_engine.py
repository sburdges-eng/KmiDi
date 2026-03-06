"""
Pad Engine - Generates emotion-driven sustained textural backgrounds.

Part of Kelly MIDI Companion Tier 2.

Philosophy: Pads are the emotional atmosphere. They don't demand attention
but they define the emotional space. A grief pad breathes slowly and warmly.
A rage pad grinds and distorts. An anxiety pad shifts restlessly, never settling.
The pad IS the emotional air the song breathes.

Integration:
    from kellymidicompanion_pad_engine import PadEngine, PadConfig

    engine = PadEngine()
    pad = engine.generate(
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

CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

TICKS_PER_BEAT = 480

# Chord quality to intervals
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
    "9": [0, 4, 7, 10, 14],
    "6": [0, 4, 7, 9],
}

# GM Pad instruments (programs 88-95)
GM_PADS = {
    "new_age": 88,
    "warm": 89,
    "polysynth": 90,
    "choir": 91,
    "bowed": 92,
    "metallic": 93,
    "halo": 94,
    "sweep": 95,
}

# Extended pad-like instruments
EXTENDED_PADS = {
    "strings": 48,      # String Ensemble 1
    "slow_strings": 49,  # String Ensemble 2
    "synth_strings": 50,
    "choir_aahs": 52,
    "voice_oohs": 53,
    "synth_voice": 54,
    "orchestra_hit": 55,
}


# =============================================================================
# ENUMS
# =============================================================================

class PadTexture(Enum):
    """Pad timbral character."""
    WARM = "warm"               # Soft, rounded, comforting
    BRIGHT = "bright"           # Airy, open, hopeful
    DARK = "dark"               # Heavy, somber, thick
    GLASSY = "glassy"           # Crystalline, fragile, distant
    GRITTY = "gritty"           # Distorted, aggressive, textured
    AIRY = "airy"               # Thin, ethereal, spacious
    THICK = "thick"             # Dense, full, enveloping
    HOLLOW = "hollow"           # Empty, haunted, sparse
    SHIMMERING = "shimmering"   # Movement, modulation, alive
    FROZEN = "frozen"           # Static, cold, still


class PadMovement(Enum):
    """How the pad evolves over time."""
    STATIC = "static"           # No movement, sustained
    BREATHING = "breathing"     # Slow swell in/out
    SWELLING = "swelling"       # Building intensity
    FADING = "fading"           # Dying away
    PULSING = "pulsing"         # Rhythmic undulation
    DRIFTING = "drifting"       # Slow pitch/filter movement
    TREMOLO = "tremolo"         # Fast amplitude modulation
    EVOLVING = "evolving"       # Continuous transformation
    UNSTABLE = "unstable"       # Erratic, nervous movement


class PadVoicing(Enum):
    """How pad notes are spread."""
    CLOSE = "close"             # Tight cluster
    OPEN = "open"               # Spread voicing
    OCTAVES = "octaves"         # Root + octave doubling
    FIFTHS = "fifths"           # Power chord style
    FULL = "full"               # All chord tones
    SPARSE = "sparse"           # Just root + one other
    CLUSTER = "cluster"         # Dissonant close voicing
    SHELL = "shell"             # Root + 3rd + 7th only


class PadRegister(Enum):
    """Pitch range for pad."""
    SUB = "sub"                 # Very low, felt more than heard
    LOW = "low"                 # Foundation
    MID = "mid"                 # Standard pad range
    HIGH = "high"               # Bright, airy
    WIDE = "wide"               # Spanning multiple octaves


class AttackType(Enum):
    """How the pad begins."""
    INSTANT = "instant"         # Immediate full volume
    SOFT = "soft"               # Gentle fade in
    SLOW = "slow"               # Very gradual entry
    SWELL = "swell"             # Dramatic build
    STAB = "stab"               # Sharp attack, quick decay


class ReleaseType(Enum):
    """How the pad ends."""
    SUSTAIN = "sustain"         # Hold until next chord
    FADE = "fade"               # Gentle release
    CUT = "cut"                 # Abrupt stop
    TRAIL = "trail"             # Long reverb tail (implied)
    OVERLAP = "overlap"         # Bleed into next chord


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_PROFILES = {
    "grief": {
        "texture": PadTexture.WARM,
        "movement": PadMovement.BREATHING,
        "voicing": PadVoicing.OPEN,
        "register": PadRegister.MID,
        "attack": AttackType.SLOW,
        "release": ReleaseType.TRAIL,
        "velocity_range": (40, 70),
        "density": 0.7,              # How full the voicing
        "sustain_ratio": 0.95,       # How much of bar to sustain
        "breathing_rate": 0.5,       # Slow breath
        "add_extensions": True,      # Add 7ths, 9ths
        "extension_probability": 0.4,
        "octave_doubling": 0.3,
        "gm_instrument": GM_PADS["warm"],
    },

    "sadness": {
        "texture": PadTexture.HOLLOW,
        "movement": PadMovement.FADING,
        "voicing": PadVoicing.SPARSE,
        "register": PadRegister.MID,
        "attack": AttackType.SOFT,
        "release": ReleaseType.FADE,
        "velocity_range": (35, 60),
        "density": 0.5,
        "sustain_ratio": 0.85,
        "breathing_rate": 0.3,
        "add_extensions": True,
        "extension_probability": 0.5,
        "octave_doubling": 0.2,
        "gm_instrument": GM_PADS["choir"],
    },

    "hope": {
        "texture": PadTexture.BRIGHT,
        "movement": PadMovement.SWELLING,
        "voicing": PadVoicing.OPEN,
        "register": PadRegister.HIGH,
        "attack": AttackType.SOFT,
        "release": ReleaseType.SUSTAIN,
        "velocity_range": (50, 85),
        "density": 0.6,
        "sustain_ratio": 0.9,
        "breathing_rate": 0.6,
        "add_extensions": True,
        "extension_probability": 0.3,
        "octave_doubling": 0.4,
        "gm_instrument": GM_PADS["new_age"],
    },

    "joy": {
        "texture": PadTexture.SHIMMERING,
        "movement": PadMovement.PULSING,
        "voicing": PadVoicing.FULL,
        "register": PadRegister.HIGH,
        "attack": AttackType.SOFT,
        "release": ReleaseType.SUSTAIN,
        "velocity_range": (60, 95),
        "density": 0.8,
        "sustain_ratio": 0.95,
        "breathing_rate": 0.8,
        "add_extensions": False,
        "extension_probability": 0.2,
        "octave_doubling": 0.5,
        "gm_instrument": GM_PADS["polysynth"],
    },

    "rage": {
        "texture": PadTexture.GRITTY,
        "movement": PadMovement.STATIC,
        "voicing": PadVoicing.FIFTHS,
        "register": PadRegister.LOW,
        "attack": AttackType.INSTANT,
        "release": ReleaseType.CUT,
        "velocity_range": (90, 127),
        "density": 0.9,
        "sustain_ratio": 0.98,
        "breathing_rate": 0.0,
        "add_extensions": False,
        "extension_probability": 0.1,
        "octave_doubling": 0.7,
        "gm_instrument": GM_PADS["metallic"],
    },

    "anger": {
        "texture": PadTexture.DARK,
        "movement": PadMovement.PULSING,
        "voicing": PadVoicing.CLOSE,
        "register": PadRegister.LOW,
        "attack": AttackType.INSTANT,
        "release": ReleaseType.SUSTAIN,
        "velocity_range": (80, 115),
        "density": 0.85,
        "sustain_ratio": 0.9,
        "breathing_rate": 0.2,
        "add_extensions": False,
        "extension_probability": 0.15,
        "octave_doubling": 0.6,
        "gm_instrument": GM_PADS["bowed"],
    },

    "fear": {
        "texture": PadTexture.HOLLOW,
        "movement": PadMovement.UNSTABLE,
        "voicing": PadVoicing.CLUSTER,
        "register": PadRegister.MID,
        "attack": AttackType.SOFT,
        "release": ReleaseType.TRAIL,
        "velocity_range": (30, 65),
        "density": 0.4,
        "sustain_ratio": 0.7,
        "breathing_rate": 0.9,
        "add_extensions": True,
        "extension_probability": 0.6,
        "octave_doubling": 0.1,
        "gm_instrument": GM_PADS["sweep"],
    },

    "anxiety": {
        "texture": PadTexture.GLASSY,
        "movement": PadMovement.UNSTABLE,
        "voicing": PadVoicing.SPARSE,
        "register": PadRegister.HIGH,
        "attack": AttackType.SOFT,
        "release": ReleaseType.FADE,
        "velocity_range": (40, 75),
        "density": 0.5,
        "sustain_ratio": 0.6,
        "breathing_rate": 1.2,
        "add_extensions": True,
        "extension_probability": 0.5,
        "octave_doubling": 0.2,
        "gm_instrument": GM_PADS["halo"],
    },

    "tension": {
        "texture": PadTexture.DARK,
        "movement": PadMovement.SWELLING,
        "voicing": PadVoicing.CLUSTER,
        "register": PadRegister.MID,
        "attack": AttackType.SLOW,
        "release": ReleaseType.SUSTAIN,
        "velocity_range": (50, 90),
        "density": 0.7,
        "sustain_ratio": 0.95,
        "breathing_rate": 0.4,
        "add_extensions": True,
        "extension_probability": 0.7,
        "octave_doubling": 0.3,
        "gm_instrument": GM_PADS["bowed"],
    },

    "longing": {
        "texture": PadTexture.WARM,
        "movement": PadMovement.BREATHING,
        "voicing": PadVoicing.OPEN,
        "register": PadRegister.MID,
        "attack": AttackType.SLOW,
        "release": ReleaseType.TRAIL,
        "velocity_range": (45, 75),
        "density": 0.6,
        "sustain_ratio": 0.9,
        "breathing_rate": 0.4,
        "add_extensions": True,
        "extension_probability": 0.6,
        "octave_doubling": 0.25,
        "gm_instrument": GM_PADS["choir"],
    },

    "nostalgia": {
        "texture": PadTexture.WARM,
        "movement": PadMovement.DRIFTING,
        "voicing": PadVoicing.OPEN,
        "register": PadRegister.MID,
        "attack": AttackType.SOFT,
        "release": ReleaseType.OVERLAP,
        "velocity_range": (40, 70),
        "density": 0.65,
        "sustain_ratio": 1.1,  # Overlap into next
        "breathing_rate": 0.35,
        "add_extensions": True,
        "extension_probability": 0.5,
        "octave_doubling": 0.3,
        "gm_instrument": GM_PADS["warm"],
    },

    "peace": {
        "texture": PadTexture.AIRY,
        "movement": PadMovement.STATIC,
        "voicing": PadVoicing.OPEN,
        "register": PadRegister.HIGH,
        "attack": AttackType.SLOW,
        "release": ReleaseType.TRAIL,
        "velocity_range": (30, 55),
        "density": 0.5,
        "sustain_ratio": 1.0,
        "breathing_rate": 0.2,
        "add_extensions": True,
        "extension_probability": 0.4,
        "octave_doubling": 0.2,
        "gm_instrument": GM_PADS["new_age"],
    },

    "euphoria": {
        "texture": PadTexture.SHIMMERING,
        "movement": PadMovement.SWELLING,
        "voicing": PadVoicing.FULL,
        "register": PadRegister.WIDE,
        "attack": AttackType.SOFT,
        "release": ReleaseType.SUSTAIN,
        "velocity_range": (70, 110),
        "density": 0.9,
        "sustain_ratio": 1.0,
        "breathing_rate": 0.7,
        "add_extensions": True,
        "extension_probability": 0.5,
        "octave_doubling": 0.6,
        "gm_instrument": GM_PADS["polysynth"],
    },

    "dissociation": {
        "texture": PadTexture.FROZEN,
        "movement": PadMovement.STATIC,
        "voicing": PadVoicing.SPARSE,
        "register": PadRegister.HIGH,
        "attack": AttackType.SLOW,
        "release": ReleaseType.TRAIL,
        "velocity_range": (20, 45),
        "density": 0.3,
        "sustain_ratio": 1.2,
        "breathing_rate": 0.0,
        "add_extensions": False,
        "extension_probability": 0.2,
        "octave_doubling": 0.5,
        "gm_instrument": GM_PADS["halo"],
    },

    "dread": {
        "texture": PadTexture.DARK,
        "movement": PadMovement.EVOLVING,
        "voicing": PadVoicing.CLUSTER,
        "register": PadRegister.SUB,
        "attack": AttackType.SLOW,
        "release": ReleaseType.SUSTAIN,
        "velocity_range": (35, 70),
        "density": 0.6,
        "sustain_ratio": 1.0,
        "breathing_rate": 0.3,
        "add_extensions": True,
        "extension_probability": 0.8,
        "octave_doubling": 0.4,
        "gm_instrument": GM_PADS["sweep"],
    },

    "wonder": {
        "texture": PadTexture.GLASSY,
        "movement": PadMovement.BREATHING,
        "voicing": PadVoicing.OPEN,
        "register": PadRegister.HIGH,
        "attack": AttackType.SOFT,
        "release": ReleaseType.TRAIL,
        "velocity_range": (45, 75),
        "density": 0.55,
        "sustain_ratio": 0.9,
        "breathing_rate": 0.5,
        "add_extensions": True,
        "extension_probability": 0.6,
        "octave_doubling": 0.35,
        "gm_instrument": GM_PADS["halo"],
    },

    "neutral": {
        "texture": PadTexture.WARM,
        "movement": PadMovement.STATIC,
        "voicing": PadVoicing.OPEN,
        "register": PadRegister.MID,
        "attack": AttackType.SOFT,
        "release": ReleaseType.SUSTAIN,
        "velocity_range": (50, 75),
        "density": 0.6,
        "sustain_ratio": 0.9,
        "breathing_rate": 0.3,
        "add_extensions": False,
        "extension_probability": 0.2,
        "octave_doubling": 0.3,
        "gm_instrument": GM_PADS["warm"],
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PadNote:
    """A single note in the pad voicing."""
    pitch: int                  # MIDI pitch
    start_tick: int             # Start time in ticks
    duration_ticks: int         # Duration
    velocity: int               # MIDI velocity
    voice_function: str         # "root", "third", "fifth", "seventh", "extension", "octave"
    velocity_curve: List[int] = field(default_factory=list)  # For breathing/swell


@dataclass
class PadChord:
    """A complete pad voicing for one chord."""
    chord_symbol: str
    notes: List[PadNote]
    start_tick: int
    duration_ticks: int
    texture: PadTexture
    movement: PadMovement


@dataclass
class PadConfig:
    """Configuration for pad generation."""
    emotion: str = "neutral"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)

    # Override options
    texture_override: Optional[PadTexture] = None
    movement_override: Optional[PadMovement] = None
    voicing_override: Optional[PadVoicing] = None
    register_override: Optional[PadRegister] = None

    # Fine-tuning
    velocity_scale: float = 1.0
    density_scale: float = 1.0
    sustain_scale: float = 1.0


@dataclass
class PadOutput:
    """Complete pad output."""
    chords: List[PadChord]
    notes: List[PadNote]         # Flattened for MIDI export
    emotion: str
    texture_used: PadTexture
    movement_used: PadMovement
    voicing_used: PadVoicing
    register_used: PadRegister
    gm_instrument: int
    total_ticks: int
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def note_name_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.replace('b', '#')  # Normalize flats

    # Handle double sharps/flats
    base = note[0].upper()
    modifier = note[1:] if len(note) > 1 else ''

    base_idx = CHROMATIC.index(base) if base in CHROMATIC else 0

    if modifier == '#':
        base_idx += 1
    elif modifier == 'b':
        base_idx -= 1
    elif modifier == '##':
        base_idx += 2
    elif modifier == 'bb':
        base_idx -= 2

    base_idx = base_idx % 12
    return base_idx + (octave + 1) * 12


def parse_chord(chord_symbol: str) -> Tuple[str, str, Optional[str]]:
    """Parse chord symbol into root, quality, bass note."""
    chord_symbol = chord_symbol.strip()

    # Handle slash chords
    bass_note = None
    if '/' in chord_symbol:
        chord_symbol, bass_note = chord_symbol.split('/')

    # Extract root
    if len(chord_symbol) > 1 and chord_symbol[1] in ['#', 'b']:
        root = chord_symbol[:2]
        quality = chord_symbol[2:] or "maj"
    else:
        root = chord_symbol[0]
        quality = chord_symbol[1:] or "maj"

    # Normalize quality
    quality = quality.lower()
    if quality == '':
        quality = 'maj'
    elif quality == 'm':
        quality = 'min'
    elif quality == '-':
        quality = 'min'

    return root, quality, bass_note


def get_chord_pitches(root: str, quality: str, base_octave: int = 4) -> List[int]:
    """Get MIDI pitches for a chord."""
    root_midi = note_name_to_midi(root, base_octave)
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])
    return [root_midi + i for i in intervals]


def get_register_octave(register: PadRegister) -> int:
    """Get base octave for register."""
    mapping = {
        PadRegister.SUB: 2,
        PadRegister.LOW: 3,
        PadRegister.MID: 4,
        PadRegister.HIGH: 5,
        PadRegister.WIDE: 3,  # Starts low, extends high
    }
    return mapping.get(register, 4)


def apply_voicing(
    pitches: List[int],
    voicing: PadVoicing,
    register: PadRegister,
    density: float
) -> List[Tuple[int, str]]:
    """Apply voicing style to chord pitches. Returns (pitch, function) tuples."""
    if not pitches:
        return []

    root = pitches[0]
    result = []

    if voicing == PadVoicing.SPARSE:
        # Just root and one other
        result.append((root, "root"))
        if len(pitches) > 2 and random.random() < density:
            result.append((pitches[2], "fifth"))
        elif len(pitches) > 1:
            result.append((pitches[1], "third"))

    elif voicing == PadVoicing.FIFTHS:
        # Power chord style
        result.append((root, "root"))
        fifth = root + 7
        result.append((fifth, "fifth"))
        if random.random() < density * 0.5:
            result.append((root + 12, "octave"))

    elif voicing == PadVoicing.OCTAVES:
        # Root with octave doublings
        result.append((root, "root"))
        result.append((root + 12, "octave"))
        if random.random() < density:
            result.append((root - 12, "sub_octave"))

    elif voicing == PadVoicing.CLOSE:
        # All notes within one octave
        for i, p in enumerate(pitches):
            while p > root + 12:
                p -= 12
            func = ["root", "third", "fifth", "seventh", "extension"][min(i, 4)]
            result.append((p, func))

    elif voicing == PadVoicing.OPEN:
        # Spread voicing
        result.append((root, "root"))
        for i, p in enumerate(pitches[1:], 1):
            spread_p = p + (12 if i % 2 == 0 else 0)
            func = ["root", "third", "fifth", "seventh", "extension"][min(i, 4)]
            if random.random() < density:
                result.append((spread_p, func))

    elif voicing == PadVoicing.FULL:
        # All chord tones
        for i, p in enumerate(pitches):
            func = ["root", "third", "fifth", "seventh", "extension"][min(i, 4)]
            if i == 0 or random.random() < density:
                result.append((p, func))

    elif voicing == PadVoicing.CLUSTER:
        # Tight, potentially dissonant
        result.append((root, "root"))
        for i, p in enumerate(pitches[1:], 1):
            # Keep everything close
            while p > root + 6:
                p -= 12
            while p < root - 6:
                p += 12
            func = ["root", "third", "fifth", "seventh", "extension"][min(i, 4)]
            if random.random() < density:
                result.append((p, func))

    elif voicing == PadVoicing.SHELL:
        # Root, 3rd, 7th only
        result.append((root, "root"))
        if len(pitches) > 1:
            result.append((pitches[1], "third"))
        if len(pitches) > 3:
            result.append((pitches[3], "seventh"))
        elif len(pitches) > 2:
            result.append((pitches[2], "fifth"))

    # Apply register adjustment
    base_oct = get_register_octave(register)
    target_midi = base_oct * 12 + 60  # Rough center

    adjusted = []
    for pitch, func in result:
        # Shift to target register
        while pitch < target_midi - 18:
            pitch += 12
        while pitch > target_midi + 18:
            pitch -= 12
        adjusted.append((pitch, func))

    # For WIDE register, add octave spread
    if register == PadRegister.WIDE and result:
        root_pitch = adjusted[0][0]
        adjusted.append((root_pitch - 12, "sub_octave"))
        adjusted.append((root_pitch + 12, "high_octave"))

    return adjusted


def generate_velocity_curve(
    base_velocity: int,
    duration_ticks: int,
    movement: PadMovement,
    breathing_rate: float
) -> List[int]:
    """Generate velocity curve for a pad note."""
    # Number of CC points (roughly one per beat)
    num_points = max(4, duration_ticks // TICKS_PER_BEAT)

    if movement == PadMovement.STATIC:
        return [base_velocity] * num_points

    elif movement == PadMovement.BREATHING:
        # Sine wave breathing
        curve = []
        for i in range(num_points):
            phase = (i / num_points) * math.pi * 2 * breathing_rate
            mod = math.sin(phase) * 0.2  # ±20% variation
            v = int(base_velocity * (1 + mod))
            curve.append(max(1, min(127, v)))
        return curve

    elif movement == PadMovement.SWELLING:
        # Gradual increase
        curve = []
        for i in range(num_points):
            progress = i / num_points
            v = int(base_velocity * (0.6 + 0.4 * progress))
            curve.append(max(1, min(127, v)))
        return curve

    elif movement == PadMovement.FADING:
        # Gradual decrease
        curve = []
        for i in range(num_points):
            progress = i / num_points
            v = int(base_velocity * (1.0 - 0.4 * progress))
            curve.append(max(1, min(127, v)))
        return curve

    elif movement == PadMovement.PULSING:
        # Rhythmic pulses
        curve = []
        for i in range(num_points):
            pulse = (i % 2 == 0)
            v = base_velocity if pulse else int(base_velocity * 0.7)
            curve.append(max(1, min(127, v)))
        return curve

    elif movement == PadMovement.DRIFTING:
        # Slow random drift
        curve = []
        current = base_velocity
        for i in range(num_points):
            drift = random.uniform(-5, 5)
            current = max(base_velocity * 0.7, min(base_velocity * 1.2, current + drift))
            curve.append(max(1, min(127, int(current))))
        return curve

    elif movement == PadMovement.TREMOLO:
        # Fast amplitude modulation
        curve = []
        for i in range(num_points):
            phase = (i / num_points) * math.pi * 8  # Fast oscillation
            mod = math.sin(phase) * 0.15
            v = int(base_velocity * (1 + mod))
            curve.append(max(1, min(127, v)))
        return curve

    elif movement == PadMovement.EVOLVING:
        # Complex evolution
        curve = []
        for i in range(num_points):
            progress = i / num_points
            # Combine multiple waves
            wave1 = math.sin(progress * math.pi * 2) * 0.1
            wave2 = math.sin(progress * math.pi * 3.7) * 0.05
            v = int(base_velocity * (1 + wave1 + wave2))
            curve.append(max(1, min(127, v)))
        return curve

    elif movement == PadMovement.UNSTABLE:
        # Erratic movement
        curve = []
        for i in range(num_points):
            jitter = random.uniform(-0.2, 0.2)
            v = int(base_velocity * (1 + jitter))
            curve.append(max(1, min(127, v)))
        return curve

    return [base_velocity] * num_points


def add_extensions(
    pitches: List[int],
    root_midi: int,
    extension_prob: float
) -> List[int]:
    """Potentially add 7ths, 9ths, etc."""
    extended = pitches.copy()

    # Add 7th
    if random.random() < extension_prob:
        seventh = root_midi + 10  # Minor 7th (more common)
        if random.random() < 0.3:
            seventh = root_midi + 11  # Major 7th
        extended.append(seventh)

    # Add 9th
    if random.random() < extension_prob * 0.5:
        ninth = root_midi + 14
        extended.append(ninth)

    # Add 11th (rare)
    if random.random() < extension_prob * 0.2:
        eleventh = root_midi + 17
        extended.append(eleventh)

    return extended


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class PadEngine:
    """
    Generates emotion-driven pad textures.

    Pads provide the atmospheric foundation - they don't demand attention
    but they define the emotional space of the music.
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
        config: Optional[PadConfig] = None,
        texture_override: Optional[PadTexture] = None,
        movement_override: Optional[PadMovement] = None,
        voicing_override: Optional[PadVoicing] = None,
        register_override: Optional[PadRegister] = None,
    ) -> PadOutput:
        """
        Generate pad voicings for a chord progression.

        Args:
            emotion: Emotional state to express
            chord_progression: List of chord symbols
            key: Key center
            bars: Number of bars
            tempo_bpm: Tempo
            time_signature: Time signature as tuple
            config: Optional full configuration
            *_override: Override specific parameters

        Returns:
            PadOutput with all generated notes and metadata
        """
        # Get emotion profile
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])

        # Apply overrides
        texture = texture_override or profile["texture"]
        movement = movement_override or profile["movement"]
        voicing = voicing_override or profile["voicing"]
        register = register_override or profile["register"]

        # Calculate timing
        beats_per_bar = time_signature[0]
        ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
        total_ticks = ticks_per_bar * bars

        # Distribute chords across bars
        _chords_per_bar = len(chord_progression) / bars  # noqa: F841
        ticks_per_chord = int(total_ticks / len(chord_progression))

        # Generate pad for each chord
        pad_chords = []
        all_notes = []
        current_tick = 0

        for i, chord_symbol in enumerate(chord_progression):
            # Parse chord
            root, quality, bass = parse_chord(chord_symbol)
            base_octave = get_register_octave(register)

            # Get base pitches
            pitches = get_chord_pitches(root, quality, base_octave)

            # Add extensions if profile allows
            if profile["add_extensions"]:
                root_midi = note_name_to_midi(root, base_octave)
                pitches = add_extensions(pitches, root_midi, profile["extension_probability"])

            # Apply voicing
            voiced = apply_voicing(
                pitches,
                voicing,
                register,
                profile["density"]
            )

            # Add octave doubling
            if random.random() < profile["octave_doubling"]:
                if voiced:
                    root_pitch = voiced[0][0]
                    voiced.append((root_pitch - 12, "low_octave"))

            # Calculate duration with sustain ratio
            base_duration = ticks_per_chord
            duration = int(base_duration * profile["sustain_ratio"])

            # Apply attack offset
            attack = profile["attack"]
            start_offset = 0
            if attack == AttackType.SLOW:
                start_offset = -TICKS_PER_BEAT // 4  # Start early, swell in
            elif attack == AttackType.SWELL:
                start_offset = -TICKS_PER_BEAT // 2

            actual_start = max(0, current_tick + start_offset)

            # Generate notes
            chord_notes = []
            vel_min, vel_max = profile["velocity_range"]

            for pitch, func in voiced:
                # Velocity with some variation
                base_vel = random.randint(vel_min, vel_max)

                # Velocity curve for movement
                vel_curve = generate_velocity_curve(
                    base_vel,
                    duration,
                    movement,
                    profile["breathing_rate"]
                )

                note = PadNote(
                    pitch=pitch,
                    start_tick=actual_start,
                    duration_ticks=duration,
                    velocity=base_vel,
                    voice_function=func,
                    velocity_curve=vel_curve,
                )
                chord_notes.append(note)
                all_notes.append(note)

            pad_chord = PadChord(
                chord_symbol=chord_symbol,
                notes=chord_notes,
                start_tick=actual_start,
                duration_ticks=duration,
                texture=texture,
                movement=movement,
            )
            pad_chords.append(pad_chord)

            current_tick += ticks_per_chord

        return PadOutput(
            chords=pad_chords,
            notes=all_notes,
            emotion=emotion,
            texture_used=texture,
            movement_used=movement,
            voicing_used=voicing,
            register_used=register,
            gm_instrument=profile["gm_instrument"],
            total_ticks=total_ticks,
            metadata={
                "tempo_bpm": tempo_bpm,
                "time_signature": time_signature,
                "key": key,
                "chord_progression": chord_progression,
                "profile_used": emotion,
            }
        )

    def generate_for_section(
        self,
        emotion: str,
        chord_progression: List[str],
        section_type: str,
        key: str = "C",
        bars: int = 4,
        tempo_bpm: int = 120,
    ) -> PadOutput:
        """
        Generate pad adjusted for song section.

        Sections have different energy:
        - intro: sparse, mysterious
        - verse: supportive, not dominating
        - pre_chorus: building
        - chorus: full, powerful
        - bridge: transitional, different texture
        - outro: fading, resolving
        """
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])

        # Section-specific adjustments
        if section_type == "intro":
            movement = PadMovement.SWELLING
            voicing = PadVoicing.SPARSE
        elif section_type == "verse":
            movement = profile["movement"]
            voicing = PadVoicing.OPEN
        elif section_type == "pre_chorus":
            movement = PadMovement.SWELLING
            voicing = profile["voicing"]
        elif section_type == "chorus":
            movement = profile["movement"]
            voicing = PadVoicing.FULL
        elif section_type == "bridge":
            movement = PadMovement.EVOLVING
            voicing = PadVoicing.CLUSTER if random.random() < 0.3 else PadVoicing.OPEN
        elif section_type == "outro":
            movement = PadMovement.FADING
            voicing = PadVoicing.SPARSE
        else:
            movement = profile["movement"]
            voicing = profile["voicing"]

        return self.generate(
            emotion=emotion,
            chord_progression=chord_progression,
            key=key,
            bars=bars,
            tempo_bpm=tempo_bpm,
            movement_override=movement,
            voicing_override=voicing,
        )

    def get_available_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return list(self.profiles.keys())

    def get_emotion_profile(self, emotion: str) -> Dict:
        """Get the full profile for an emotion."""
        return self.profiles.get(emotion.lower(), self.profiles["neutral"])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_grief_pad(chords: List[str], key: str = "F", bars: int = 4) -> PadOutput:
    """Quick grief pad generation."""
    return PadEngine().generate("grief", chords, key, bars, 72)


def generate_ambient_pad(chords: List[str], key: str = "C", bars: int = 8) -> PadOutput:
    """Ambient, atmospheric pad."""
    engine = PadEngine()
    return engine.generate(
        "peace",
        chords,
        key,
        bars,
        60,
        texture_override=PadTexture.AIRY,
        movement_override=PadMovement.DRIFTING,
    )


def generate_tension_pad(chords: List[str], key: str = "Am", bars: int = 4) -> PadOutput:
    """Building tension pad."""
    engine = PadEngine()
    return engine.generate(
        "tension",
        chords,
        key,
        bars,
        90,
        voicing_override=PadVoicing.CLUSTER,
        movement_override=PadMovement.SWELLING,
    )


# =============================================================================
# MIDI EXPORT HELPER
# =============================================================================

def pad_to_midi_events(pad_output: PadOutput, channel: int = 3) -> List[Dict]:
    """
    Convert PadOutput to MIDI event list.

    Returns list of dicts with:
        - type: "note_on" or "note_off" or "program_change"
        - tick: absolute tick position
        - channel: MIDI channel
        - pitch: note number (for note events)
        - velocity: velocity (for note_on)
        - program: instrument number (for program_change)
    """
    events = []

    # Program change at start
    events.append({
        "type": "program_change",
        "tick": 0,
        "channel": channel,
        "program": pad_output.gm_instrument,
    })

    # Note events
    for note in pad_output.notes:
        # Note on
        events.append({
            "type": "note_on",
            "tick": note.start_tick,
            "channel": channel,
            "pitch": note.pitch,
            "velocity": note.velocity,
        })

        # Note off
        events.append({
            "type": "note_off",
            "tick": note.start_tick + note.duration_ticks,
            "channel": channel,
            "pitch": note.pitch,
            "velocity": 0,
        })

    # Sort by tick
    events.sort(key=lambda e: (e["tick"], 0 if e["type"] == "note_on" else 1))

    return events


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    engine = PadEngine()

    print("=== PAD ENGINE DEMO ===\n")

    chords = ["F", "C", "Dm", "Bbm"]

    emotions = ["grief", "hope", "rage", "anxiety", "peace"]

    for emotion in emotions:
        pad = engine.generate(
            emotion=emotion,
            chord_progression=chords,
            key="F",
            bars=4,
            tempo_bpm=82,
        )

        print(f"--- {emotion.upper()} ---")
        print(f"Texture: {pad.texture_used.value}")
        print(f"Movement: {pad.movement_used.value}")
        print(f"Voicing: {pad.voicing_used.value}")
        print(f"Register: {pad.register_used.value}")
        print(f"GM Instrument: {pad.gm_instrument}")
        print(f"Total notes: {len(pad.notes)}")
        print(f"Chords: {len(pad.chords)}")
        print()

    # Section demo
    print("=== SECTION VARIATION ===\n")
    sections = ["intro", "verse", "chorus", "outro"]
    for section in sections:
        pad = engine.generate_for_section(
            emotion="grief",
            chord_progression=chords,
            section_type=section,
            key="F",
            bars=4,
            tempo_bpm=82,
        )
        print(f"{section}: {pad.movement_used.value} / {pad.voicing_used.value}")
