"""
Fill Engine - Generates transitional embellishments and fills.

Part of Kelly MIDI Companion Tier 2.

Philosophy: Fills are the breath between phrases. They signal transitions,
build tension, release energy. A grief fill might be a soft roll that
fades. A rage fill might be a explosive burst. An anxious fill stutters
and hesitates. The fill IS the emotional punctuation.

Types of Fills:
    - Drum fills (tom rolls, snare builds, cymbal swells)
    - Melodic fills (runs, ornaments, grace notes)
    - Harmonic fills (chord stabs, arpeggiated sweeps)
    - Textural fills (risers, impacts, swells)

Integration:
    from kellymidicompanion_fill_engine import FillEngine, FillConfig
    
    engine = FillEngine()
    fill = engine.generate_drum_fill(
        emotion="grief",
        bars=1,
        tempo_bpm=82,
        position="end"  # "start", "middle", "end", "turnaround"
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

# GM Drum Map (Channel 10)
GM_DRUMS = {
    "kick": 36,
    "snare": 38,
    "sidestick": 37,
    "clap": 39,
    "low_tom": 45,
    "mid_tom": 47,
    "high_tom": 50,
    "floor_tom": 43,
    "closed_hat": 42,
    "open_hat": 46,
    "pedal_hat": 44,
    "crash": 49,
    "crash_2": 57,
    "ride": 51,
    "ride_bell": 53,
    "china": 52,
    "splash": 55,
    "tambourine": 54,
    "cowbell": 56,
}

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}


# =============================================================================
# ENUMS
# =============================================================================

class FillType(Enum):
    """Type of fill."""
    DRUM = "drum"                   # Drum kit fill
    MELODIC = "melodic"             # Melodic run/ornament
    HARMONIC = "harmonic"           # Chord-based fill
    TEXTURAL = "textural"           # Riser/impact/swell
    PERCUSSION = "percussion"       # Non-kit percussion


class DrumFillStyle(Enum):
    """Drum fill approach."""
    TOM_ROLL = "tom_roll"           # Classic tom descent
    SNARE_BUILD = "snare_build"     # Snare roll crescendo
    KICK_SNARE = "kick_snare"       # Alternating kick/snare
    LINEAR = "linear"               # Single hits, no overlaps
    BLAST = "blast"                 # All drums fast
    SPARSE = "sparse"               # Minimal, space
    CYMBAL_SWELL = "cymbal_swell"   # Building cymbal roll
    BROKEN = "broken"               # Irregular, hesitant
    FLAM = "flam"                   # Flammed notes
    GHOST = "ghost"                 # Ghost notes and touches


class MelodicFillStyle(Enum):
    """Melodic fill approach."""
    RUN_UP = "run_up"               # Ascending scale run
    RUN_DOWN = "run_down"           # Descending scale run
    ARPEGGIO = "arpeggio"           # Chord arpeggio
    TURN = "turn"                   # Ornamental turn
    TRILL = "trill"                 # Rapid alternation
    GRACE = "grace"                 # Grace notes
    CHROMATIC = "chromatic"         # Chromatic approach
    PENTATONIC = "pentatonic"       # Pentatonic lick


class FillPosition(Enum):
    """Where in the phrase the fill occurs."""
    START = "start"                 # Beginning of section
    MIDDLE = "middle"               # Mid-phrase
    END = "end"                     # End of phrase
    TURNAROUND = "turnaround"       # Before repeat
    TRANSITION = "transition"       # Between sections
    PICKUP = "pickup"               # Lead-in


class FillIntensity(Enum):
    """Energy level of fill."""
    WHISPER = "whisper"             # Very subtle
    SOFT = "soft"                   # Gentle
    MODERATE = "moderate"           # Standard
    STRONG = "strong"               # Powerful
    EXPLOSIVE = "explosive"         # Maximum impact


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_PROFILES = {
    "grief": {
        "drum_style": DrumFillStyle.SPARSE,
        "melodic_style": MelodicFillStyle.RUN_DOWN,
        "intensity": FillIntensity.SOFT,
        "velocity_range": (35, 65),
        "density": 0.4,              # How many notes
        "duration_beats": 2,         # Typical fill length
        "crescendo": False,
        "decrescendo": True,
        "humanize_timing": 25,
        "humanize_velocity": 12,
        "preferred_drums": ["snare", "floor_tom", "ride"],
        "cymbal_crash": False,
    },
    
    "sadness": {
        "drum_style": DrumFillStyle.GHOST,
        "melodic_style": MelodicFillStyle.RUN_DOWN,
        "intensity": FillIntensity.WHISPER,
        "velocity_range": (30, 55),
        "density": 0.3,
        "duration_beats": 1,
        "crescendo": False,
        "decrescendo": True,
        "humanize_timing": 20,
        "humanize_velocity": 10,
        "preferred_drums": ["sidestick", "closed_hat", "ride"],
        "cymbal_crash": False,
    },
    
    "hope": {
        "drum_style": DrumFillStyle.TOM_ROLL,
        "melodic_style": MelodicFillStyle.RUN_UP,
        "intensity": FillIntensity.MODERATE,
        "velocity_range": (55, 90),
        "density": 0.6,
        "duration_beats": 2,
        "crescendo": True,
        "decrescendo": False,
        "humanize_timing": 12,
        "humanize_velocity": 10,
        "preferred_drums": ["snare", "high_tom", "mid_tom", "crash"],
        "cymbal_crash": True,
    },
    
    "joy": {
        "drum_style": DrumFillStyle.LINEAR,
        "melodic_style": MelodicFillStyle.ARPEGGIO,
        "intensity": FillIntensity.STRONG,
        "velocity_range": (70, 110),
        "density": 0.8,
        "duration_beats": 2,
        "crescendo": True,
        "decrescendo": False,
        "humanize_timing": 8,
        "humanize_velocity": 12,
        "preferred_drums": ["snare", "high_tom", "mid_tom", "low_tom", "crash"],
        "cymbal_crash": True,
    },
    
    "rage": {
        "drum_style": DrumFillStyle.BLAST,
        "melodic_style": MelodicFillStyle.CHROMATIC,
        "intensity": FillIntensity.EXPLOSIVE,
        "velocity_range": (100, 127),
        "density": 1.0,
        "duration_beats": 2,
        "crescendo": True,
        "decrescendo": False,
        "humanize_timing": 3,
        "humanize_velocity": 5,
        "preferred_drums": ["kick", "snare", "crash", "china"],
        "cymbal_crash": True,
    },
    
    "anger": {
        "drum_style": DrumFillStyle.KICK_SNARE,
        "melodic_style": MelodicFillStyle.CHROMATIC,
        "intensity": FillIntensity.STRONG,
        "velocity_range": (85, 115),
        "density": 0.85,
        "duration_beats": 2,
        "crescendo": True,
        "decrescendo": False,
        "humanize_timing": 5,
        "humanize_velocity": 8,
        "preferred_drums": ["kick", "snare", "floor_tom", "crash"],
        "cymbal_crash": True,
    },
    
    "fear": {
        "drum_style": DrumFillStyle.BROKEN,
        "melodic_style": MelodicFillStyle.GRACE,
        "intensity": FillIntensity.SOFT,
        "velocity_range": (30, 70),
        "density": 0.4,
        "duration_beats": 1,
        "crescendo": False,
        "decrescendo": False,
        "humanize_timing": 35,
        "humanize_velocity": 20,
        "preferred_drums": ["sidestick", "closed_hat", "splash"],
        "cymbal_crash": False,
    },
    
    "anxiety": {
        "drum_style": DrumFillStyle.BROKEN,
        "melodic_style": MelodicFillStyle.TRILL,
        "intensity": FillIntensity.MODERATE,
        "velocity_range": (50, 85),
        "density": 0.6,
        "duration_beats": 1,
        "crescendo": False,
        "decrescendo": False,
        "humanize_timing": 40,
        "humanize_velocity": 18,
        "preferred_drums": ["snare", "closed_hat", "open_hat"],
        "cymbal_crash": False,
    },
    
    "tension": {
        "drum_style": DrumFillStyle.SNARE_BUILD,
        "melodic_style": MelodicFillStyle.CHROMATIC,
        "intensity": FillIntensity.STRONG,
        "velocity_range": (40, 100),
        "density": 0.9,
        "duration_beats": 4,
        "crescendo": True,
        "decrescendo": False,
        "humanize_timing": 10,
        "humanize_velocity": 8,
        "preferred_drums": ["snare", "kick"],
        "cymbal_crash": True,
    },
    
    "longing": {
        "drum_style": DrumFillStyle.CYMBAL_SWELL,
        "melodic_style": MelodicFillStyle.RUN_UP,
        "intensity": FillIntensity.SOFT,
        "velocity_range": (40, 75),
        "density": 0.5,
        "duration_beats": 2,
        "crescendo": True,
        "decrescendo": True,
        "humanize_timing": 18,
        "humanize_velocity": 12,
        "preferred_drums": ["ride", "crash", "floor_tom"],
        "cymbal_crash": False,
    },
    
    "nostalgia": {
        "drum_style": DrumFillStyle.FLAM,
        "melodic_style": MelodicFillStyle.TURN,
        "intensity": FillIntensity.SOFT,
        "velocity_range": (45, 70),
        "density": 0.45,
        "duration_beats": 1,
        "crescendo": False,
        "decrescendo": True,
        "humanize_timing": 20,
        "humanize_velocity": 10,
        "preferred_drums": ["snare", "mid_tom", "ride"],
        "cymbal_crash": False,
    },
    
    "peace": {
        "drum_style": DrumFillStyle.GHOST,
        "melodic_style": MelodicFillStyle.GRACE,
        "intensity": FillIntensity.WHISPER,
        "velocity_range": (25, 50),
        "density": 0.25,
        "duration_beats": 1,
        "crescendo": False,
        "decrescendo": True,
        "humanize_timing": 15,
        "humanize_velocity": 8,
        "preferred_drums": ["ride", "closed_hat"],
        "cymbal_crash": False,
    },
    
    "euphoria": {
        "drum_style": DrumFillStyle.TOM_ROLL,
        "melodic_style": MelodicFillStyle.ARPEGGIO,
        "intensity": FillIntensity.EXPLOSIVE,
        "velocity_range": (80, 120),
        "density": 0.95,
        "duration_beats": 4,
        "crescendo": True,
        "decrescendo": False,
        "humanize_timing": 5,
        "humanize_velocity": 10,
        "preferred_drums": ["high_tom", "mid_tom", "low_tom", "floor_tom", "crash", "crash_2"],
        "cymbal_crash": True,
    },
    
    "determination": {
        "drum_style": DrumFillStyle.LINEAR,
        "melodic_style": MelodicFillStyle.RUN_UP,
        "intensity": FillIntensity.STRONG,
        "velocity_range": (75, 105),
        "density": 0.8,
        "duration_beats": 2,
        "crescendo": True,
        "decrescendo": False,
        "humanize_timing": 5,
        "humanize_velocity": 6,
        "preferred_drums": ["snare", "kick", "crash"],
        "cymbal_crash": True,
    },
    
    "neutral": {
        "drum_style": DrumFillStyle.TOM_ROLL,
        "melodic_style": MelodicFillStyle.RUN_DOWN,
        "intensity": FillIntensity.MODERATE,
        "velocity_range": (60, 85),
        "density": 0.6,
        "duration_beats": 2,
        "crescendo": False,
        "decrescendo": False,
        "humanize_timing": 10,
        "humanize_velocity": 8,
        "preferred_drums": ["snare", "high_tom", "mid_tom", "low_tom"],
        "cymbal_crash": True,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FillNote:
    """A single note in a fill."""
    pitch: int                  # MIDI pitch (or drum note)
    start_tick: int
    duration_ticks: int
    velocity: int
    instrument_type: str        # "drum", "melodic", "harmonic"
    articulation: str           # "normal", "flam", "ghost", "accent"


@dataclass
class FillConfig:
    """Configuration for fill generation."""
    emotion: str = "neutral"
    position: FillPosition = FillPosition.END
    duration_beats: Optional[int] = None
    intensity_override: Optional[FillIntensity] = None


@dataclass
class FillOutput:
    """Complete fill output."""
    notes: List[FillNote]
    fill_type: FillType
    emotion: str
    style_used: str             # DrumFillStyle or MelodicFillStyle value
    intensity_used: FillIntensity
    position: FillPosition
    total_ticks: int
    has_crash: bool
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def note_name_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.strip()
    if len(note) > 1 and note[1] in ['#', 'b']:
        base = note[0].upper()
        modifier = note[1]
    else:
        base = note[0].upper()
        modifier = ''
    
    base_idx = CHROMATIC.index(base) if base in CHROMATIC else 0
    
    if modifier == '#':
        base_idx += 1
    elif modifier == 'b':
        base_idx -= 1
    
    base_idx = base_idx % 12
    return base_idx + (octave + 1) * 12


def get_scale_pitches(root: str, scale_type: str, octave: int = 4) -> List[int]:
    """Get pitches in a scale."""
    root_midi = note_name_to_midi(root, octave)
    intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])
    return [root_midi + i for i in intervals]


def calculate_velocity_envelope(
    base_velocity: int,
    num_notes: int,
    crescendo: bool,
    decrescendo: bool,
    velocity_range: Tuple[int, int]
) -> List[int]:
    """Calculate velocity envelope for fill notes."""
    vel_min, vel_max = velocity_range
    velocities = []
    
    for i in range(num_notes):
        progress = i / max(1, num_notes - 1)
        
        if crescendo and decrescendo:
            # Swell: up then down
            if progress < 0.5:
                mod = progress * 2
            else:
                mod = (1 - progress) * 2
        elif crescendo:
            mod = progress
        elif decrescendo:
            mod = 1 - progress
        else:
            mod = 0.5
        
        vel = int(vel_min + (vel_max - vel_min) * mod)
        velocities.append(max(1, min(127, vel)))
    
    return velocities


# =============================================================================
# DRUM FILL GENERATORS
# =============================================================================

def generate_tom_roll(
    profile: Dict,
    total_ticks: int,
    time_signature: Tuple[int, int]
) -> List[FillNote]:
    """Classic descending tom fill."""
    notes = []
    
    toms = ["high_tom", "mid_tom", "low_tom", "floor_tom"]
    toms = [t for t in toms if t in profile["preferred_drums"] or t in ["high_tom", "mid_tom", "low_tom"]]
    
    subdivision = TICKS_PER_BEAT // 4  # 16ths
    num_hits = int(total_ticks / subdivision * profile["density"])
    
    velocities = calculate_velocity_envelope(
        70, num_hits,
        profile["crescendo"],
        profile["decrescendo"],
        profile["velocity_range"]
    )
    
    current_tick = 0
    tom_idx = 0
    
    for i in range(num_hits):
        if current_tick >= total_ticks:
            break
        
        drum = toms[tom_idx % len(toms)]
        
        timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
        vel_offset = random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
        
        notes.append(FillNote(
            pitch=GM_DRUMS[drum],
            start_tick=max(0, current_tick + timing_offset),
            duration_ticks=subdivision - 20,
            velocity=max(1, min(127, velocities[i] + vel_offset)),
            instrument_type="drum",
            articulation="normal",
        ))
        
        current_tick += subdivision
        if i % 2 == 1:
            tom_idx += 1
    
    return notes


def generate_snare_build(
    profile: Dict,
    total_ticks: int,
    time_signature: Tuple[int, int]
) -> List[FillNote]:
    """Building snare roll."""
    notes = []
    
    # Start with 8ths, move to 16ths, then 32nds
    phases = [
        (TICKS_PER_BEAT // 2, 0.3),   # 8ths
        (TICKS_PER_BEAT // 4, 0.5),   # 16ths
        (TICKS_PER_BEAT // 8, 0.2),   # 32nds
    ]
    
    current_tick = 0
    all_positions = []
    
    for subdivision, phase_ratio in phases:
        phase_ticks = int(total_ticks * phase_ratio)
        while current_tick < sum(int(total_ticks * r) for s, r in phases[:phases.index((subdivision, phase_ratio)) + 1]):
            all_positions.append((current_tick, subdivision))
            current_tick += subdivision
            if current_tick >= total_ticks:
                break
    
    velocities = calculate_velocity_envelope(
        60, len(all_positions),
        True, False,  # Always crescendo for snare build
        profile["velocity_range"]
    )
    
    for i, (tick, subdiv) in enumerate(all_positions):
        if tick >= total_ticks:
            break
        
        timing_offset = random.randint(-profile["humanize_timing"] // 2, profile["humanize_timing"] // 2)
        vel_offset = random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
        
        notes.append(FillNote(
            pitch=GM_DRUMS["snare"],
            start_tick=max(0, tick + timing_offset),
            duration_ticks=subdiv - 10,
            velocity=max(1, min(127, velocities[i] + vel_offset)),
            instrument_type="drum",
            articulation="normal" if i % 4 != 0 else "accent",
        ))
    
    return notes


def generate_linear_fill(
    profile: Dict,
    total_ticks: int,
    time_signature: Tuple[int, int]
) -> List[FillNote]:
    """Linear fill - no overlapping hits."""
    notes = []
    drums = profile["preferred_drums"]
    
    subdivision = TICKS_PER_BEAT // 4
    num_hits = int(total_ticks / subdivision * profile["density"])
    
    velocities = calculate_velocity_envelope(
        70, num_hits,
        profile["crescendo"],
        profile["decrescendo"],
        profile["velocity_range"]
    )
    
    current_tick = 0
    
    for i in range(num_hits):
        if current_tick >= total_ticks:
            break
        
        drum = random.choice(drums)
        
        timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
        vel_offset = random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
        
        notes.append(FillNote(
            pitch=GM_DRUMS.get(drum, 38),
            start_tick=max(0, current_tick + timing_offset),
            duration_ticks=subdivision - 20,
            velocity=max(1, min(127, velocities[i] + vel_offset)),
            instrument_type="drum",
            articulation="normal",
        ))
        
        current_tick += subdivision
    
    return notes


def generate_broken_fill(
    profile: Dict,
    total_ticks: int,
    time_signature: Tuple[int, int]
) -> List[FillNote]:
    """Irregular, hesitant fill."""
    notes = []
    drums = profile["preferred_drums"]
    
    subdivisions = [TICKS_PER_BEAT // 4, TICKS_PER_BEAT // 3, TICKS_PER_BEAT // 2]
    
    current_tick = 0
    note_count = 0
    
    while current_tick < total_ticks:
        # Random gaps
        if random.random() > profile["density"]:
            current_tick += random.choice(subdivisions)
            continue
        
        drum = random.choice(drums)
        subdiv = random.choice(subdivisions)
        
        timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
        vel = random.randint(*profile["velocity_range"])
        
        notes.append(FillNote(
            pitch=GM_DRUMS.get(drum, 38),
            start_tick=max(0, current_tick + timing_offset),
            duration_ticks=subdiv // 2,
            velocity=vel,
            instrument_type="drum",
            articulation="ghost" if vel < 50 else "normal",
        ))
        
        current_tick += subdiv
        note_count += 1
    
    return notes


def generate_ghost_fill(
    profile: Dict,
    total_ticks: int,
    time_signature: Tuple[int, int]
) -> List[FillNote]:
    """Ghost notes and touches."""
    notes = []
    
    subdivision = TICKS_PER_BEAT // 4
    num_slots = total_ticks // subdivision
    
    for i in range(num_slots):
        if random.random() > profile["density"]:
            continue
        
        tick = i * subdivision
        vel = random.randint(20, 50)  # All quiet
        
        timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
        
        # Mostly snare ghosts
        drum = "snare" if random.random() < 0.7 else random.choice(profile["preferred_drums"])
        
        notes.append(FillNote(
            pitch=GM_DRUMS.get(drum, 38),
            start_tick=max(0, tick + timing_offset),
            duration_ticks=subdivision // 3,
            velocity=vel,
            instrument_type="drum",
            articulation="ghost",
        ))
    
    return notes


def generate_blast_fill(
    profile: Dict,
    total_ticks: int,
    time_signature: Tuple[int, int]
) -> List[FillNote]:
    """Maximum intensity blast."""
    notes = []
    
    subdivision = TICKS_PER_BEAT // 8  # 32nds
    
    current_tick = 0
    vel_min, vel_max = profile["velocity_range"]
    
    while current_tick < total_ticks:
        # Kick and snare together
        for drum in ["kick", "snare"]:
            timing_offset = random.randint(-3, 3)
            notes.append(FillNote(
                pitch=GM_DRUMS[drum],
                start_tick=max(0, current_tick + timing_offset),
                duration_ticks=subdivision - 5,
                velocity=random.randint(vel_min, vel_max),
                instrument_type="drum",
                articulation="accent",
            ))
        
        current_tick += subdivision
    
    return notes


# =============================================================================
# MELODIC FILL GENERATORS
# =============================================================================

def generate_run_up(
    profile: Dict,
    total_ticks: int,
    key: str,
    scale: str,
    start_octave: int
) -> List[FillNote]:
    """Ascending scale run."""
    notes = []
    
    pitches = get_scale_pitches(key, scale, start_octave)
    pitches += [p + 12 for p in pitches]  # Add upper octave
    
    num_notes = int(len(pitches) * profile["density"])
    subdivision = total_ticks // num_notes if num_notes > 0 else TICKS_PER_BEAT // 4
    
    velocities = calculate_velocity_envelope(
        70, num_notes,
        profile["crescendo"],
        profile["decrescendo"],
        profile["velocity_range"]
    )
    
    for i in range(num_notes):
        tick = i * subdivision
        if tick >= total_ticks:
            break
        
        timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
        vel_offset = random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
        
        notes.append(FillNote(
            pitch=pitches[i % len(pitches)],
            start_tick=max(0, tick + timing_offset),
            duration_ticks=int(subdivision * 0.8),
            velocity=max(1, min(127, velocities[i] + vel_offset)),
            instrument_type="melodic",
            articulation="normal",
        ))
    
    return notes


def generate_run_down(
    profile: Dict,
    total_ticks: int,
    key: str,
    scale: str,
    start_octave: int
) -> List[FillNote]:
    """Descending scale run."""
    notes = []
    
    pitches = get_scale_pitches(key, scale, start_octave + 1)
    pitches += get_scale_pitches(key, scale, start_octave)
    pitches = pitches[::-1]  # Reverse for descending
    
    num_notes = int(len(pitches) * profile["density"])
    subdivision = total_ticks // num_notes if num_notes > 0 else TICKS_PER_BEAT // 4
    
    velocities = calculate_velocity_envelope(
        70, num_notes,
        profile["crescendo"],
        profile["decrescendo"],
        profile["velocity_range"]
    )
    
    for i in range(num_notes):
        tick = i * subdivision
        if tick >= total_ticks:
            break
        
        timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
        vel_offset = random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
        
        notes.append(FillNote(
            pitch=pitches[i % len(pitches)],
            start_tick=max(0, tick + timing_offset),
            duration_ticks=int(subdivision * 0.8),
            velocity=max(1, min(127, velocities[i] + vel_offset)),
            instrument_type="melodic",
            articulation="normal",
        ))
    
    return notes


def generate_chromatic_run(
    profile: Dict,
    total_ticks: int,
    target_pitch: int,
    ascending: bool = True
) -> List[FillNote]:
    """Chromatic approach to target."""
    notes = []
    
    num_notes = int(8 * profile["density"])
    subdivision = total_ticks // num_notes if num_notes > 0 else TICKS_PER_BEAT // 4
    
    if ascending:
        pitches = list(range(target_pitch - num_notes, target_pitch + 1))
    else:
        pitches = list(range(target_pitch + num_notes, target_pitch - 1, -1))
    
    velocities = calculate_velocity_envelope(
        70, len(pitches),
        profile["crescendo"],
        profile["decrescendo"],
        profile["velocity_range"]
    )
    
    for i, pitch in enumerate(pitches):
        tick = i * subdivision
        if tick >= total_ticks:
            break
        
        timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
        vel_offset = random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
        
        notes.append(FillNote(
            pitch=pitch,
            start_tick=max(0, tick + timing_offset),
            duration_ticks=int(subdivision * 0.7),
            velocity=max(1, min(127, velocities[i] + vel_offset)),
            instrument_type="melodic",
            articulation="normal",
        ))
    
    return notes


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class FillEngine:
    """
    Generates transitional fills for drums and melodic instruments.
    
    Fills provide emotional punctuation - signaling transitions,
    building tension, and releasing energy.
    """
    
    def __init__(self):
        self.profiles = EMOTION_PROFILES
    
    def generate_drum_fill(
        self,
        emotion: str,
        bars: float = 0.5,
        tempo_bpm: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        position: FillPosition = FillPosition.END,
        style_override: Optional[DrumFillStyle] = None,
        intensity_override: Optional[FillIntensity] = None,
    ) -> FillOutput:
        """Generate a drum fill."""
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])
        
        style = style_override or profile["drum_style"]
        intensity = intensity_override or profile["intensity"]
        
        beats_per_bar = time_signature[0]
        duration_beats = profile["duration_beats"]
        total_ticks = int(TICKS_PER_BEAT * beats_per_bar * bars)
        
        # Generate based on style
        if style == DrumFillStyle.TOM_ROLL:
            notes = generate_tom_roll(profile, total_ticks, time_signature)
        elif style == DrumFillStyle.SNARE_BUILD:
            notes = generate_snare_build(profile, total_ticks, time_signature)
        elif style == DrumFillStyle.LINEAR:
            notes = generate_linear_fill(profile, total_ticks, time_signature)
        elif style == DrumFillStyle.BROKEN:
            notes = generate_broken_fill(profile, total_ticks, time_signature)
        elif style == DrumFillStyle.GHOST:
            notes = generate_ghost_fill(profile, total_ticks, time_signature)
        elif style == DrumFillStyle.BLAST:
            notes = generate_blast_fill(profile, total_ticks, time_signature)
        elif style == DrumFillStyle.KICK_SNARE:
            notes = generate_linear_fill(profile, total_ticks, time_signature)
        elif style == DrumFillStyle.CYMBAL_SWELL:
            notes = generate_snare_build(profile, total_ticks, time_signature)
        else:
            notes = generate_tom_roll(profile, total_ticks, time_signature)
        
        # Add crash at end if specified
        has_crash = profile["cymbal_crash"]
        if has_crash and notes:
            crash_tick = total_ticks - 10
            notes.append(FillNote(
                pitch=GM_DRUMS["crash"],
                start_tick=crash_tick,
                duration_ticks=TICKS_PER_BEAT * 2,
                velocity=profile["velocity_range"][1],
                instrument_type="drum",
                articulation="accent",
            ))
        
        notes.sort(key=lambda n: n.start_tick)
        
        return FillOutput(
            notes=notes,
            fill_type=FillType.DRUM,
            emotion=emotion,
            style_used=style.value,
            intensity_used=intensity,
            position=position,
            total_ticks=total_ticks,
            has_crash=has_crash,
            metadata={
                "tempo_bpm": tempo_bpm,
                "time_signature": time_signature,
                "bars": bars,
            }
        )
    
    def generate_melodic_fill(
        self,
        emotion: str,
        key: str = "C",
        scale: str = "major",
        bars: float = 0.5,
        tempo_bpm: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        target_pitch: int = 60,
        position: FillPosition = FillPosition.END,
        style_override: Optional[MelodicFillStyle] = None,
    ) -> FillOutput:
        """Generate a melodic fill/run."""
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])
        
        style = style_override or profile["melodic_style"]
        intensity = profile["intensity"]
        
        beats_per_bar = time_signature[0]
        total_ticks = int(TICKS_PER_BEAT * beats_per_bar * bars)
        
        start_octave = (target_pitch // 12) - 1
        
        # Generate based on style
        if style == MelodicFillStyle.RUN_UP:
            notes = generate_run_up(profile, total_ticks, key, scale, start_octave)
        elif style == MelodicFillStyle.RUN_DOWN:
            notes = generate_run_down(profile, total_ticks, key, scale, start_octave)
        elif style == MelodicFillStyle.CHROMATIC:
            notes = generate_chromatic_run(profile, total_ticks, target_pitch, True)
        elif style == MelodicFillStyle.ARPEGGIO:
            notes = generate_run_up(profile, total_ticks, key, "pentatonic_major", start_octave)
        elif style == MelodicFillStyle.PENTATONIC:
            notes = generate_run_up(profile, total_ticks, key, "pentatonic_minor", start_octave)
        else:
            notes = generate_run_down(profile, total_ticks, key, scale, start_octave)
        
        notes.sort(key=lambda n: n.start_tick)
        
        return FillOutput(
            notes=notes,
            fill_type=FillType.MELODIC,
            emotion=emotion,
            style_used=style.value,
            intensity_used=intensity,
            position=position,
            total_ticks=total_ticks,
            has_crash=False,
            metadata={
                "tempo_bpm": tempo_bpm,
                "time_signature": time_signature,
                "key": key,
                "scale": scale,
                "target_pitch": target_pitch,
            }
        )
    
    def get_available_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return list(self.profiles.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_grief_fill(bars: float = 0.5, tempo: int = 72) -> FillOutput:
    """Quick grief drum fill."""
    return FillEngine().generate_drum_fill("grief", bars, tempo)


def generate_epic_fill(bars: float = 1.0, tempo: int = 100) -> FillOutput:
    """Epic building fill."""
    return FillEngine().generate_drum_fill("euphoria", bars, tempo)


def fill_to_midi_events(fill_output: FillOutput, channel: int = 9) -> List[Dict]:
    """Convert FillOutput to MIDI event list."""
    events = []
    
    for note in fill_output.notes:
        events.append({
            "type": "note_on",
            "tick": note.start_tick,
            "channel": channel,
            "pitch": note.pitch,
            "velocity": note.velocity,
        })
        events.append({
            "type": "note_off",
            "tick": note.start_tick + note.duration_ticks,
            "channel": channel,
            "pitch": note.pitch,
            "velocity": 0,
        })
    
    events.sort(key=lambda e: (e["tick"], 0 if e["type"] == "note_on" else 1))
    return events


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    engine = FillEngine()
    
    print("=== FILL ENGINE DEMO ===\n")
    
    print("--- DRUM FILLS ---\n")
    
    emotions = ["grief", "hope", "rage", "anxiety", "peace"]
    
    for emotion in emotions:
        fill = engine.generate_drum_fill(
            emotion=emotion,
            bars=0.5,
            tempo_bpm=100,
        )
        
        print(f"{emotion.upper()}")
        print(f"  Style: {fill.style_used}")
        print(f"  Intensity: {fill.intensity_used.value}")
        print(f"  Notes: {len(fill.notes)}")
        print(f"  Crash: {fill.has_crash}")
    
    print("\n--- MELODIC FILLS ---\n")
    
    for emotion in ["hope", "sadness", "joy"]:
        fill = engine.generate_melodic_fill(
            emotion=emotion,
            key="F",
            scale="major",
            bars=0.5,
            tempo_bpm=100,
            target_pitch=65,
        )
        
        print(f"{emotion.upper()}")
        print(f"  Style: {fill.style_used}")
        print(f"  Notes: {len(fill.notes)}")
        if fill.notes:
            pitches = [n.pitch for n in fill.notes]
            print(f"  Range: {min(pitches)} - {max(pitches)}")
