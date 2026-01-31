"""
Rhythm Pattern Engine - Generates emotion-driven drum patterns.

Part of Kelly MIDI Companion Tier 1 MVP.

Philosophy: Drums are the heartbeat. Grief drums hesitate and skip beats.
Rage drums attack relentlessly. Anxiety drums stutter and never settle.
The pattern IS the pulse of the emotion.

GM Drum Map (Channel 10):
    36 = Kick
    38 = Snare
    42 = Closed Hi-Hat
    46 = Open Hi-Hat
    49 = Crash
    51 = Ride
    45 = Low Tom
    47 = Mid Tom
    50 = High Tom
    39 = Clap
    37 = Sidestick

Integration:
    from kellymidicompanion_rhythm_engine import RhythmEngine, RhythmConfig
    
    engine = RhythmEngine()
    pattern = engine.generate(
        emotion="grief",
        bars=4,
        tempo_bpm=82,
        genre="lo-fi"
    )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import random
import math


# =============================================================================
# CONSTANTS
# =============================================================================

TICKS_PER_BEAT = 480

# General MIDI Drum Map (Channel 10)
GM_DRUMS = {
    "kick": 36,
    "snare": 38,
    "sidestick": 37,
    "clap": 39,
    "closed_hat": 42,
    "open_hat": 46,
    "pedal_hat": 44,
    "crash": 49,
    "ride": 51,
    "ride_bell": 53,
    "low_tom": 45,
    "mid_tom": 47,
    "high_tom": 50,
    "tambourine": 54,
    "cowbell": 56,
    "shaker": 70,
    "rim": 37,
}

# Instrument groups for selection
DRUM_GROUPS = {
    "foundation": ["kick", "snare"],
    "hats": ["closed_hat", "open_hat", "pedal_hat"],
    "cymbals": ["crash", "ride", "ride_bell"],
    "toms": ["low_tom", "mid_tom", "high_tom"],
    "percussion": ["tambourine", "cowbell", "shaker", "clap"],
    "ghost": ["sidestick", "rim"],
}


# =============================================================================
# ENUMS
# =============================================================================

class GrooveType(Enum):
    """Core groove feel."""
    STRAIGHT = "straight"           # Even 8ths/16ths
    SWING = "swing"                 # Triplet feel
    SHUFFLE = "shuffle"             # Heavy shuffle
    HALF_TIME = "half_time"         # Snare on 3
    DOUBLE_TIME = "double_time"     # Fast feel
    BROKEN = "broken"               # Fragmented
    FOUR_ON_FLOOR = "four_on_floor" # Dance kick
    BOOM_BAP = "boom_bap"           # Hip-hop
    TRAP = "trap"                   # Hi-hat rolls
    HUMAN = "human"                 # Imperfect, organic


class PatternDensity(Enum):
    """How busy the pattern is."""
    MINIMAL = "minimal"     # Kick and snare only
    SPARSE = "sparse"       # Light hats
    MODERATE = "moderate"   # Standard kit
    BUSY = "busy"           # Full kit, fills
    CHAOTIC = "chaotic"     # Overloaded


class AccentPattern(Enum):
    """Where accents fall."""
    DOWNBEAT = "downbeat"       # 1 and 3
    BACKBEAT = "backbeat"       # 2 and 4
    OFFBEAT = "offbeat"         # And's
    RANDOM = "random"           # Unpredictable
    NONE = "none"               # Even dynamics


# =============================================================================
# EMOTION TO RHYTHM MAPPING
# =============================================================================

EMOTION_RHYTHM_PROFILES = {
    "grief": {
        "grooves": [GrooveType.HALF_TIME, GrooveType.BROKEN, GrooveType.HUMAN],
        "density": PatternDensity.SPARSE,
        "accent": AccentPattern.DOWNBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Just beat 1
        "snare_pattern": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Beat 3
        "hat_probability": 0.3,
        "ghost_note_prob": 0.1,
        "fill_probability": 0.1,
        "swing_amount": 0.0,
        "velocity_range": (35, 70),
        "timing_humanize": 0.03,      # Percentage of beat
        "drop_probability": 0.2,       # Chance to skip a hit
        "instrument_preference": ["kick", "snare", "sidestick", "closed_hat"],
    },
    
    "hope": {
        "grooves": [GrooveType.STRAIGHT, GrooveType.HUMAN, GrooveType.SWING],
        "density": PatternDensity.MODERATE,
        "accent": AccentPattern.BACKBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.7,
        "ghost_note_prob": 0.2,
        "fill_probability": 0.2,
        "swing_amount": 0.1,
        "velocity_range": (55, 95),
        "timing_humanize": 0.02,
        "drop_probability": 0.05,
        "instrument_preference": ["kick", "snare", "closed_hat", "ride", "tambourine"],
    },
    
    "rage": {
        "grooves": [GrooveType.STRAIGHT, GrooveType.DOUBLE_TIME, GrooveType.FOUR_ON_FLOOR],
        "density": PatternDensity.BUSY,
        "accent": AccentPattern.DOWNBEAT,
        "kick_pattern": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Relentless
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.9,
        "ghost_note_prob": 0.3,
        "fill_probability": 0.4,
        "swing_amount": 0.0,
        "velocity_range": (90, 127),
        "timing_humanize": 0.01,       # Tight
        "drop_probability": 0.0,
        "instrument_preference": ["kick", "snare", "crash", "closed_hat", "low_tom"],
    },
    
    "fear": {
        "grooves": [GrooveType.BROKEN, GrooveType.TRAP, GrooveType.HUMAN],
        "density": PatternDensity.SPARSE,
        "accent": AccentPattern.RANDOM,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Unexpected
        "hat_probability": 0.4,
        "ghost_note_prob": 0.4,
        "fill_probability": 0.15,
        "swing_amount": 0.0,
        "velocity_range": (30, 90),
        "timing_humanize": 0.04,
        "drop_probability": 0.3,
        "instrument_preference": ["kick", "sidestick", "closed_hat", "shaker"],
    },
    
    "joy": {
        "grooves": [GrooveType.SWING, GrooveType.SHUFFLE, GrooveType.STRAIGHT],
        "density": PatternDensity.MODERATE,
        "accent": AccentPattern.BACKBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.8,
        "ghost_note_prob": 0.25,
        "fill_probability": 0.25,
        "swing_amount": 0.2,
        "velocity_range": (70, 110),
        "timing_humanize": 0.02,
        "drop_probability": 0.0,
        "instrument_preference": ["kick", "snare", "open_hat", "closed_hat", "tambourine"],
    },
    
    "longing": {
        "grooves": [GrooveType.HALF_TIME, GrooveType.HUMAN, GrooveType.SWING],
        "density": PatternDensity.SPARSE,
        "accent": AccentPattern.DOWNBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.4,
        "ghost_note_prob": 0.15,
        "fill_probability": 0.1,
        "swing_amount": 0.1,
        "velocity_range": (40, 75),
        "timing_humanize": 0.03,
        "drop_probability": 0.15,
        "instrument_preference": ["kick", "snare", "ride", "closed_hat"],
    },
    
    "anxiety": {
        "grooves": [GrooveType.TRAP, GrooveType.BROKEN, GrooveType.STRAIGHT],
        "density": PatternDensity.BUSY,
        "accent": AccentPattern.OFFBEAT,
        "kick_pattern": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        "hat_probability": 0.95,
        "ghost_note_prob": 0.4,
        "fill_probability": 0.3,
        "swing_amount": 0.0,
        "velocity_range": (50, 100),
        "timing_humanize": 0.025,
        "drop_probability": 0.1,
        "instrument_preference": ["kick", "snare", "closed_hat", "shaker", "clap"],
    },
    
    "tenderness": {
        "grooves": [GrooveType.HUMAN, GrooveType.SWING, GrooveType.HALF_TIME],
        "density": PatternDensity.MINIMAL,
        "accent": AccentPattern.NONE,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.2,
        "ghost_note_prob": 0.05,
        "fill_probability": 0.05,
        "swing_amount": 0.05,
        "velocity_range": (30, 60),
        "timing_humanize": 0.035,
        "drop_probability": 0.25,
        "instrument_preference": ["kick", "sidestick", "shaker"],
    },
    
    "defiance": {
        "grooves": [GrooveType.STRAIGHT, GrooveType.BOOM_BAP, GrooveType.DOUBLE_TIME],
        "density": PatternDensity.MODERATE,
        "accent": AccentPattern.BACKBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        "hat_probability": 0.75,
        "ghost_note_prob": 0.3,
        "fill_probability": 0.35,
        "swing_amount": 0.0,
        "velocity_range": (75, 115),
        "timing_humanize": 0.015,
        "drop_probability": 0.0,
        "instrument_preference": ["kick", "snare", "crash", "closed_hat", "clap"],
    },
    
    "melancholy": {
        "grooves": [GrooveType.HALF_TIME, GrooveType.SWING, GrooveType.HUMAN],
        "density": PatternDensity.SPARSE,
        "accent": AccentPattern.DOWNBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "hat_probability": 0.35,
        "ghost_note_prob": 0.1,
        "fill_probability": 0.1,
        "swing_amount": 0.1,
        "velocity_range": (35, 70),
        "timing_humanize": 0.04,
        "drop_probability": 0.2,
        "instrument_preference": ["kick", "snare", "ride", "sidestick"],
    },
    
    "nostalgia": {
        "grooves": [GrooveType.SHUFFLE, GrooveType.SWING, GrooveType.HUMAN],
        "density": PatternDensity.MODERATE,
        "accent": AccentPattern.BACKBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.6,
        "ghost_note_prob": 0.2,
        "fill_probability": 0.2,
        "swing_amount": 0.25,
        "velocity_range": (50, 85),
        "timing_humanize": 0.025,
        "drop_probability": 0.1,
        "instrument_preference": ["kick", "snare", "closed_hat", "tambourine", "ride"],
    },
    
    "euphoria": {
        "grooves": [GrooveType.FOUR_ON_FLOOR, GrooveType.STRAIGHT, GrooveType.DOUBLE_TIME],
        "density": PatternDensity.BUSY,
        "accent": AccentPattern.DOWNBEAT,
        "kick_pattern": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.9,
        "ghost_note_prob": 0.35,
        "fill_probability": 0.4,
        "swing_amount": 0.0,
        "velocity_range": (80, 120),
        "timing_humanize": 0.01,
        "drop_probability": 0.0,
        "instrument_preference": ["kick", "snare", "open_hat", "closed_hat", "crash"],
    },
    
    "dissociation": {
        "grooves": [GrooveType.BROKEN, GrooveType.HALF_TIME, GrooveType.HUMAN],
        "density": PatternDensity.MINIMAL,
        "accent": AccentPattern.RANDOM,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "hat_probability": 0.15,
        "ghost_note_prob": 0.05,
        "fill_probability": 0.05,
        "swing_amount": 0.0,
        "velocity_range": (25, 50),
        "timing_humanize": 0.05,
        "drop_probability": 0.5,
        "instrument_preference": ["kick", "sidestick", "shaker"],
    },
    
    "neutral": {
        "grooves": [GrooveType.STRAIGHT, GrooveType.HUMAN],
        "density": PatternDensity.MODERATE,
        "accent": AccentPattern.BACKBEAT,
        "kick_pattern": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "snare_pattern": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat_probability": 0.7,
        "ghost_note_prob": 0.2,
        "fill_probability": 0.2,
        "swing_amount": 0.0,
        "velocity_range": (60, 95),
        "timing_humanize": 0.02,
        "drop_probability": 0.05,
        "instrument_preference": ["kick", "snare", "closed_hat", "open_hat"],
    },
}

# Genre modifiers
GENRE_MODIFIERS = {
    "lo-fi": {
        "velocity_mult": 0.7,
        "timing_humanize_mult": 2.0,
        "hat_probability_mult": 0.6,
        "preferred_instruments": ["kick", "snare", "closed_hat", "sidestick"],
    },
    "rock": {
        "velocity_mult": 1.1,
        "timing_humanize_mult": 0.8,
        "hat_probability_mult": 1.0,
        "preferred_instruments": ["kick", "snare", "crash", "open_hat", "closed_hat"],
    },
    "electronic": {
        "velocity_mult": 1.0,
        "timing_humanize_mult": 0.3,
        "hat_probability_mult": 1.2,
        "preferred_instruments": ["kick", "snare", "clap", "closed_hat", "open_hat"],
    },
    "jazz": {
        "velocity_mult": 0.85,
        "timing_humanize_mult": 1.5,
        "hat_probability_mult": 0.8,
        "preferred_instruments": ["kick", "snare", "ride", "closed_hat"],
        "swing_override": 0.3,
    },
    "hip-hop": {
        "velocity_mult": 1.0,
        "timing_humanize_mult": 1.0,
        "hat_probability_mult": 1.1,
        "preferred_instruments": ["kick", "snare", "clap", "closed_hat", "open_hat"],
    },
    "ambient": {
        "velocity_mult": 0.5,
        "timing_humanize_mult": 2.5,
        "hat_probability_mult": 0.3,
        "preferred_instruments": ["kick", "shaker", "ride"],
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DrumHit:
    """A single drum hit."""
    instrument: str         # "kick", "snare", etc.
    midi_note: int          # GM drum note
    start_tick: int         # Position
    duration_ticks: int     # Length
    velocity: int           # 0-127
    is_ghost: bool          # Ghost note
    timing_offset: int      # Humanization offset


@dataclass
class RhythmConfig:
    """Configuration for rhythm generation."""
    emotion: str = "neutral"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    genre: Optional[str] = None
    seed: Optional[int] = None
    groove_override: Optional[GrooveType] = None
    density_override: Optional[PatternDensity] = None
    include_fills: bool = True
    fill_on_bar: Optional[List[int]] = None  # Which bars get fills


@dataclass
class RhythmOutput:
    """Output from rhythm generation."""
    hits: List[DrumHit]
    config: RhythmConfig
    groove_used: GrooveType
    density_used: PatternDensity
    total_ticks: int
    instruments_used: Set[str]
    
    def to_midi_events(self) -> List[Dict]:
        """Convert to MIDI events (channel 9 for drums)."""
        events = []
        for hit in self.hits:
            events.append({
                "type": "note_on",
                "channel": 9,
                "note": hit.midi_note,
                "velocity": hit.velocity,
                "time": hit.start_tick + hit.timing_offset,
            })
            events.append({
                "type": "note_off",
                "channel": 9,
                "note": hit.midi_note,
                "velocity": 0,
                "time": hit.start_tick + hit.timing_offset + hit.duration_ticks,
            })
        return sorted(events, key=lambda e: (e["time"], e["type"] == "note_off"))
    
    def to_hit_list(self) -> List[Tuple[str, int, int, int]]:
        """Return as (instrument, tick, duration, velocity)."""
        return [
            (h.instrument, h.start_tick + h.timing_offset, h.duration_ticks, h.velocity)
            for h in self.hits
        ]


# =============================================================================
# PATTERN GENERATORS
# =============================================================================

def apply_swing(tick: int, beat_ticks: int, swing_amount: float) -> int:
    """Apply swing to a tick position."""
    if swing_amount == 0:
        return tick
    
    eighth = beat_ticks // 2
    position_in_beat = tick % beat_ticks
    
    # Swing affects off-beats (the "and"s)
    if position_in_beat >= eighth:
        # This is an off-beat - delay it
        swing_offset = int(eighth * swing_amount)
        return tick + swing_offset
    
    return tick


def generate_hat_pattern(
    bar_ticks: int,
    profile: Dict,
    groove: GrooveType,
    time_sig: Tuple[int, int]
) -> List[Tuple[int, str]]:
    """Generate hi-hat pattern: (tick, "closed_hat"/"open_hat")"""
    hits = []
    
    sixteenth = TICKS_PER_BEAT // 4
    num_sixteenths = bar_ticks // sixteenth
    
    hat_prob = profile["hat_probability"]
    
    for i in range(num_sixteenths):
        if random.random() < hat_prob:
            tick = i * sixteenth
            
            # Open hat on certain beats
            if groove == GrooveType.SWING and i % 8 == 4:
                instrument = "open_hat"
            elif i % 8 == 6:  # Before beat 3
                instrument = "open_hat" if random.random() < 0.3 else "closed_hat"
            else:
                instrument = "closed_hat"
            
            hits.append((tick, instrument))
    
    return hits


def generate_trap_hats(
    bar_ticks: int,
    profile: Dict,
    time_sig: Tuple[int, int]
) -> List[Tuple[int, str, int]]:
    """Generate trap-style hi-hat rolls: (tick, instrument, velocity)"""
    hits = []
    
    thirty_second = TICKS_PER_BEAT // 8
    
    # Random roll positions
    num_rolls = random.randint(2, 4)
    roll_starts = sorted(random.sample(range(0, bar_ticks // TICKS_PER_BEAT * 4), num_rolls))
    
    for roll_beat in roll_starts:
        roll_start = roll_beat * TICKS_PER_BEAT // 4
        roll_length = random.randint(4, 8)  # 32nd notes in roll
        
        for i in range(roll_length):
            tick = roll_start + i * thirty_second
            if tick < bar_ticks:
                # Velocity ramp
                vel = 60 + int(30 * (i / roll_length))
                hits.append((tick, "closed_hat", vel))
    
    return hits


def generate_fill(
    start_tick: int,
    duration_ticks: int,
    profile: Dict
) -> List[Tuple[int, str, int]]:
    """Generate a drum fill: (tick, instrument, velocity)"""
    hits = []
    
    sixteenth = TICKS_PER_BEAT // 4
    num_notes = duration_ticks // sixteenth
    
    # Fill instruments
    fill_instruments = ["snare", "high_tom", "mid_tom", "low_tom"]
    
    for i in range(num_notes):
        tick = start_tick + i * sixteenth
        instrument = random.choice(fill_instruments)
        
        # Build velocity toward end
        vel = 70 + int(40 * (i / num_notes))
        vel = min(127, vel)
        
        hits.append((tick, instrument, vel))
    
    # Crash at end
    hits.append((start_tick + duration_ticks, "crash", 100))
    
    return hits


# =============================================================================
# MAIN ENGINE
# =============================================================================

class RhythmEngine:
    """
    Generates emotion-driven drum patterns.
    
    Usage:
        engine = RhythmEngine()
        pattern = engine.generate(
            emotion="grief",
            bars=4,
            tempo_bpm=82,
            genre="lo-fi"
        )
    """
    
    def __init__(self, custom_profiles: Optional[Dict] = None):
        self.profiles = EMOTION_RHYTHM_PROFILES.copy()
        if custom_profiles:
            self.profiles.update(custom_profiles)
    
    def generate(
        self,
        emotion: str = "neutral",
        bars: int = 4,
        tempo_bpm: int = 120,
        genre: Optional[str] = None,
        **kwargs
    ) -> RhythmOutput:
        """
        Generate drum pattern.
        
        Args:
            emotion: Emotional state
            bars: Number of bars
            tempo_bpm: Tempo
            genre: Optional genre modifier
        """
        config = RhythmConfig(
            emotion=emotion.lower(),
            bars=bars,
            tempo_bpm=tempo_bpm,
            genre=genre,
            **kwargs
        )
        
        return self._generate_from_config(config)
    
    def _generate_from_config(self, config: RhythmConfig) -> RhythmOutput:
        """Internal generation."""
        
        if config.seed is not None:
            random.seed(config.seed)
        
        profile = self.profiles.get(config.emotion, self.profiles["neutral"])
        
        # Apply genre modifiers
        if config.genre and config.genre in GENRE_MODIFIERS:
            modifier = GENRE_MODIFIERS[config.genre]
            profile = profile.copy()
            profile["velocity_range"] = (
                int(profile["velocity_range"][0] * modifier["velocity_mult"]),
                int(profile["velocity_range"][1] * modifier["velocity_mult"])
            )
            profile["timing_humanize"] *= modifier["timing_humanize_mult"]
            profile["hat_probability"] *= modifier["hat_probability_mult"]
            if "swing_override" in modifier:
                profile["swing_amount"] = modifier["swing_override"]
        
        # Select groove and density
        groove = config.groove_override or random.choice(profile["grooves"])
        density = config.density_override or profile["density"]
        
        # Calculate timing
        beats_per_bar = config.time_signature[0]
        ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
        total_ticks = ticks_per_bar * config.bars
        
        all_hits = []
        instruments_used = set()
        
        # Generate each bar
        for bar in range(config.bars):
            bar_start = bar * ticks_per_bar
            
            # Check for fill
            is_fill_bar = (
                config.include_fills and
                random.random() < profile["fill_probability"] and
                (config.fill_on_bar is None or bar in config.fill_on_bar)
            )
            
            if is_fill_bar and bar == config.bars - 1:
                # Fill on last beat of bar
                fill_start = bar_start + ticks_per_bar - TICKS_PER_BEAT
                fill_hits = generate_fill(fill_start, TICKS_PER_BEAT, profile)
                
                for tick, instrument, vel in fill_hits:
                    hit = DrumHit(
                        instrument=instrument,
                        midi_note=GM_DRUMS.get(instrument, 38),
                        start_tick=tick,
                        duration_ticks=TICKS_PER_BEAT // 4,
                        velocity=vel,
                        is_ghost=False,
                        timing_offset=0,
                    )
                    all_hits.append(hit)
                    instruments_used.add(instrument)
            
            # Base patterns (16th note grid)
            kick_pattern = profile["kick_pattern"]
            snare_pattern = profile["snare_pattern"]
            
            sixteenth = TICKS_PER_BEAT // 4
            
            # Kick
            for i, hit in enumerate(kick_pattern):
                if hit and random.random() > profile["drop_probability"]:
                    tick = bar_start + i * sixteenth
                    tick = apply_swing(tick, TICKS_PER_BEAT, profile["swing_amount"])
                    
                    # Humanize timing
                    timing_offset = int(
                        random.gauss(0, profile["timing_humanize"] * TICKS_PER_BEAT)
                    )
                    
                    # Velocity
                    vel_min, vel_max = profile["velocity_range"]
                    velocity = random.randint(vel_min, vel_max)
                    
                    # Accent beat 1
                    if i == 0:
                        velocity = min(127, velocity + 15)
                    
                    drum_hit = DrumHit(
                        instrument="kick",
                        midi_note=GM_DRUMS["kick"],
                        start_tick=tick,
                        duration_ticks=sixteenth,
                        velocity=velocity,
                        is_ghost=False,
                        timing_offset=timing_offset,
                    )
                    all_hits.append(drum_hit)
                    instruments_used.add("kick")
            
            # Snare
            for i, hit in enumerate(snare_pattern):
                if hit and random.random() > profile["drop_probability"]:
                    tick = bar_start + i * sixteenth
                    tick = apply_swing(tick, TICKS_PER_BEAT, profile["swing_amount"])
                    
                    timing_offset = int(
                        random.gauss(0, profile["timing_humanize"] * TICKS_PER_BEAT)
                    )
                    
                    vel_min, vel_max = profile["velocity_range"]
                    velocity = random.randint(vel_min, vel_max)
                    
                    # Backbeat accent
                    if i in [4, 12]:  # Beats 2 and 4
                        velocity = min(127, velocity + 10)
                    
                    drum_hit = DrumHit(
                        instrument="snare",
                        midi_note=GM_DRUMS["snare"],
                        start_tick=tick,
                        duration_ticks=sixteenth,
                        velocity=velocity,
                        is_ghost=False,
                        timing_offset=timing_offset,
                    )
                    all_hits.append(drum_hit)
                    instruments_used.add("snare")
            
            # Hi-hats
            if groove == GrooveType.TRAP:
                hat_hits = generate_trap_hats(ticks_per_bar, profile, config.time_signature)
                for tick, instrument, vel in hat_hits:
                    actual_tick = bar_start + tick
                    timing_offset = int(
                        random.gauss(0, profile["timing_humanize"] * TICKS_PER_BEAT * 0.5)
                    )
                    
                    drum_hit = DrumHit(
                        instrument=instrument,
                        midi_note=GM_DRUMS[instrument],
                        start_tick=actual_tick,
                        duration_ticks=TICKS_PER_BEAT // 8,
                        velocity=vel,
                        is_ghost=False,
                        timing_offset=timing_offset,
                    )
                    all_hits.append(drum_hit)
                    instruments_used.add(instrument)
            else:
                hat_hits = generate_hat_pattern(ticks_per_bar, profile, groove, config.time_signature)
                for tick, instrument in hat_hits:
                    actual_tick = bar_start + tick
                    actual_tick = apply_swing(actual_tick, TICKS_PER_BEAT, profile["swing_amount"])
                    
                    timing_offset = int(
                        random.gauss(0, profile["timing_humanize"] * TICKS_PER_BEAT)
                    )
                    
                    vel_min, vel_max = profile["velocity_range"]
                    velocity = random.randint(vel_min - 20, vel_max - 10)
                    velocity = max(30, velocity)
                    
                    drum_hit = DrumHit(
                        instrument=instrument,
                        midi_note=GM_DRUMS[instrument],
                        start_tick=actual_tick,
                        duration_ticks=sixteenth,
                        velocity=velocity,
                        is_ghost=False,
                        timing_offset=timing_offset,
                    )
                    all_hits.append(drum_hit)
                    instruments_used.add(instrument)
            
            # Ghost notes
            if random.random() < profile["ghost_note_prob"]:
                # Add ghost snare
                ghost_positions = [2, 6, 10, 14]  # 16th note positions
                ghost_pos = random.choice(ghost_positions)
                tick = bar_start + ghost_pos * sixteenth
                
                drum_hit = DrumHit(
                    instrument="snare",
                    midi_note=GM_DRUMS["snare"],
                    start_tick=tick,
                    duration_ticks=sixteenth,
                    velocity=random.randint(25, 45),
                    is_ghost=True,
                    timing_offset=0,
                )
                all_hits.append(drum_hit)
        
        # Sort by time
        all_hits.sort(key=lambda h: h.start_tick + h.timing_offset)
        
        return RhythmOutput(
            hits=all_hits,
            config=config,
            groove_used=groove,
            density_used=density,
            total_ticks=total_ticks,
            instruments_used=instruments_used,
        )
    
    def generate_intro(
        self,
        emotion: str,
        bars: int = 2,
        tempo_bpm: int = 120,
        genre: Optional[str] = None,
    ) -> RhythmOutput:
        """Generate sparse intro pattern."""
        return self.generate(
            emotion=emotion,
            bars=bars,
            tempo_bpm=tempo_bpm,
            genre=genre,
            density_override=PatternDensity.MINIMAL,
            include_fills=False,
        )
    
    def generate_buildup(
        self,
        emotion: str,
        bars: int = 4,
        tempo_bpm: int = 120,
        genre: Optional[str] = None,
    ) -> RhythmOutput:
        """Generate building pattern toward climax."""
        return self.generate(
            emotion=emotion,
            bars=bars,
            tempo_bpm=tempo_bpm,
            genre=genre,
            density_override=PatternDensity.BUSY,
            include_fills=True,
            fill_on_bar=[bars - 1],
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_grief_drums(bars: int = 4, tempo: int = 72) -> RhythmOutput:
    """Quick grief pattern."""
    return RhythmEngine().generate("grief", bars, tempo, "lo-fi")


def generate_driving_drums(bars: int = 4, tempo: int = 140) -> RhythmOutput:
    """Quick driving pattern."""
    return RhythmEngine().generate("rage", bars, tempo, "rock")


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    engine = RhythmEngine()
    
    print("=== RHYTHM ENGINE DEMO ===\n")
    
    emotions = ["grief", "hope", "rage", "anxiety"]
    
    for emotion in emotions:
        pattern = engine.generate(
            emotion=emotion,
            bars=2,
            tempo_bpm=100,
            genre="lo-fi",
        )
        
        print(f"--- {emotion.upper()} ---")
        print(f"Groove: {pattern.groove_used.value}")
        print(f"Density: {pattern.density_used.value}")
        print(f"Hits: {len(pattern.hits)}")
        print(f"Instruments: {pattern.instruments_used}")
        print()
