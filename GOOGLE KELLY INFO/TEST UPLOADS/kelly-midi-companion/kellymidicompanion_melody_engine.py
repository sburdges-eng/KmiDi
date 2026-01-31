"""
Melody Engine - Generates emotion-driven melodic lines.

Part of Kelly MIDI Companion Tier 1 MVP.

Philosophy: Melodies should reflect emotional contour, not just 
scale correctness. A grief melody descends and lingers. A hope 
melody reaches upward with hesitation. A rage melody attacks 
with angular intervals.

Integration:
    from kellymidicompanion_melody_engine import MelodyEngine, MelodyConfig
    
    engine = MelodyEngine()
    melody = engine.generate(
        emotion="grief",
        key="F",
        mode="major",
        bars=4,
        tempo_bpm=82
    )
    
    # Returns MelodyOutput with MIDI-ready note data
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

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}

# Ticks per beat (standard MIDI resolution)
TICKS_PER_BEAT = 480


# =============================================================================
# ENUMS
# =============================================================================

class ContourType(Enum):
    """Melodic contour shapes."""
    ASCENDING = "ascending"           # Climbing upward
    DESCENDING = "descending"         # Falling downward
    ARCH = "arch"                     # Up then down
    INVERSE_ARCH = "inverse_arch"     # Down then up
    STATIC = "static"                 # Hovering around center
    WAVE = "wave"                     # Oscillating
    SPIRAL_DOWN = "spiral_down"       # Gradual descent with oscillation
    SPIRAL_UP = "spiral_up"           # Gradual ascent with oscillation
    JAGGED = "jagged"                 # Angular, unpredictable
    COLLAPSE = "collapse"             # Sudden fall


class RhythmDensity(Enum):
    """How many notes per bar."""
    SPARSE = "sparse"       # 2-4 notes/bar
    MODERATE = "moderate"   # 4-8 notes/bar
    DENSE = "dense"         # 8-16 notes/bar
    FRANTIC = "frantic"     # 16+ notes/bar


class ArticulationType(Enum):
    """Note articulation style."""
    LEGATO = "legato"           # Connected, flowing
    STACCATO = "staccato"       # Short, detached
    TENUTO = "tenuto"           # Held full value
    MARCATO = "marcato"         # Accented
    BREATH = "breath"           # Natural pauses


# =============================================================================
# EMOTION TO MELODY MAPPING
# =============================================================================

EMOTION_MELODY_PROFILES = {
    # Grief: descending, sparse, legato, minor 2nds and 3rds
    "grief": {
        "contours": [ContourType.DESCENDING, ContourType.SPIRAL_DOWN, ContourType.COLLAPSE],
        "density": RhythmDensity.SPARSE,
        "articulation": ArticulationType.LEGATO,
        "interval_weights": {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.1, 5: 0.1, 7: 0.05},
        "rest_probability": 0.3,
        "octave_range": (0, 1),  # Stay low
        "velocity_range": (40, 80),
        "resolution_tendency": 0.3,  # Low - avoid resolution
        "chromatic_tendency": 0.2,
        "repetition_tendency": 0.4,  # Stuck in loops
    },
    
    # Hope: ascending with hesitation, moderate density
    "hope": {
        "contours": [ContourType.ASCENDING, ContourType.SPIRAL_UP, ContourType.ARCH],
        "density": RhythmDensity.MODERATE,
        "articulation": ArticulationType.TENUTO,
        "interval_weights": {2: 0.3, 3: 0.25, 4: 0.2, 5: 0.15, 7: 0.1},
        "rest_probability": 0.2,
        "octave_range": (0, 2),
        "velocity_range": (50, 90),
        "resolution_tendency": 0.6,
        "chromatic_tendency": 0.1,
        "repetition_tendency": 0.2,
    },
    
    # Rage: angular, dense, marcato, wide leaps
    "rage": {
        "contours": [ContourType.JAGGED, ContourType.ASCENDING, ContourType.ARCH],
        "density": RhythmDensity.DENSE,
        "articulation": ArticulationType.MARCATO,
        "interval_weights": {1: 0.1, 4: 0.2, 5: 0.2, 6: 0.15, 7: 0.2, 8: 0.15},
        "rest_probability": 0.1,
        "octave_range": (-1, 2),
        "velocity_range": (80, 127),
        "resolution_tendency": 0.2,
        "chromatic_tendency": 0.3,
        "repetition_tendency": 0.5,  # Obsessive
    },
    
    # Fear: erratic, chromatic, staccato
    "fear": {
        "contours": [ContourType.JAGGED, ContourType.COLLAPSE, ContourType.WAVE],
        "density": RhythmDensity.MODERATE,
        "articulation": ArticulationType.STACCATO,
        "interval_weights": {1: 0.3, 2: 0.2, 6: 0.2, 7: 0.15, 11: 0.15},
        "rest_probability": 0.35,
        "octave_range": (0, 2),
        "velocity_range": (30, 100),  # Wide dynamic swings
        "resolution_tendency": 0.1,
        "chromatic_tendency": 0.4,
        "repetition_tendency": 0.3,
    },
    
    # Joy: bouncy, bright, leaping 3rds and 5ths
    "joy": {
        "contours": [ContourType.ARCH, ContourType.WAVE, ContourType.ASCENDING],
        "density": RhythmDensity.MODERATE,
        "articulation": ArticulationType.STACCATO,
        "interval_weights": {2: 0.2, 3: 0.25, 4: 0.2, 5: 0.2, 7: 0.15},
        "rest_probability": 0.15,
        "octave_range": (0, 2),
        "velocity_range": (70, 110),
        "resolution_tendency": 0.7,
        "chromatic_tendency": 0.05,
        "repetition_tendency": 0.2,
    },
    
    # Longing: reaching upward, falling back, suspended
    "longing": {
        "contours": [ContourType.SPIRAL_UP, ContourType.ARCH, ContourType.INVERSE_ARCH],
        "density": RhythmDensity.SPARSE,
        "articulation": ArticulationType.LEGATO,
        "interval_weights": {2: 0.2, 4: 0.25, 5: 0.2, 7: 0.2, 9: 0.15},
        "rest_probability": 0.25,
        "octave_range": (0, 2),
        "velocity_range": (45, 85),
        "resolution_tendency": 0.2,
        "chromatic_tendency": 0.15,
        "repetition_tendency": 0.35,
    },
    
    # Anxiety: restless, repetitive, chromatic neighbor tones
    "anxiety": {
        "contours": [ContourType.WAVE, ContourType.STATIC, ContourType.JAGGED],
        "density": RhythmDensity.DENSE,
        "articulation": ArticulationType.STACCATO,
        "interval_weights": {1: 0.4, 2: 0.3, 3: 0.15, 4: 0.1, 5: 0.05},
        "rest_probability": 0.1,
        "octave_range": (0, 1),
        "velocity_range": (50, 90),
        "resolution_tendency": 0.1,
        "chromatic_tendency": 0.5,
        "repetition_tendency": 0.6,
    },
    
    # Tenderness: gentle, stepwise, breathing
    "tenderness": {
        "contours": [ContourType.WAVE, ContourType.ARCH, ContourType.STATIC],
        "density": RhythmDensity.SPARSE,
        "articulation": ArticulationType.BREATH,
        "interval_weights": {1: 0.1, 2: 0.35, 3: 0.3, 4: 0.15, 5: 0.1},
        "rest_probability": 0.3,
        "octave_range": (0, 1),
        "velocity_range": (35, 70),
        "resolution_tendency": 0.5,
        "chromatic_tendency": 0.1,
        "repetition_tendency": 0.25,
    },
    
    # Defiance: strong, angular, syncopated feel
    "defiance": {
        "contours": [ContourType.ASCENDING, ContourType.JAGGED, ContourType.ARCH],
        "density": RhythmDensity.MODERATE,
        "articulation": ArticulationType.MARCATO,
        "interval_weights": {3: 0.2, 4: 0.25, 5: 0.25, 7: 0.2, 8: 0.1},
        "rest_probability": 0.2,
        "octave_range": (0, 2),
        "velocity_range": (70, 115),
        "resolution_tendency": 0.4,
        "chromatic_tendency": 0.15,
        "repetition_tendency": 0.3,
    },
    
    # Melancholy: wandering, minor 2nds, unresolved
    "melancholy": {
        "contours": [ContourType.WAVE, ContourType.DESCENDING, ContourType.SPIRAL_DOWN],
        "density": RhythmDensity.SPARSE,
        "articulation": ArticulationType.LEGATO,
        "interval_weights": {1: 0.25, 2: 0.3, 3: 0.2, 4: 0.15, 5: 0.1},
        "rest_probability": 0.25,
        "octave_range": (-1, 1),
        "velocity_range": (40, 75),
        "resolution_tendency": 0.25,
        "chromatic_tendency": 0.25,
        "repetition_tendency": 0.35,
    },
    
    # Nostalgia: familiar patterns, gentle, bittersweet
    "nostalgia": {
        "contours": [ContourType.ARCH, ContourType.WAVE, ContourType.INVERSE_ARCH],
        "density": RhythmDensity.MODERATE,
        "articulation": ArticulationType.TENUTO,
        "interval_weights": {2: 0.25, 3: 0.3, 4: 0.2, 5: 0.15, 7: 0.1},
        "rest_probability": 0.2,
        "octave_range": (0, 1),
        "velocity_range": (50, 80),
        "resolution_tendency": 0.5,
        "chromatic_tendency": 0.1,
        "repetition_tendency": 0.4,  # Callbacks
    },
    
    # Euphoria: soaring, wide range, triumphant
    "euphoria": {
        "contours": [ContourType.ASCENDING, ContourType.SPIRAL_UP, ContourType.ARCH],
        "density": RhythmDensity.MODERATE,
        "articulation": ArticulationType.LEGATO,
        "interval_weights": {3: 0.2, 4: 0.2, 5: 0.25, 7: 0.2, 8: 0.15},
        "rest_probability": 0.1,
        "octave_range": (0, 3),
        "velocity_range": (80, 120),
        "resolution_tendency": 0.8,
        "chromatic_tendency": 0.05,
        "repetition_tendency": 0.15,
    },
    
    # Dissociation: fragmented, disconnected, gaps
    "dissociation": {
        "contours": [ContourType.STATIC, ContourType.JAGGED, ContourType.COLLAPSE],
        "density": RhythmDensity.SPARSE,
        "articulation": ArticulationType.STACCATO,
        "interval_weights": {1: 0.2, 5: 0.2, 7: 0.2, 8: 0.2, 11: 0.2},
        "rest_probability": 0.5,
        "octave_range": (-1, 2),
        "velocity_range": (30, 60),
        "resolution_tendency": 0.1,
        "chromatic_tendency": 0.3,
        "repetition_tendency": 0.2,
    },
    
    # Neutral/default
    "neutral": {
        "contours": [ContourType.WAVE, ContourType.ARCH, ContourType.STATIC],
        "density": RhythmDensity.MODERATE,
        "articulation": ArticulationType.TENUTO,
        "interval_weights": {2: 0.3, 3: 0.25, 4: 0.2, 5: 0.15, 7: 0.1},
        "rest_probability": 0.2,
        "octave_range": (0, 1),
        "velocity_range": (60, 90),
        "resolution_tendency": 0.5,
        "chromatic_tendency": 0.1,
        "repetition_tendency": 0.2,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MelodyNote:
    """A single note in a melody."""
    pitch: int              # MIDI pitch (0-127)
    start_tick: int         # Start time in ticks
    duration_ticks: int     # Duration in ticks
    velocity: int           # MIDI velocity (0-127)
    scale_degree: int       # Scale degree (1-7, 0 for chromatic)
    interval_from_prev: int # Semitones from previous note
    is_chromatic: bool      # Outside the scale
    articulation: ArticulationType


@dataclass
class MelodyConfig:
    """Configuration for melody generation."""
    emotion: str = "neutral"
    key: str = "C"
    mode: str = "major"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    octave_base: int = 4          # Middle C octave
    seed: Optional[int] = None    # For reproducibility
    contour_override: Optional[ContourType] = None
    density_override: Optional[RhythmDensity] = None
    starting_pitch: Optional[int] = None
    ending_pitch: Optional[int] = None
    chord_tones: Optional[List[int]] = None  # Emphasize these
    avoid_pitches: Optional[List[int]] = None
    phrase_length_bars: int = 2   # How long before natural break


@dataclass
class MelodyOutput:
    """Output from melody generation."""
    notes: List[MelodyNote]
    config: MelodyConfig
    contour_used: ContourType
    density_used: RhythmDensity
    articulation_used: ArticulationType
    total_ticks: int
    pitch_range: Tuple[int, int]  # (lowest, highest)
    
    def to_midi_events(self) -> List[Dict]:
        """Convert to MIDI event list for mido."""
        events = []
        for note in self.notes:
            events.append({
                "type": "note_on",
                "note": note.pitch,
                "velocity": note.velocity,
                "time": note.start_tick,
            })
            events.append({
                "type": "note_off",
                "note": note.pitch,
                "velocity": 0,
                "time": note.start_tick + note.duration_ticks,
            })
        return sorted(events, key=lambda e: e["time"])
    
    def to_note_list(self) -> List[Tuple[int, int, int, int]]:
        """Return as list of (pitch, start_tick, duration, velocity)."""
        return [
            (n.pitch, n.start_tick, n.duration_ticks, n.velocity)
            for n in self.notes
        ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def note_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.upper().replace("B", "B").replace("S", "#")
    # Handle flats
    flat_map = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    if note in flat_map:
        note = flat_map[note]
    
    if note in CHROMATIC:
        return CHROMATIC.index(note) + (octave + 1) * 12
    return 60  # Default to middle C


def get_scale_pitches(key: str, mode: str, octave_base: int, octave_range: Tuple[int, int]) -> List[int]:
    """Get all pitches in scale across octave range."""
    intervals = SCALE_INTERVALS.get(mode.lower(), SCALE_INTERVALS["major"])
    root = note_to_midi(key, 0) % 12
    
    pitches = []
    for octave_offset in range(octave_range[0], octave_range[1] + 1):
        octave = octave_base + octave_offset
        for interval in intervals:
            pitch = root + interval + (octave + 1) * 12
            if 0 <= pitch <= 127:
                pitches.append(pitch)
    
    return sorted(pitches)


def weighted_choice(weights: Dict[int, float]) -> int:
    """Choose from weighted options."""
    items = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    
    r = random.random()
    cumulative = 0
    for item, prob in zip(items, probs):
        cumulative += prob
        if r <= cumulative:
            return item
    return items[-1]


def generate_contour_targets(
    contour: ContourType,
    num_points: int,
    octave_range: Tuple[int, int]
) -> List[float]:
    """
    Generate target positions (0.0 to 1.0) for each point in melody.
    0.0 = bottom of range, 1.0 = top of range.
    """
    targets = []
    
    for i in range(num_points):
        t = i / max(1, num_points - 1)  # 0.0 to 1.0
        
        if contour == ContourType.ASCENDING:
            target = t * 0.7 + 0.15
        
        elif contour == ContourType.DESCENDING:
            target = (1 - t) * 0.7 + 0.15
        
        elif contour == ContourType.ARCH:
            # Parabola peaking at middle
            target = -4 * (t - 0.5) ** 2 + 1
            target = target * 0.6 + 0.2
        
        elif contour == ContourType.INVERSE_ARCH:
            target = 4 * (t - 0.5) ** 2
            target = target * 0.6 + 0.2
        
        elif contour == ContourType.STATIC:
            target = 0.5 + random.uniform(-0.1, 0.1)
        
        elif contour == ContourType.WAVE:
            target = 0.5 + 0.3 * math.sin(t * math.pi * 2)
        
        elif contour == ContourType.SPIRAL_DOWN:
            base = (1 - t) * 0.6 + 0.2
            wave = 0.15 * math.sin(t * math.pi * 4)
            target = base + wave
        
        elif contour == ContourType.SPIRAL_UP:
            base = t * 0.6 + 0.2
            wave = 0.15 * math.sin(t * math.pi * 4)
            target = base + wave
        
        elif contour == ContourType.JAGGED:
            target = random.uniform(0.2, 0.8)
        
        elif contour == ContourType.COLLAPSE:
            if t < 0.7:
                target = 0.6 + random.uniform(-0.1, 0.1)
            else:
                collapse_t = (t - 0.7) / 0.3
                target = 0.6 - collapse_t * 0.5
        
        else:
            target = 0.5
        
        targets.append(max(0.0, min(1.0, target)))
    
    return targets


def get_rhythm_pattern(
    density: RhythmDensity,
    bars: int,
    time_sig: Tuple[int, int],
    rest_probability: float
) -> List[Tuple[int, int]]:
    """
    Generate rhythm pattern as list of (start_tick, duration_ticks).
    """
    beats_per_bar = time_sig[0]
    ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
    total_ticks = ticks_per_bar * bars
    
    # Notes per bar based on density
    if density == RhythmDensity.SPARSE:
        notes_per_bar = random.randint(2, 4)
    elif density == RhythmDensity.MODERATE:
        notes_per_bar = random.randint(4, 8)
    elif density == RhythmDensity.DENSE:
        notes_per_bar = random.randint(8, 12)
    else:  # FRANTIC
        notes_per_bar = random.randint(12, 16)
    
    # Build rhythm
    rhythm = []
    current_tick = 0
    
    while current_tick < total_ticks:
        # Decide if rest
        if random.random() < rest_probability and len(rhythm) > 0:
            # Skip ahead
            skip = random.choice([
                TICKS_PER_BEAT // 4,
                TICKS_PER_BEAT // 2,
                TICKS_PER_BEAT,
            ])
            current_tick += skip
            continue
        
        # Duration options
        duration_options = [
            TICKS_PER_BEAT // 4,   # 16th
            TICKS_PER_BEAT // 2,   # 8th
            TICKS_PER_BEAT,        # Quarter
            TICKS_PER_BEAT * 2,    # Half
        ]
        
        # Weight by density
        if density == RhythmDensity.SPARSE:
            weights = [0.1, 0.2, 0.4, 0.3]
        elif density == RhythmDensity.MODERATE:
            weights = [0.2, 0.4, 0.3, 0.1]
        elif density == RhythmDensity.DENSE:
            weights = [0.4, 0.4, 0.15, 0.05]
        else:
            weights = [0.6, 0.3, 0.1, 0.0]
        
        duration = random.choices(duration_options, weights=weights)[0]
        
        # Don't exceed total
        if current_tick + duration > total_ticks:
            duration = total_ticks - current_tick
        
        if duration > 0:
            rhythm.append((current_tick, duration))
        
        current_tick += duration
    
    return rhythm


def apply_articulation(
    duration: int,
    articulation: ArticulationType
) -> int:
    """Modify duration based on articulation."""
    if articulation == ArticulationType.STACCATO:
        return max(TICKS_PER_BEAT // 8, int(duration * 0.4))
    elif articulation == ArticulationType.LEGATO:
        return int(duration * 0.95)
    elif articulation == ArticulationType.TENUTO:
        return duration
    elif articulation == ArticulationType.MARCATO:
        return max(TICKS_PER_BEAT // 4, int(duration * 0.6))
    elif articulation == ArticulationType.BREATH:
        return max(TICKS_PER_BEAT // 4, int(duration * 0.8))
    return duration


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class MelodyEngine:
    """
    Generates emotion-driven melodic lines.
    
    Usage:
        engine = MelodyEngine()
        output = engine.generate(
            emotion="grief",
            key="F",
            mode="major",
            bars=4,
            tempo_bpm=82
        )
    """
    
    def __init__(self, custom_profiles: Optional[Dict] = None):
        """
        Initialize melody engine.
        
        Args:
            custom_profiles: Override or extend EMOTION_MELODY_PROFILES
        """
        self.profiles = EMOTION_MELODY_PROFILES.copy()
        if custom_profiles:
            self.profiles.update(custom_profiles)
    
    def generate(
        self,
        emotion: str = "neutral",
        key: str = "C",
        mode: str = "major",
        bars: int = 4,
        tempo_bpm: int = 120,
        **kwargs
    ) -> MelodyOutput:
        """
        Generate a melody based on emotional parameters.
        
        Args:
            emotion: Emotional state (grief, hope, rage, etc.)
            key: Musical key (C, F#, Bb, etc.)
            mode: Scale mode (major, minor, dorian, etc.)
            bars: Number of bars
            tempo_bpm: Tempo in BPM
            **kwargs: Additional MelodyConfig parameters
        
        Returns:
            MelodyOutput with generated notes
        """
        config = MelodyConfig(
            emotion=emotion.lower(),
            key=key,
            mode=mode.lower(),
            bars=bars,
            tempo_bpm=tempo_bpm,
            **kwargs
        )
        
        return self._generate_from_config(config)
    
    def _generate_from_config(self, config: MelodyConfig) -> MelodyOutput:
        """Internal generation from config."""
        
        # Set random seed if provided
        if config.seed is not None:
            random.seed(config.seed)
        
        # Get emotion profile
        profile = self.profiles.get(config.emotion, self.profiles["neutral"])
        
        # Select contour
        contour = config.contour_override or random.choice(profile["contours"])
        
        # Select density
        density = config.density_override or profile["density"]
        
        # Get articulation
        articulation = profile["articulation"]
        
        # Get scale pitches
        scale_pitches = get_scale_pitches(
            config.key,
            config.mode,
            config.octave_base,
            profile["octave_range"]
        )
        
        if not scale_pitches:
            # Fallback
            scale_pitches = list(range(48, 84))
        
        # Generate rhythm pattern
        rhythm = get_rhythm_pattern(
            density,
            config.bars,
            config.time_signature,
            profile["rest_probability"]
        )
        
        if not rhythm:
            # Ensure at least one note
            rhythm = [(0, TICKS_PER_BEAT)]
        
        # Generate contour targets
        contour_targets = generate_contour_targets(
            contour,
            len(rhythm),
            profile["octave_range"]
        )
        
        # Generate notes
        notes = []
        prev_pitch = config.starting_pitch or scale_pitches[len(scale_pitches) // 2]
        
        for i, ((start, dur), target) in enumerate(zip(rhythm, contour_targets)):
            # Target pitch from contour
            target_idx = int(target * (len(scale_pitches) - 1))
            target_pitch = scale_pitches[target_idx]
            
            # Decide on interval
            interval = weighted_choice(profile["interval_weights"])
            
            # Direction toward target
            if target_pitch > prev_pitch:
                direction = 1
            elif target_pitch < prev_pitch:
                direction = -1
            else:
                direction = random.choice([-1, 1])
            
            # Calculate new pitch
            new_pitch = prev_pitch + (interval * direction)
            
            # Chromatic or diatonic?
            is_chromatic = False
            if random.random() < profile["chromatic_tendency"]:
                # Allow chromatic
                is_chromatic = new_pitch not in scale_pitches
            else:
                # Snap to scale
                if new_pitch not in scale_pitches:
                    # Find nearest scale pitch
                    distances = [(abs(p - new_pitch), p) for p in scale_pitches]
                    new_pitch = min(distances)[1]
            
            # Clamp to valid range
            new_pitch = max(scale_pitches[0], min(scale_pitches[-1], new_pitch))
            
            # Repetition tendency
            if random.random() < profile["repetition_tendency"] and len(notes) > 0:
                new_pitch = prev_pitch
            
            # Avoid specific pitches
            if config.avoid_pitches and new_pitch in config.avoid_pitches:
                alternatives = [p for p in scale_pitches if p not in config.avoid_pitches]
                if alternatives:
                    new_pitch = min(alternatives, key=lambda p: abs(p - new_pitch))
            
            # Chord tone emphasis
            if config.chord_tones and random.random() < 0.4:
                if any(ct % 12 == new_pitch % 12 for ct in config.chord_tones):
                    pass  # Already a chord tone
                else:
                    # Try to land on chord tone
                    chord_matches = [p for p in scale_pitches 
                                   if any(ct % 12 == p % 12 for ct in config.chord_tones)]
                    if chord_matches:
                        new_pitch = min(chord_matches, key=lambda p: abs(p - prev_pitch))
            
            # Velocity from profile with variation
            vel_min, vel_max = profile["velocity_range"]
            velocity = random.randint(vel_min, vel_max)
            
            # Accent on strong beats
            beat_position = start % (TICKS_PER_BEAT * config.time_signature[0])
            if beat_position < TICKS_PER_BEAT // 2:  # Near downbeat
                velocity = min(127, velocity + 10)
            
            # Apply articulation to duration
            actual_duration = apply_articulation(dur, articulation)
            
            # Calculate scale degree
            root_pitch = note_to_midi(config.key, 0) % 12
            scale_degree = ((new_pitch % 12) - root_pitch) % 12
            scale_degrees = SCALE_INTERVALS.get(config.mode, [0, 2, 4, 5, 7, 9, 11])
            if scale_degree in scale_degrees:
                scale_degree_num = scale_degrees.index(scale_degree) + 1
            else:
                scale_degree_num = 0  # Chromatic
            
            # Create note
            note = MelodyNote(
                pitch=new_pitch,
                start_tick=start,
                duration_ticks=actual_duration,
                velocity=velocity,
                scale_degree=scale_degree_num,
                interval_from_prev=abs(new_pitch - prev_pitch) if notes else 0,
                is_chromatic=is_chromatic,
                articulation=articulation,
            )
            
            notes.append(note)
            prev_pitch = new_pitch
        
        # Handle ending pitch preference
        if config.ending_pitch is not None and notes:
            # Adjust last note
            notes[-1] = MelodyNote(
                pitch=config.ending_pitch,
                start_tick=notes[-1].start_tick,
                duration_ticks=notes[-1].duration_ticks,
                velocity=notes[-1].velocity,
                scale_degree=notes[-1].scale_degree,
                interval_from_prev=abs(config.ending_pitch - notes[-2].pitch) if len(notes) > 1 else 0,
                is_chromatic=config.ending_pitch not in scale_pitches,
                articulation=notes[-1].articulation,
            )
        
        # Resolution tendency - maybe end on root/5th
        if notes and random.random() < profile["resolution_tendency"]:
            root = note_to_midi(config.key, config.octave_base)
            fifth = root + 7
            # Nudge last note toward resolution
            last = notes[-1]
            if abs(last.pitch - root) < abs(last.pitch - fifth):
                resolve_to = root
            else:
                resolve_to = fifth
            
            notes[-1] = MelodyNote(
                pitch=resolve_to,
                start_tick=last.start_tick,
                duration_ticks=last.duration_ticks,
                velocity=last.velocity,
                scale_degree=1 if resolve_to == root else 5,
                interval_from_prev=abs(resolve_to - notes[-2].pitch) if len(notes) > 1 else 0,
                is_chromatic=False,
                articulation=last.articulation,
            )
        
        # Calculate totals
        total_ticks = config.bars * config.time_signature[0] * TICKS_PER_BEAT
        pitches = [n.pitch for n in notes]
        pitch_range = (min(pitches), max(pitches)) if pitches else (60, 60)
        
        return MelodyOutput(
            notes=notes,
            config=config,
            contour_used=contour,
            density_used=density,
            articulation_used=articulation,
            total_ticks=total_ticks,
            pitch_range=pitch_range,
        )
    
    def generate_phrase(
        self,
        emotion: str,
        key: str,
        mode: str,
        phrase_bars: int = 2,
        tempo_bpm: int = 120,
        ending: str = "open"  # "open", "closed", "suspended"
    ) -> MelodyOutput:
        """
        Generate a single musical phrase.
        
        Args:
            ending: How to end the phrase
                - "open": Don't resolve
                - "closed": Resolve to tonic
                - "suspended": End on 5th or 2nd
        """
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])
        
        # Determine ending pitch
        root = note_to_midi(key, 4)
        
        if ending == "closed":
            ending_pitch = root
        elif ending == "suspended":
            ending_pitch = root + 7  # 5th
        else:
            ending_pitch = None  # Let it be
        
        return self.generate(
            emotion=emotion,
            key=key,
            mode=mode,
            bars=phrase_bars,
            tempo_bpm=tempo_bpm,
            ending_pitch=ending_pitch,
        )
    
    def generate_call_response(
        self,
        emotion: str,
        key: str,
        mode: str,
        call_bars: int = 2,
        response_bars: int = 2,
        tempo_bpm: int = 120,
    ) -> Tuple[MelodyOutput, MelodyOutput]:
        """
        Generate a call-and-response melodic pair.
        
        Returns:
            Tuple of (call_melody, response_melody)
        """
        # Call ends open
        call = self.generate_phrase(
            emotion=emotion,
            key=key,
            mode=mode,
            phrase_bars=call_bars,
            tempo_bpm=tempo_bpm,
            ending="open"
        )
        
        # Response ends closed
        response_start = call.notes[-1].pitch if call.notes else note_to_midi(key, 4)
        response = self.generate(
            emotion=emotion,
            key=key,
            mode=mode,
            bars=response_bars,
            tempo_bpm=tempo_bpm,
            starting_pitch=response_start,
            ending_pitch=note_to_midi(key, 4),  # Resolve to root
        )
        
        return call, response


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_grief_melody(key: str = "F", bars: int = 4, tempo: int = 72) -> MelodyOutput:
    """Quick grief melody generation."""
    return MelodyEngine().generate("grief", key, "minor", bars, tempo)


def generate_hope_melody(key: str = "C", bars: int = 4, tempo: int = 90) -> MelodyOutput:
    """Quick hope melody generation."""
    return MelodyEngine().generate("hope", key, "major", bars, tempo)


def generate_rage_melody(key: str = "E", bars: int = 4, tempo: int = 140) -> MelodyOutput:
    """Quick rage melody generation."""
    return MelodyEngine().generate("rage", key, "phrygian", bars, tempo)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demo generation
    engine = MelodyEngine()
    
    print("=== MELODY ENGINE DEMO ===\n")
    
    emotions = ["grief", "hope", "rage", "anxiety", "tenderness"]
    
    for emotion in emotions:
        melody = engine.generate(
            emotion=emotion,
            key="F",
            mode="minor" if emotion in ["grief", "rage", "anxiety"] else "major",
            bars=4,
            tempo_bpm=82,
        )
        
        print(f"--- {emotion.upper()} ---")
        print(f"Contour: {melody.contour_used.value}")
        print(f"Density: {melody.density_used.value}")
        print(f"Notes: {len(melody.notes)}")
        print(f"Pitch range: {melody.pitch_range}")
        print(f"First 5 pitches: {[n.pitch for n in melody.notes[:5]]}")
        print()
