"""
Counter-Melody Engine - Generates secondary melodic lines that complement the main melody.

Part of Kelly MIDI Companion Tier 2.

Philosophy: A counter-melody is a conversation. It answers, echoes, or
contrasts the main melody. A grief counter-melody might mirror and
descend. A hopeful counter-melody might answer and rise. The relationship
between voices IS the emotional dialogue.

Techniques:
    - Parallel motion (following at interval)
    - Contrary motion (opposite direction)
    - Oblique motion (one voice stationary)
    - Call and response (echo/answer patterns)
    - Pedal (sustained note against moving melody)
    - Canon (delayed imitation)

Integration:
    from kellymidicompanion_counter_melody_engine import CounterMelodyEngine
    
    engine = CounterMelodyEngine()
    counter = engine.generate(
        emotion="grief",
        main_melody=melody_notes,  # From MelodyEngine
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

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

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
}

# GM instruments for counter-melodies
GM_COUNTER_INSTRUMENTS = {
    "flute": 73,
    "oboe": 68,
    "clarinet": 71,
    "english_horn": 69,
    "bassoon": 70,
    "french_horn": 60,
    "strings": 48,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "synth_lead": 80,
    "voice_oohs": 53,
}


# =============================================================================
# ENUMS
# =============================================================================

class CounterMotion(Enum):
    """Relationship to main melody."""
    PARALLEL = "parallel"           # Same direction, fixed interval
    CONTRARY = "contrary"           # Opposite direction
    OBLIQUE = "oblique"             # One voice held, other moves
    SIMILAR = "similar"             # Same direction, varying interval
    FREE = "free"                   # Independent motion


class CounterTechnique(Enum):
    """Compositional technique."""
    THIRDS = "thirds"               # Parallel thirds
    SIXTHS = "sixths"               # Parallel sixths
    TENTHS = "tenths"               # Parallel tenths (3rd + octave)
    CALL_RESPONSE = "call_response" # Echo/answer
    CANON = "canon"                 # Delayed imitation
    PEDAL = "pedal"                 # Sustained note
    FILL = "fill"                   # Fill spaces in melody
    SHADOW = "shadow"               # Slightly delayed shadow
    INVERSION = "inversion"         # Mirror image
    COUNTERPOINT = "counterpoint"   # Traditional counterpoint


class CounterRhythm(Enum):
    """Rhythmic relationship."""
    SYNC = "sync"                   # Same rhythm as melody
    OFFSET = "offset"               # Slightly delayed
    DOUBLE = "double"               # Twice as fast
    HALF = "half"                   # Half as fast
    COMPLEMENT = "complement"       # Fills gaps
    INDEPENDENT = "independent"     # Own rhythm


class CounterRegister(Enum):
    """Pitch relationship to melody."""
    ABOVE = "above"                 # Higher than melody
    BELOW = "below"                 # Lower than melody
    WEAVING = "weaving"             # Crosses above and below
    OCTAVE_ABOVE = "octave_above"   # Full octave higher
    OCTAVE_BELOW = "octave_below"   # Full octave lower


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_PROFILES = {
    "grief": {
        "motion": CounterMotion.CONTRARY,
        "technique": CounterTechnique.SIXTHS,
        "rhythm": CounterRhythm.SYNC,
        "register": CounterRegister.BELOW,
        "interval_preference": [-8, -5, -3],  # Minor 6th, 4th, 3rd below
        "velocity_ratio": 0.7,       # Softer than main
        "density": 0.6,              # Not every note
        "delay_ticks": 0,
        "humanize_timing": 20,
        "humanize_velocity": 10,
        "sustain_ratio": 0.9,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["cello"],
    },
    
    "sadness": {
        "motion": CounterMotion.SIMILAR,
        "technique": CounterTechnique.SHADOW,
        "rhythm": CounterRhythm.OFFSET,
        "register": CounterRegister.BELOW,
        "interval_preference": [-5, -7, -3],
        "velocity_ratio": 0.65,
        "density": 0.5,
        "delay_ticks": TICKS_PER_BEAT // 4,
        "humanize_timing": 15,
        "humanize_velocity": 8,
        "sustain_ratio": 0.85,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["viola"],
    },
    
    "hope": {
        "motion": CounterMotion.PARALLEL,
        "technique": CounterTechnique.THIRDS,
        "rhythm": CounterRhythm.SYNC,
        "register": CounterRegister.ABOVE,
        "interval_preference": [3, 4, 7],  # 3rd, M3rd, 5th above
        "velocity_ratio": 0.8,
        "density": 0.7,
        "delay_ticks": 0,
        "humanize_timing": 10,
        "humanize_velocity": 8,
        "sustain_ratio": 0.9,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["flute"],
    },
    
    "joy": {
        "motion": CounterMotion.PARALLEL,
        "technique": CounterTechnique.THIRDS,
        "rhythm": CounterRhythm.SYNC,
        "register": CounterRegister.ABOVE,
        "interval_preference": [4, 7, 12],
        "velocity_ratio": 0.9,
        "density": 0.85,
        "delay_ticks": 0,
        "humanize_timing": 5,
        "humanize_velocity": 10,
        "sustain_ratio": 0.8,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["oboe"],
    },
    
    "rage": {
        "motion": CounterMotion.PARALLEL,
        "technique": CounterTechnique.TENTHS,
        "rhythm": CounterRhythm.SYNC,
        "register": CounterRegister.OCTAVE_BELOW,
        "interval_preference": [-12, -15, -19],  # Octave, 10th, 12th below
        "velocity_ratio": 1.0,
        "density": 0.95,
        "delay_ticks": 0,
        "humanize_timing": 3,
        "humanize_velocity": 5,
        "sustain_ratio": 0.7,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["french_horn"],
    },
    
    "anger": {
        "motion": CounterMotion.CONTRARY,
        "technique": CounterTechnique.COUNTERPOINT,
        "rhythm": CounterRhythm.COMPLEMENT,
        "register": CounterRegister.BELOW,
        "interval_preference": [-5, -7, -12],
        "velocity_ratio": 0.9,
        "density": 0.7,
        "delay_ticks": 0,
        "humanize_timing": 5,
        "humanize_velocity": 8,
        "sustain_ratio": 0.6,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["bassoon"],
    },
    
    "fear": {
        "motion": CounterMotion.OBLIQUE,
        "technique": CounterTechnique.PEDAL,
        "rhythm": CounterRhythm.HALF,
        "register": CounterRegister.BELOW,
        "interval_preference": [-7, -5, 0],
        "velocity_ratio": 0.5,
        "density": 0.4,
        "delay_ticks": 0,
        "humanize_timing": 30,
        "humanize_velocity": 15,
        "sustain_ratio": 1.0,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["strings"],
    },
    
    "anxiety": {
        "motion": CounterMotion.FREE,
        "technique": CounterTechnique.FILL,
        "rhythm": CounterRhythm.COMPLEMENT,
        "register": CounterRegister.WEAVING,
        "interval_preference": [2, -2, 5, -5],
        "velocity_ratio": 0.6,
        "density": 0.5,
        "delay_ticks": TICKS_PER_BEAT // 8,
        "humanize_timing": 35,
        "humanize_velocity": 18,
        "sustain_ratio": 0.5,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["clarinet"],
    },
    
    "tension": {
        "motion": CounterMotion.CONTRARY,
        "technique": CounterTechnique.COUNTERPOINT,
        "rhythm": CounterRhythm.INDEPENDENT,
        "register": CounterRegister.BELOW,
        "interval_preference": [-1, -2, 1, 2],  # Dissonant
        "velocity_ratio": 0.75,
        "density": 0.6,
        "delay_ticks": 0,
        "humanize_timing": 15,
        "humanize_velocity": 12,
        "sustain_ratio": 0.95,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["english_horn"],
    },
    
    "longing": {
        "motion": CounterMotion.SIMILAR,
        "technique": CounterTechnique.CALL_RESPONSE,
        "rhythm": CounterRhythm.OFFSET,
        "register": CounterRegister.BELOW,
        "interval_preference": [-3, -5, -8],
        "velocity_ratio": 0.7,
        "density": 0.55,
        "delay_ticks": TICKS_PER_BEAT,
        "humanize_timing": 20,
        "humanize_velocity": 10,
        "sustain_ratio": 0.9,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["cello"],
    },
    
    "nostalgia": {
        "motion": CounterMotion.PARALLEL,
        "technique": CounterTechnique.CANON,
        "rhythm": CounterRhythm.OFFSET,
        "register": CounterRegister.BELOW,
        "interval_preference": [-5, -7, -12],
        "velocity_ratio": 0.6,
        "density": 0.5,
        "delay_ticks": TICKS_PER_BEAT * 2,
        "humanize_timing": 18,
        "humanize_velocity": 10,
        "sustain_ratio": 0.85,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["violin"],
    },
    
    "peace": {
        "motion": CounterMotion.OBLIQUE,
        "technique": CounterTechnique.PEDAL,
        "rhythm": CounterRhythm.HALF,
        "register": CounterRegister.BELOW,
        "interval_preference": [0, -7, 7],
        "velocity_ratio": 0.5,
        "density": 0.35,
        "delay_ticks": 0,
        "humanize_timing": 10,
        "humanize_velocity": 5,
        "sustain_ratio": 1.0,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["voice_oohs"],
    },
    
    "euphoria": {
        "motion": CounterMotion.PARALLEL,
        "technique": CounterTechnique.SIXTHS,
        "rhythm": CounterRhythm.SYNC,
        "register": CounterRegister.ABOVE,
        "interval_preference": [8, 9, 12],
        "velocity_ratio": 0.85,
        "density": 0.8,
        "delay_ticks": 0,
        "humanize_timing": 8,
        "humanize_velocity": 10,
        "sustain_ratio": 0.9,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["flute"],
    },
    
    "wonder": {
        "motion": CounterMotion.FREE,
        "technique": CounterTechnique.FILL,
        "rhythm": CounterRhythm.COMPLEMENT,
        "register": CounterRegister.WEAVING,
        "interval_preference": [7, 12, -5, -12],
        "velocity_ratio": 0.6,
        "density": 0.45,
        "delay_ticks": TICKS_PER_BEAT // 2,
        "humanize_timing": 20,
        "humanize_velocity": 12,
        "sustain_ratio": 0.75,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["oboe"],
    },
    
    "determination": {
        "motion": CounterMotion.PARALLEL,
        "technique": CounterTechnique.TENTHS,
        "rhythm": CounterRhythm.SYNC,
        "register": CounterRegister.BELOW,
        "interval_preference": [-15, -12, -8],
        "velocity_ratio": 0.95,
        "density": 0.9,
        "delay_ticks": 0,
        "humanize_timing": 3,
        "humanize_velocity": 5,
        "sustain_ratio": 0.85,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["french_horn"],
    },
    
    "neutral": {
        "motion": CounterMotion.PARALLEL,
        "technique": CounterTechnique.THIRDS,
        "rhythm": CounterRhythm.SYNC,
        "register": CounterRegister.BELOW,
        "interval_preference": [-3, -4, -5],
        "velocity_ratio": 0.75,
        "density": 0.6,
        "delay_ticks": 0,
        "humanize_timing": 10,
        "humanize_velocity": 8,
        "sustain_ratio": 0.85,
        "gm_instrument": GM_COUNTER_INSTRUMENTS["clarinet"],
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MelodyNote:
    """Input melody note (from MelodyEngine or similar)."""
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int


@dataclass
class CounterNote:
    """A single counter-melody note."""
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    source_note_idx: int         # Which melody note this relates to
    interval_from_source: int    # Semitones from source note
    technique_applied: CounterTechnique


@dataclass
class CounterMelodyConfig:
    """Configuration for counter-melody generation."""
    emotion: str = "neutral"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    motion_override: Optional[CounterMotion] = None
    technique_override: Optional[CounterTechnique] = None
    register_override: Optional[CounterRegister] = None


@dataclass
class CounterMelodyOutput:
    """Complete counter-melody output."""
    notes: List[CounterNote]
    emotion: str
    motion_used: CounterMotion
    technique_used: CounterTechnique
    rhythm_used: CounterRhythm
    register_used: CounterRegister
    gm_instrument: int
    total_ticks: int
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


def get_scale_pitches(root: str, mode: str, octave: int = 4) -> List[int]:
    """Get all pitches in a scale."""
    root_midi = note_name_to_midi(root, octave)
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    return [root_midi + i for i in intervals]


def snap_to_scale(pitch: int, scale_pitches: List[int]) -> int:
    """Snap a pitch to the nearest scale degree."""
    pitch_class = pitch % 12
    octave = pitch // 12
    
    scale_classes = [p % 12 for p in scale_pitches]
    
    # Find nearest scale degree
    min_dist = 12
    best_class = pitch_class
    for sc in scale_classes:
        dist = min(abs(pitch_class - sc), 12 - abs(pitch_class - sc))
        if dist < min_dist:
            min_dist = dist
            best_class = sc
    
    return octave * 12 + best_class


def calculate_interval(source_pitch: int, interval: int, register: CounterRegister) -> int:
    """Calculate target pitch based on interval and register preference."""
    target = source_pitch + interval
    
    # Adjust for register preference
    if register == CounterRegister.ABOVE and target <= source_pitch:
        target += 12
    elif register == CounterRegister.BELOW and target >= source_pitch:
        target -= 12
    elif register == CounterRegister.OCTAVE_ABOVE:
        target = source_pitch + abs(interval) + 12
    elif register == CounterRegister.OCTAVE_BELOW:
        target = source_pitch - abs(interval) - 12
    
    return target


def apply_contrary_motion(source_pitch: int, source_center: int, interval: int) -> int:
    """Apply contrary motion relative to a center pitch."""
    offset = source_pitch - source_center
    return source_center - offset + interval


def generate_pedal_pitch(scale_pitches: List[int], register: CounterRegister) -> int:
    """Generate a pedal tone pitch."""
    root = scale_pitches[0]
    fifth = root + 7
    
    if register in [CounterRegister.BELOW, CounterRegister.OCTAVE_BELOW]:
        return root - 12
    else:
        return fifth


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class CounterMelodyEngine:
    """
    Generates counter-melodies that complement a main melody.
    
    The counter-melody creates emotional dialogue through its
    relationship to the main voice.
    """
    
    def __init__(self):
        self.profiles = EMOTION_PROFILES
    
    def generate(
        self,
        emotion: str,
        main_melody: List[MelodyNote],
        chord_progression: List[str],
        key: str = "C",
        mode: str = "major",
        bars: int = 4,
        tempo_bpm: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        config: Optional[CounterMelodyConfig] = None,
        motion_override: Optional[CounterMotion] = None,
        technique_override: Optional[CounterTechnique] = None,
        register_override: Optional[CounterRegister] = None,
    ) -> CounterMelodyOutput:
        """Generate counter-melody for a main melody."""
        profile = self.profiles.get(emotion.lower(), self.profiles["neutral"])
        
        motion = motion_override or profile["motion"]
        technique = technique_override or profile["technique"]
        rhythm = profile["rhythm"]
        register = register_override or profile["register"]
        
        # Get scale for snapping
        scale_pitches = get_scale_pitches(key, mode)
        
        # Calculate melody center for contrary motion
        if main_melody:
            melody_center = sum(n.pitch for n in main_melody) // len(main_melody)
        else:
            melody_center = 60
        
        beats_per_bar = time_signature[0]
        ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
        total_ticks = ticks_per_bar * bars
        
        counter_notes = []
        
        # Process based on technique
        if technique == CounterTechnique.PEDAL:
            # Generate pedal tone
            pedal_pitch = generate_pedal_pitch(scale_pitches, register)
            pedal_note = CounterNote(
                pitch=pedal_pitch,
                start_tick=0,
                duration_ticks=total_ticks,
                velocity=int(main_melody[0].velocity * profile["velocity_ratio"]) if main_melody else 60,
                source_note_idx=-1,
                interval_from_source=0,
                technique_applied=technique,
            )
            counter_notes.append(pedal_note)
            
        elif technique == CounterTechnique.CANON:
            # Delayed imitation
            delay = profile["delay_ticks"]
            for i, note in enumerate(main_melody):
                if random.random() > profile["density"]:
                    continue
                
                interval = random.choice(profile["interval_preference"])
                target_pitch = calculate_interval(note.pitch, interval, register)
                target_pitch = snap_to_scale(target_pitch, scale_pitches)
                
                new_tick = note.start_tick + delay
                if new_tick < total_ticks:
                    velocity = int(note.velocity * profile["velocity_ratio"])
                    velocity += random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
                    
                    counter_notes.append(CounterNote(
                        pitch=target_pitch,
                        start_tick=new_tick,
                        duration_ticks=int(note.duration_ticks * profile["sustain_ratio"]),
                        velocity=max(1, min(127, velocity)),
                        source_note_idx=i,
                        interval_from_source=interval,
                        technique_applied=technique,
                    ))
                    
        elif technique == CounterTechnique.CALL_RESPONSE:
            # Echo with response modification
            delay = profile["delay_ticks"]
            for i, note in enumerate(main_melody):
                if random.random() > profile["density"]:
                    continue
                
                # Vary interval more for "response" feel
                interval = random.choice(profile["interval_preference"])
                if random.random() < 0.3:
                    interval = -interval  # Sometimes flip
                
                target_pitch = calculate_interval(note.pitch, interval, register)
                target_pitch = snap_to_scale(target_pitch, scale_pitches)
                
                new_tick = note.start_tick + delay
                timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
                new_tick = max(0, new_tick + timing_offset)
                
                if new_tick < total_ticks:
                    velocity = int(note.velocity * profile["velocity_ratio"])
                    velocity += random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
                    
                    counter_notes.append(CounterNote(
                        pitch=target_pitch,
                        start_tick=new_tick,
                        duration_ticks=int(note.duration_ticks * profile["sustain_ratio"]),
                        velocity=max(1, min(127, velocity)),
                        source_note_idx=i,
                        interval_from_source=interval,
                        technique_applied=technique,
                    ))
                    
        elif technique == CounterTechnique.FILL:
            # Fill gaps between melody notes
            for i in range(len(main_melody) - 1):
                note = main_melody[i]
                next_note = main_melody[i + 1]
                
                gap_start = note.start_tick + note.duration_ticks
                gap_end = next_note.start_tick
                gap_duration = gap_end - gap_start
                
                if gap_duration > TICKS_PER_BEAT // 4 and random.random() < profile["density"]:
                    interval = random.choice(profile["interval_preference"])
                    target_pitch = calculate_interval(note.pitch, interval, register)
                    target_pitch = snap_to_scale(target_pitch, scale_pitches)
                    
                    fill_start = gap_start + random.randint(0, gap_duration // 4)
                    fill_duration = min(gap_duration - TICKS_PER_BEAT // 8, 
                                       int(TICKS_PER_BEAT * profile["sustain_ratio"]))
                    
                    velocity = int(note.velocity * profile["velocity_ratio"])
                    
                    counter_notes.append(CounterNote(
                        pitch=target_pitch,
                        start_tick=fill_start,
                        duration_ticks=max(60, fill_duration),
                        velocity=max(1, min(127, velocity)),
                        source_note_idx=i,
                        interval_from_source=interval,
                        technique_applied=technique,
                    ))
                    
        else:
            # Standard parallel/contrary/similar motion techniques
            for i, note in enumerate(main_melody):
                if random.random() > profile["density"]:
                    continue
                
                interval = random.choice(profile["interval_preference"])
                
                if motion == CounterMotion.CONTRARY:
                    target_pitch = apply_contrary_motion(note.pitch, melody_center, interval)
                elif motion == CounterMotion.OBLIQUE:
                    # Use same pitch as previous or held pitch
                    if counter_notes:
                        target_pitch = counter_notes[-1].pitch
                    else:
                        target_pitch = calculate_interval(note.pitch, interval, register)
                else:
                    target_pitch = calculate_interval(note.pitch, interval, register)
                
                target_pitch = snap_to_scale(target_pitch, scale_pitches)
                
                # Apply rhythm modifications
                new_tick = note.start_tick + profile["delay_ticks"]
                timing_offset = random.randint(-profile["humanize_timing"], profile["humanize_timing"])
                new_tick = max(0, new_tick + timing_offset)
                
                if rhythm == CounterRhythm.DOUBLE:
                    duration = note.duration_ticks // 2
                elif rhythm == CounterRhythm.HALF:
                    duration = note.duration_ticks * 2
                else:
                    duration = note.duration_ticks
                
                duration = int(duration * profile["sustain_ratio"])
                
                velocity = int(note.velocity * profile["velocity_ratio"])
                velocity += random.randint(-profile["humanize_velocity"], profile["humanize_velocity"])
                
                if new_tick < total_ticks:
                    counter_notes.append(CounterNote(
                        pitch=target_pitch,
                        start_tick=new_tick,
                        duration_ticks=max(60, duration),
                        velocity=max(1, min(127, velocity)),
                        source_note_idx=i,
                        interval_from_source=interval,
                        technique_applied=technique,
                    ))
        
        # Sort by start time
        counter_notes.sort(key=lambda n: n.start_tick)
        
        return CounterMelodyOutput(
            notes=counter_notes,
            emotion=emotion,
            motion_used=motion,
            technique_used=technique,
            rhythm_used=rhythm,
            register_used=register,
            gm_instrument=profile["gm_instrument"],
            total_ticks=total_ticks,
            metadata={
                "tempo_bpm": tempo_bpm,
                "time_signature": time_signature,
                "key": key,
                "mode": mode,
                "main_melody_notes": len(main_melody),
            }
        )
    
    def generate_from_pitches(
        self,
        emotion: str,
        pitches: List[int],
        key: str = "C",
        mode: str = "major",
        bars: int = 4,
        tempo_bpm: int = 120,
    ) -> CounterMelodyOutput:
        """Generate counter-melody from simple pitch list."""
        # Convert pitches to MelodyNote objects
        ticks_per_note = (TICKS_PER_BEAT * 4 * bars) // len(pitches)
        melody_notes = []
        for i, pitch in enumerate(pitches):
            melody_notes.append(MelodyNote(
                pitch=pitch,
                start_tick=i * ticks_per_note,
                duration_ticks=int(ticks_per_note * 0.9),
                velocity=70,
            ))
        
        return self.generate(
            emotion=emotion,
            main_melody=melody_notes,
            chord_progression=[],
            key=key,
            mode=mode,
            bars=bars,
            tempo_bpm=tempo_bpm,
        )
    
    def get_available_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return list(self.profiles.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_harmony_thirds(pitches: List[int], key: str = "C") -> CounterMelodyOutput:
    """Quick parallel thirds harmonization."""
    engine = CounterMelodyEngine()
    return engine.generate_from_pitches(
        "joy", pitches, key, "major", 
    )


def counter_to_midi_events(counter_output: CounterMelodyOutput, channel: int = 2) -> List[Dict]:
    """Convert CounterMelodyOutput to MIDI event list."""
    events = []
    
    events.append({
        "type": "program_change",
        "tick": 0,
        "channel": channel,
        "program": counter_output.gm_instrument,
    })
    
    for note in counter_output.notes:
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
    engine = CounterMelodyEngine()
    
    print("=== COUNTER-MELODY ENGINE DEMO ===\n")
    
    # Simple melody
    melody = [
        MelodyNote(pitch=65, start_tick=0, duration_ticks=480, velocity=80),
        MelodyNote(pitch=67, start_tick=480, duration_ticks=480, velocity=75),
        MelodyNote(pitch=69, start_tick=960, duration_ticks=480, velocity=70),
        MelodyNote(pitch=65, start_tick=1440, duration_ticks=960, velocity=85),
    ]
    
    emotions = ["grief", "hope", "joy", "anxiety", "peace"]
    
    for emotion in emotions:
        counter = engine.generate(
            emotion=emotion,
            main_melody=melody,
            chord_progression=["F", "C", "Dm", "Bb"],
            key="F",
            mode="major",
            bars=2,
            tempo_bpm=82,
        )
        
        print(f"--- {emotion.upper()} ---")
        print(f"Motion: {counter.motion_used.value}")
        print(f"Technique: {counter.technique_used.value}")
        print(f"Rhythm: {counter.rhythm_used.value}")
        print(f"Register: {counter.register_used.value}")
        print(f"Counter notes: {len(counter.notes)}")
        if counter.notes:
            intervals = [n.interval_from_source for n in counter.notes]
            print(f"Interval range: {min(intervals)} to {max(intervals)}")
        print()
