"""
Tension Engine - Harmonic tension/release cycles and anticipation curves.

Part of Kelly MIDI Companion Tier 3.

Philosophy: Music is a conversation between tension and release. A chord that
"wants" to go somewhere creates anticipation. A resolution provides relief.
Grief sustains tension without release - the wound that won't heal. Joy cycles
quickly through tension and release. Fear builds tension that threatens but
never fully arrives. The tension curve IS the emotional drama.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Callable
import random
import math


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class TensionLevel(Enum):
    """Tension levels from complete resolution to maximum dissonance."""
    RESOLVED = 0            # Complete rest (tonic, stable)
    LOW = 1                 # Slight movement
    MILD = 2                # Some tension
    MODERATE = 3            # Noticeable tension
    ELEVATED = 4            # Building
    HIGH = 5                # Significant tension
    INTENSE = 6             # Strong pull
    EXTREME = 7             # Maximum before resolution
    UNRESOLVED = 8          # Deliberately unresolved (suspense)


class TensionTechnique(Enum):
    """Musical techniques that create tension."""
    DOMINANT = "dominant"                   # V chord, strongest pull to I
    SECONDARY_DOMINANT = "secondary_dominant"  # V/x, temporary tonicization
    DIMINISHED = "diminished"               # Diminished chords
    AUGMENTED = "augmented"                 # Augmented chords
    SUSPENSION = "suspension"               # Sus2, Sus4
    TRITONE = "tritone"                     # The devil's interval
    CHROMATIC = "chromatic"                 # Chromatic approach
    PEDAL = "pedal"                         # Pedal tone against moving harmony
    SEQUENCE = "sequence"                   # Repeated pattern building
    DECEPTIVE = "deceptive"                 # Deceptive cadence
    HALF_CADENCE = "half_cadence"           # Ending on V
    MODAL_MIXTURE = "modal_mixture"         # Borrowing from parallel mode
    NEAPOLITAN = "neapolitan"               # bII chord
    AUGMENTED_SIXTH = "augmented_sixth"     # +6 chords
    SILENCE = "silence"                     # Dramatic pause
    REGISTER_SHIFT = "register_shift"       # Moving to extreme register
    DENSITY = "density"                     # Note density buildup
    RHYTHM = "rhythm"                       # Rhythmic acceleration
    ANTICIPATION = "anticipation"           # Early arrival of next chord


class ReleaseType(Enum):
    """Types of tension release."""
    AUTHENTIC = "authentic"         # V-I, strongest
    PLAGAL = "plagal"              # IV-I, softer (amen cadence)
    DECEPTIVE = "deceptive"        # V-vi, surprise
    HALF = "half"                  # Ending on V, incomplete
    MODAL = "modal"                # Mode-specific resolution
    STEPWISE = "stepwise"          # Chromatic/stepwise resolution
    AVOIDED = "avoided"            # Tension sustained
    INTERRUPTED = "interrupted"    # Resolution interrupted by new tension


class TensionCurveShape(Enum):
    """Shapes for tension over time."""
    LINEAR_BUILD = "linear_build"           # Steady increase
    LINEAR_RELEASE = "linear_release"       # Steady decrease
    SPIKE = "spike"                         # Quick peak and release
    PLATEAU = "plateau"                     # Build, sustain, release
    SAWTOOTH = "sawtooth"                   # Build-release cycles
    INVERSE_SAWTOOTH = "inverse_sawtooth"   # Release-build cycles (unusual)
    WAVE = "wave"                           # Oscillating
    SUSTAINED = "sustained"                 # Constant tension level
    CLIMAX = "climax"                       # Build to single peak
    DELAYED_RESOLUTION = "delayed_resolution"  # Build, sustain, late release
    NO_RESOLUTION = "no_resolution"         # Build without release


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TensionPoint:
    """A point on the tension curve."""
    position: float              # 0.0 - 1.0 through section
    tension_level: TensionLevel
    technique: Optional[TensionTechnique] = None
    target_chord_function: Optional[str] = None  # e.g., "V", "ii", "bVI"
    is_release_point: bool = False


@dataclass
class TensionCurve:
    """Complete tension curve for a section."""
    points: List[TensionPoint]
    shape: TensionCurveShape
    peak_position: float = 0.5
    peak_level: TensionLevel = TensionLevel.HIGH
    base_level: TensionLevel = TensionLevel.LOW
    
    # Release characteristics
    release_type: ReleaseType = ReleaseType.AUTHENTIC
    has_final_release: bool = True
    
    # Techniques used
    primary_techniques: List[TensionTechnique] = field(default_factory=list)


@dataclass
class HarmonicTension:
    """Tension information for a specific chord/moment."""
    chord_symbol: str           # e.g., "G7", "Dm", "Bbmaj7"
    function: str               # e.g., "V7", "ii", "bVI"
    tension_level: TensionLevel
    resolution_target: Optional[str] = None  # What it wants to resolve to
    techniques_active: List[TensionTechnique] = field(default_factory=list)
    
    # Voice leading tension
    has_leading_tone: bool = False
    has_tritone: bool = False
    has_suspension: bool = False
    
    # Contextual
    is_borrowed: bool = False           # From parallel mode
    is_secondary_function: bool = False  # V/x, etc.


@dataclass
class TensionProfile:
    """Complete tension specification for a section."""
    curve: TensionCurve
    harmonic_tensions: List[HarmonicTension]
    
    # Rhythmic tension
    rhythmic_density_curve: List[float] = field(default_factory=list)
    use_rhythmic_tension: bool = True
    
    # Register tension
    register_tension_curve: List[float] = field(default_factory=list)
    use_register_tension: bool = False
    
    # Metadata
    emotion: str = ""
    section_type: str = ""
    allows_resolution: bool = True


# =============================================================================
# CHORD TENSION DATABASE
# =============================================================================

# Inherent tension levels for chord types
CHORD_TYPE_TENSION: Dict[str, int] = {
    # Major chords
    "maj": 0,
    "": 0,
    "6": 1,
    "maj7": 2,
    "maj9": 2,
    "add9": 1,
    
    # Minor chords
    "m": 1,
    "min": 1,
    "m7": 2,
    "m9": 2,
    "m6": 2,
    "mMaj7": 4,      # Minor-major 7, high tension
    
    # Dominant
    "7": 4,          # Dominant 7 - wants to resolve
    "9": 4,
    "11": 4,
    "13": 4,
    "7#9": 6,        # Hendrix chord
    "7b9": 6,
    "7#11": 5,
    "7alt": 7,       # Altered dominant
    
    # Diminished
    "dim": 6,
    "dim7": 7,
    "m7b5": 5,       # Half-diminished
    
    # Augmented
    "aug": 5,
    "+": 5,
    "+7": 6,
    
    # Suspended
    "sus2": 3,
    "sus4": 3,
    "7sus4": 4,
    
    # Other
    "5": 0,          # Power chord - neutral
}

# Function-based tension (Roman numerals)
FUNCTION_TENSION: Dict[str, int] = {
    # Diatonic
    "I": 0,          # Tonic - home
    "i": 0,
    "ii": 2,
    "II": 2,
    "iii": 2,
    "III": 2,
    "IV": 2,
    "iv": 3,
    "V": 4,          # Dominant - strong pull
    "v": 3,
    "vi": 1,
    "VI": 1,
    "vii°": 6,
    "VII": 3,
    
    # Secondary dominants
    "V/V": 5,
    "V/ii": 4,
    "V/iii": 4,
    "V/IV": 4,
    "V/vi": 4,
    
    # Borrowed chords
    "bII": 5,        # Neapolitan
    "bIII": 3,
    "bVI": 3,
    "bVII": 3,
    "iv": 3,         # Minor iv in major
    "#iv°": 5,
    
    # Augmented sixths
    "It+6": 6,
    "Fr+6": 6,
    "Ger+6": 6,
}


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_TENSION_PROFILES: Dict[str, Dict] = {
    "grief": {
        "base_tension": TensionLevel.MODERATE,
        "max_tension": TensionLevel.HIGH,
        "min_tension": TensionLevel.MILD,
        "preferred_curves": [TensionCurveShape.SUSTAINED, TensionCurveShape.NO_RESOLUTION],
        "preferred_techniques": [
            TensionTechnique.SUSPENSION,
            TensionTechnique.MODAL_MIXTURE,
            TensionTechnique.PEDAL,
        ],
        "release_type": ReleaseType.AVOIDED,
        "allows_resolution": False,         # Grief doesn't resolve
        "tension_cycles_per_section": 1,
        "cycle_amplitude": 0.3,
    },
    
    "sadness": {
        "base_tension": TensionLevel.MILD,
        "max_tension": TensionLevel.ELEVATED,
        "min_tension": TensionLevel.LOW,
        "preferred_curves": [TensionCurveShape.WAVE, TensionCurveShape.DELAYED_RESOLUTION],
        "preferred_techniques": [
            TensionTechnique.MODAL_MIXTURE,
            TensionTechnique.DECEPTIVE,
            TensionTechnique.SUSPENSION,
        ],
        "release_type": ReleaseType.DECEPTIVE,
        "allows_resolution": True,
        "tension_cycles_per_section": 2,
        "cycle_amplitude": 0.4,
    },
    
    "melancholy": {
        "base_tension": TensionLevel.MILD,
        "max_tension": TensionLevel.MODERATE,
        "min_tension": TensionLevel.LOW,
        "preferred_curves": [TensionCurveShape.WAVE, TensionCurveShape.SUSTAINED],
        "preferred_techniques": [
            TensionTechnique.MODAL_MIXTURE,
            TensionTechnique.SUSPENSION,
            TensionTechnique.CHROMATIC,
        ],
        "release_type": ReleaseType.PLAGAL,
        "allows_resolution": True,
        "tension_cycles_per_section": 2,
        "cycle_amplitude": 0.3,
    },
    
    "rage": {
        "base_tension": TensionLevel.HIGH,
        "max_tension": TensionLevel.EXTREME,
        "min_tension": TensionLevel.ELEVATED,
        "preferred_curves": [TensionCurveShape.SPIKE, TensionCurveShape.SAWTOOTH],
        "preferred_techniques": [
            TensionTechnique.TRITONE,
            TensionTechnique.DIMINISHED,
            TensionTechnique.DENSITY,
            TensionTechnique.RHYTHM,
        ],
        "release_type": ReleaseType.INTERRUPTED,
        "allows_resolution": True,          # Brief releases before more rage
        "tension_cycles_per_section": 4,
        "cycle_amplitude": 0.6,
    },
    
    "anger": {
        "base_tension": TensionLevel.ELEVATED,
        "max_tension": TensionLevel.INTENSE,
        "min_tension": TensionLevel.MODERATE,
        "preferred_curves": [TensionCurveShape.LINEAR_BUILD, TensionCurveShape.SAWTOOTH],
        "preferred_techniques": [
            TensionTechnique.DOMINANT,
            TensionTechnique.TRITONE,
            TensionTechnique.RHYTHM,
        ],
        "release_type": ReleaseType.AUTHENTIC,
        "allows_resolution": True,
        "tension_cycles_per_section": 3,
        "cycle_amplitude": 0.5,
    },
    
    "fear": {
        "base_tension": TensionLevel.MODERATE,
        "max_tension": TensionLevel.EXTREME,
        "min_tension": TensionLevel.MILD,
        "preferred_curves": [TensionCurveShape.LINEAR_BUILD, TensionCurveShape.DELAYED_RESOLUTION],
        "preferred_techniques": [
            TensionTechnique.SILENCE,
            TensionTechnique.DIMINISHED,
            TensionTechnique.CHROMATIC,
            TensionTechnique.REGISTER_SHIFT,
        ],
        "release_type": ReleaseType.DECEPTIVE,
        "allows_resolution": False,         # Fear doesn't resolve fully
        "tension_cycles_per_section": 2,
        "cycle_amplitude": 0.7,
    },
    
    "anxiety": {
        "base_tension": TensionLevel.ELEVATED,
        "max_tension": TensionLevel.HIGH,
        "min_tension": TensionLevel.MODERATE,
        "preferred_curves": [TensionCurveShape.SUSTAINED, TensionCurveShape.WAVE],
        "preferred_techniques": [
            TensionTechnique.SUSPENSION,
            TensionTechnique.SEQUENCE,
            TensionTechnique.RHYTHM,
            TensionTechnique.ANTICIPATION,
        ],
        "release_type": ReleaseType.AVOIDED,
        "allows_resolution": False,         # Anxiety is unresolved by nature
        "tension_cycles_per_section": 4,
        "cycle_amplitude": 0.3,
    },
    
    "hope": {
        "base_tension": TensionLevel.MILD,
        "max_tension": TensionLevel.ELEVATED,
        "min_tension": TensionLevel.RESOLVED,
        "preferred_curves": [TensionCurveShape.CLIMAX, TensionCurveShape.SAWTOOTH],
        "preferred_techniques": [
            TensionTechnique.SECONDARY_DOMINANT,
            TensionTechnique.DOMINANT,
            TensionTechnique.SEQUENCE,
        ],
        "release_type": ReleaseType.AUTHENTIC,
        "allows_resolution": True,
        "tension_cycles_per_section": 2,
        "cycle_amplitude": 0.5,
    },
    
    "joy": {
        "base_tension": TensionLevel.LOW,
        "max_tension": TensionLevel.MODERATE,
        "min_tension": TensionLevel.RESOLVED,
        "preferred_curves": [TensionCurveShape.SAWTOOTH, TensionCurveShape.WAVE],
        "preferred_techniques": [
            TensionTechnique.DOMINANT,
            TensionTechnique.SECONDARY_DOMINANT,
        ],
        "release_type": ReleaseType.AUTHENTIC,
        "allows_resolution": True,
        "tension_cycles_per_section": 4,
        "cycle_amplitude": 0.4,
    },
    
    "peace": {
        "base_tension": TensionLevel.RESOLVED,
        "max_tension": TensionLevel.MILD,
        "min_tension": TensionLevel.RESOLVED,
        "preferred_curves": [TensionCurveShape.SUSTAINED, TensionCurveShape.LINEAR_RELEASE],
        "preferred_techniques": [
            TensionTechnique.PEDAL,
            TensionTechnique.SUSPENSION,
        ],
        "release_type": ReleaseType.PLAGAL,
        "allows_resolution": True,
        "tension_cycles_per_section": 1,
        "cycle_amplitude": 0.2,
    },
    
    "nostalgia": {
        "base_tension": TensionLevel.MILD,
        "max_tension": TensionLevel.MODERATE,
        "min_tension": TensionLevel.LOW,
        "preferred_curves": [TensionCurveShape.WAVE, TensionCurveShape.DELAYED_RESOLUTION],
        "preferred_techniques": [
            TensionTechnique.MODAL_MIXTURE,
            TensionTechnique.DECEPTIVE,
            TensionTechnique.SUSPENSION,
        ],
        "release_type": ReleaseType.DECEPTIVE,
        "allows_resolution": True,
        "tension_cycles_per_section": 2,
        "cycle_amplitude": 0.4,
    },
    
    "longing": {
        "base_tension": TensionLevel.MODERATE,
        "max_tension": TensionLevel.HIGH,
        "min_tension": TensionLevel.MILD,
        "preferred_curves": [TensionCurveShape.CLIMAX, TensionCurveShape.DELAYED_RESOLUTION],
        "preferred_techniques": [
            TensionTechnique.SUSPENSION,
            TensionTechnique.DOMINANT,
            TensionTechnique.CHROMATIC,
        ],
        "release_type": ReleaseType.HALF,
        "allows_resolution": False,         # Longing is unfulfilled
        "tension_cycles_per_section": 2,
        "cycle_amplitude": 0.5,
    },
    
    "tension": {
        "base_tension": TensionLevel.ELEVATED,
        "max_tension": TensionLevel.EXTREME,
        "min_tension": TensionLevel.MODERATE,
        "preferred_curves": [TensionCurveShape.LINEAR_BUILD, TensionCurveShape.PLATEAU],
        "preferred_techniques": [
            TensionTechnique.TRITONE,
            TensionTechnique.DIMINISHED,
            TensionTechnique.SEQUENCE,
            TensionTechnique.DENSITY,
        ],
        "release_type": ReleaseType.INTERRUPTED,
        "allows_resolution": False,
        "tension_cycles_per_section": 1,
        "cycle_amplitude": 0.8,
    },
    
    "defiance": {
        "base_tension": TensionLevel.HIGH,
        "max_tension": TensionLevel.EXTREME,
        "min_tension": TensionLevel.ELEVATED,
        "preferred_curves": [TensionCurveShape.SPIKE, TensionCurveShape.SAWTOOTH],
        "preferred_techniques": [
            TensionTechnique.TRITONE,
            TensionTechnique.DOMINANT,
            TensionTechnique.RHYTHM,
        ],
        "release_type": ReleaseType.AUTHENTIC,
        "allows_resolution": True,
        "tension_cycles_per_section": 3,
        "cycle_amplitude": 0.6,
    },
    
    "vulnerability": {
        "base_tension": TensionLevel.MILD,
        "max_tension": TensionLevel.MODERATE,
        "min_tension": TensionLevel.LOW,
        "preferred_curves": [TensionCurveShape.WAVE, TensionCurveShape.SUSTAINED],
        "preferred_techniques": [
            TensionTechnique.SUSPENSION,
            TensionTechnique.MODAL_MIXTURE,
            TensionTechnique.SILENCE,
        ],
        "release_type": ReleaseType.PLAGAL,
        "allows_resolution": True,
        "tension_cycles_per_section": 2,
        "cycle_amplitude": 0.3,
    },
    
    "euphoria": {
        "base_tension": TensionLevel.MODERATE,
        "max_tension": TensionLevel.INTENSE,
        "min_tension": TensionLevel.LOW,
        "preferred_curves": [TensionCurveShape.SAWTOOTH, TensionCurveShape.CLIMAX],
        "preferred_techniques": [
            TensionTechnique.DOMINANT,
            TensionTechnique.SEQUENCE,
            TensionTechnique.DENSITY,
        ],
        "release_type": ReleaseType.AUTHENTIC,
        "allows_resolution": True,
        "tension_cycles_per_section": 4,
        "cycle_amplitude": 0.6,
    },
    
    "emptiness": {
        "base_tension": TensionLevel.LOW,
        "max_tension": TensionLevel.MILD,
        "min_tension": TensionLevel.RESOLVED,
        "preferred_curves": [TensionCurveShape.SUSTAINED, TensionCurveShape.LINEAR_RELEASE],
        "preferred_techniques": [
            TensionTechnique.PEDAL,
            TensionTechnique.SILENCE,
        ],
        "release_type": ReleaseType.AVOIDED,
        "allows_resolution": False,
        "tension_cycles_per_section": 1,
        "cycle_amplitude": 0.1,
    },
}


# =============================================================================
# CURVE GENERATORS
# =============================================================================

def generate_tension_curve(
    shape: TensionCurveShape,
    num_points: int,
    base_level: int,
    peak_level: int,
    cycles: int = 2,
) -> List[int]:
    """Generate tension level values for a tension shape."""
    
    points = []
    
    if shape == TensionCurveShape.LINEAR_BUILD:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            points.append(int(base_level + t * (peak_level - base_level)))
    
    elif shape == TensionCurveShape.LINEAR_RELEASE:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            points.append(int(peak_level - t * (peak_level - base_level)))
    
    elif shape == TensionCurveShape.SPIKE:
        peak_pos = 0.7
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < peak_pos:
                # Build
                points.append(int(base_level + (t / peak_pos) * (peak_level - base_level)))
            else:
                # Quick release
                release_t = (t - peak_pos) / (1 - peak_pos)
                points.append(int(peak_level - release_t * (peak_level - base_level) * 0.8))
    
    elif shape == TensionCurveShape.PLATEAU:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.3:
                # Build
                points.append(int(base_level + (t / 0.3) * (peak_level - base_level)))
            elif t < 0.7:
                # Sustain
                points.append(peak_level)
            else:
                # Release
                release_t = (t - 0.7) / 0.3
                points.append(int(peak_level - release_t * (peak_level - base_level)))
    
    elif shape == TensionCurveShape.SAWTOOTH:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            cycle_pos = (t * cycles) % 1.0
            if cycle_pos < 0.8:
                # Build
                points.append(int(base_level + (cycle_pos / 0.8) * (peak_level - base_level)))
            else:
                # Quick release
                points.append(base_level)
    
    elif shape == TensionCurveShape.INVERSE_SAWTOOTH:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            cycle_pos = (t * cycles) % 1.0
            if cycle_pos < 0.2:
                # Quick rise
                points.append(peak_level)
            else:
                # Slow fall
                points.append(int(peak_level - ((cycle_pos - 0.2) / 0.8) * (peak_level - base_level)))
    
    elif shape == TensionCurveShape.WAVE:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            wave = math.sin(t * math.pi * cycles * 2) * 0.5 + 0.5
            mid = (base_level + peak_level) / 2
            amplitude = (peak_level - base_level) / 2
            points.append(int(mid + wave * amplitude - amplitude / 2))
    
    elif shape == TensionCurveShape.SUSTAINED:
        mid = (base_level + peak_level) // 2
        points = [mid] * num_points
    
    elif shape == TensionCurveShape.CLIMAX:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.8:
                # Long build
                points.append(int(base_level + (t / 0.8) ** 1.5 * (peak_level - base_level)))
            else:
                # Quick release
                release_t = (t - 0.8) / 0.2
                points.append(int(peak_level - release_t * (peak_level - base_level) * 0.6))
    
    elif shape == TensionCurveShape.DELAYED_RESOLUTION:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            if t < 0.5:
                # Build
                points.append(int(base_level + (t / 0.5) * (peak_level - base_level)))
            elif t < 0.9:
                # Sustain at peak
                points.append(peak_level)
            else:
                # Late release
                release_t = (t - 0.9) / 0.1
                points.append(int(peak_level - release_t * (peak_level - base_level)))
    
    elif shape == TensionCurveShape.NO_RESOLUTION:
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Build but never release
            if t < 0.6:
                points.append(int(base_level + (t / 0.6) * (peak_level - base_level)))
            else:
                # Sustain at high tension
                points.append(peak_level)
    
    else:
        points = [base_level] * num_points
    
    # Clamp
    return [max(0, min(8, p)) for p in points]


# =============================================================================
# TENSION ENGINE
# =============================================================================

class TensionEngine:
    """
    Generates harmonic tension/release curves for emotional expression.
    
    Provides tension analysis for chords and generates tension profiles
    for sections based on emotional intent.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate_tension_profile(
        self,
        emotion: str,
        section_type: str,
        bars: int,
        chords_per_bar: int = 2,
        key: str = "C",
        mode: str = "major",
        is_climax: bool = False,
    ) -> TensionProfile:
        """
        Generate tension profile for a section.
        
        Args:
            emotion: Primary emotion
            section_type: Type of section
            bars: Number of bars
            chords_per_bar: Harmonic rhythm
            key: Musical key
            mode: Scale mode
            is_climax: Whether this is the emotional peak
            
        Returns:
            TensionProfile with curve and harmonic tensions
        """
        emotion_lower = emotion.lower()
        profile = EMOTION_TENSION_PROFILES.get(
            emotion_lower,
            EMOTION_TENSION_PROFILES["sadness"]
        )
        
        # Adjust for section type
        section_modifiers = {
            "intro": (0, 0.7),
            "verse": (0, 0.8),
            "pre_chorus": (1, 0.9),
            "chorus": (1, 1.0),
            "bridge": (0, 0.8),
            "breakdown": (-1, 0.6),
            "build": (2, 1.2),
            "drop": (1, 1.0),
            "outro": (-1, 0.7),
        }
        
        level_offset, range_mult = section_modifiers.get(section_type.lower(), (0, 0.8))
        
        # Calculate tension range
        base_level = max(0, min(8, profile["base_tension"].value + level_offset))
        max_level = max(0, min(8, int(profile["max_tension"].value * range_mult)))
        if is_climax:
            max_level = profile["max_tension"].value
        
        # Select curve shape
        if section_type.lower() == "build":
            shape = TensionCurveShape.LINEAR_BUILD
        elif section_type.lower() == "breakdown":
            shape = TensionCurveShape.LINEAR_RELEASE
        elif section_type.lower() == "drop" and emotion_lower in ["rage", "defiance", "euphoria"]:
            shape = TensionCurveShape.SPIKE
        else:
            shape = self.rng.choice(profile["preferred_curves"])
        
        # Generate curve
        num_points = bars * chords_per_bar
        tension_values = generate_tension_curve(
            shape=shape,
            num_points=num_points,
            base_level=base_level,
            peak_level=max_level,
            cycles=profile["tension_cycles_per_section"],
        )
        
        # Build curve points
        curve_points = []
        for i, level in enumerate(tension_values):
            pos = i / (len(tension_values) - 1) if len(tension_values) > 1 else 0
            
            # Assign technique based on tension level
            technique = None
            if level >= 6:
                technique = self.rng.choice([
                    TensionTechnique.TRITONE,
                    TensionTechnique.DIMINISHED,
                    TensionTechnique.DOMINANT,
                ])
            elif level >= 4:
                technique = self.rng.choice(profile["preferred_techniques"])
            elif level <= 1:
                # Potential release point
                pass
            
            # Is this a release point?
            is_release = False
            if i > 0 and tension_values[i] < tension_values[i-1] - 2:
                is_release = True
            
            curve_points.append(TensionPoint(
                position=pos,
                tension_level=TensionLevel(level),
                technique=technique,
                is_release_point=is_release,
            ))
        
        # Determine release type
        release_type = profile["release_type"]
        has_final_release = profile["allows_resolution"]
        if section_type.lower() in ["breakdown", "outro"] and profile["allows_resolution"]:
            has_final_release = True
        
        curve = TensionCurve(
            points=curve_points,
            shape=shape,
            peak_level=TensionLevel(max_level),
            base_level=TensionLevel(base_level),
            release_type=release_type,
            has_final_release=has_final_release,
            primary_techniques=profile["preferred_techniques"],
        )
        
        # Generate rhythmic tension curve
        rhythmic_density = []
        if profile.get("use_rhythmic_tension", True):
            for i in range(num_points):
                pos = i / (num_points - 1) if num_points > 1 else 0
                tension_at_pos = tension_values[i]
                # Higher tension = denser rhythm
                density = 0.3 + (tension_at_pos / 8) * 0.7
                rhythmic_density.append(density)
        
        return TensionProfile(
            curve=curve,
            harmonic_tensions=[],  # Filled in when actual chords are provided
            rhythmic_density_curve=rhythmic_density,
            use_rhythmic_tension=True,
            emotion=emotion_lower,
            section_type=section_type,
            allows_resolution=has_final_release,
        )
    
    def analyze_chord_tension(
        self,
        chord_symbol: str,
        function: str = "",
        context_key: str = "C",
        context_mode: str = "major",
    ) -> HarmonicTension:
        """
        Analyze tension level of a specific chord.
        
        Args:
            chord_symbol: Chord symbol (e.g., "G7", "Dm7", "Bbmaj7")
            function: Roman numeral function (e.g., "V7", "ii7", "bVI")
            context_key: Key context
            context_mode: Mode context
            
        Returns:
            HarmonicTension with analysis
        """
        # Parse chord type from symbol
        chord_type = ""
        for ct in sorted(CHORD_TYPE_TENSION.keys(), key=len, reverse=True):
            if ct and ct in chord_symbol:
                chord_type = ct
                break
        
        # Base tension from chord type
        base_tension = CHORD_TYPE_TENSION.get(chord_type, 2)
        
        # Add function tension
        if function:
            function_tension = FUNCTION_TENSION.get(function, 2)
            base_tension = (base_tension + function_tension) // 2 + 1
        
        # Clamp
        tension_level = TensionLevel(max(0, min(8, base_tension)))
        
        # Detect techniques
        techniques = []
        has_tritone = "7" in chord_symbol and ("dim" in chord_symbol or function == "V" or "V/" in function)
        has_leading_tone = function in ["V", "vii°"] or "V/" in function
        has_suspension = "sus" in chord_symbol
        is_borrowed = function.startswith("b") or (context_mode == "major" and "m" in chord_symbol.lower() and function not in ["ii", "iii", "vi"])
        is_secondary = "/" in function
        
        if has_tritone:
            techniques.append(TensionTechnique.TRITONE)
        if has_leading_tone:
            techniques.append(TensionTechnique.DOMINANT)
        if has_suspension:
            techniques.append(TensionTechnique.SUSPENSION)
        if is_borrowed:
            techniques.append(TensionTechnique.MODAL_MIXTURE)
        if is_secondary:
            techniques.append(TensionTechnique.SECONDARY_DOMINANT)
        if "dim" in chord_symbol:
            techniques.append(TensionTechnique.DIMINISHED)
        if "aug" in chord_symbol or "+" in chord_symbol:
            techniques.append(TensionTechnique.AUGMENTED)
        
        # Determine resolution target
        resolution_target = None
        if function == "V" or function == "V7":
            resolution_target = "I"
        elif function == "vii°":
            resolution_target = "I"
        elif "V/" in function:
            resolution_target = function.split("/")[1]
        elif "sus" in chord_symbol:
            resolution_target = chord_symbol.replace("sus4", "").replace("sus2", "")
        
        return HarmonicTension(
            chord_symbol=chord_symbol,
            function=function,
            tension_level=tension_level,
            resolution_target=resolution_target,
            techniques_active=techniques,
            has_leading_tone=has_leading_tone,
            has_tritone=has_tritone,
            has_suspension=has_suspension,
            is_borrowed=is_borrowed,
            is_secondary_function=is_secondary,
        )
    
    def get_tension_at_position(
        self,
        profile: TensionProfile,
        position: float,
    ) -> TensionLevel:
        """Get tension level at a specific position."""
        
        if not profile.curve.points:
            return TensionLevel.MODERATE
        
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
            return prev_point.tension_level
        
        t = (position - prev_point.position) / (next_point.position - prev_point.position)
        level = int(prev_point.tension_level.value + t * (next_point.tension_level.value - prev_point.tension_level.value))
        
        return TensionLevel(max(0, min(8, level)))
    
    def suggest_chord_for_tension(
        self,
        target_tension: TensionLevel,
        key: str = "C",
        mode: str = "major",
        current_chord: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Suggest chords that achieve a target tension level.
        
        Returns:
            List of (chord_symbol, function) tuples
        """
        suggestions = []
        
        tension_to_chords = {
            TensionLevel.RESOLVED: [("I", ""), ("i", "")],
            TensionLevel.LOW: [("vi", "m"), ("IV", ""), ("ii", "m")],
            TensionLevel.MILD: [("IV", ""), ("ii", "m7"), ("vi", "m7")],
            TensionLevel.MODERATE: [("V", ""), ("iii", "m"), ("bVII", "")],
            TensionLevel.ELEVATED: [("V7", "7"), ("bVI", ""), ("ii", "m7")],
            TensionLevel.HIGH: [("V7", "7"), ("V/V", "7"), ("bII", "maj7")],
            TensionLevel.INTENSE: [("vii°", "dim"), ("V7b9", "7b9"), ("#iv°", "dim")],
            TensionLevel.EXTREME: [("vii°7", "dim7"), ("V7alt", "7alt"), ("+6", "")],
            TensionLevel.UNRESOLVED: [("Vsus4", "sus4"), ("iv", "m"), ("bVI", "")],
        }
        
        base_suggestions = tension_to_chords.get(target_tension, [("V", "")])
        
        # Map to actual chord symbols based on key
        key_roots = {
            "C": ["C", "D", "E", "F", "G", "A", "B"],
            "G": ["G", "A", "B", "C", "D", "E", "F#"],
            "D": ["D", "E", "F#", "G", "A", "B", "C#"],
            "F": ["F", "G", "A", "Bb", "C", "D", "E"],
            "Bb": ["Bb", "C", "D", "Eb", "F", "G", "A"],
            "Eb": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
        }
        
        roots = key_roots.get(key, key_roots["C"])
        
        function_to_degree = {
            "I": 0, "i": 0,
            "ii": 1, "II": 1,
            "iii": 2, "III": 2,
            "IV": 3, "iv": 3,
            "V": 4, "v": 4,
            "vi": 5, "VI": 5,
            "vii°": 6, "VII": 6,
            "bII": 1,  # Neapolitan (flat)
            "bVI": 5,  # Flat VI
            "bVII": 6, # Flat VII
        }
        
        for func, quality in base_suggestions:
            base_func = func.replace("7", "").replace("sus4", "").replace("dim", "").replace("°", "").replace("alt", "").replace("b9", "")
            if base_func.startswith("V/"):
                # Secondary dominant
                degree = 4  # V
                suggestions.append((f"{roots[degree]}{quality}", func))
            elif base_func in function_to_degree:
                degree = function_to_degree[base_func]
                root = roots[degree]
                if base_func.startswith("b"):
                    # Flatten the root
                    root = root + "b" if len(root) == 1 else root[0] + "b"
                suggestions.append((f"{root}{quality}", func))
            else:
                suggestions.append((f"{key}{quality}", func))
        
        return suggestions[:5]  # Return top 5
    
    def print_tension(self, profile: TensionProfile) -> str:
        """Generate visual representation of tension profile."""
        
        lines = []
        lines.append(f"═══ TENSION: {profile.emotion.upper()} / {profile.section_type.upper()} ═══")
        lines.append(f"Shape: {profile.curve.shape.value}")
        lines.append(f"Range: {profile.curve.base_level.name} → {profile.curve.peak_level.name}")
        lines.append(f"Release: {profile.curve.release_type.value} | Final: {'Yes' if profile.curve.has_final_release else 'No'}")
        lines.append("")
        
        # Visual curve
        width = 40
        lines.append("Tension Curve:")
        
        tension_labels = ["RES", "LOW", "MLD", "MOD", "ELV", "HI ", "INT", "EXT", "UNR"]
        
        for i, point in enumerate(profile.curve.points):
            level = point.tension_level.value
            bar_len = int((level / 8) * width)
            bar = "█" * bar_len + "░" * (width - bar_len)
            
            release_mark = "←" if point.is_release_point else " "
            tech_mark = f"[{point.technique.value[:3]}]" if point.technique else "     "
            
            lines.append(f"  {point.position:.2f} [{bar}] {tension_labels[level]} {release_mark} {tech_mark}")
        
        lines.append("")
        lines.append(f"Techniques: {', '.join(t.value for t in profile.curve.primary_techniques)}")
        
        if profile.rhythmic_density_curve:
            lines.append("")
            lines.append("Rhythmic Density:")
            for i in range(0, len(profile.rhythmic_density_curve), max(1, len(profile.rhythmic_density_curve) // 10)):
                density = profile.rhythmic_density_curve[i]
                bar_len = int(density * 20)
                bar = "▓" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  [{bar}] {density:.2f}")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tension_profile(
    emotion: str,
    section_type: str = "verse",
    bars: int = 8,
    seed: Optional[int] = None,
) -> TensionProfile:
    """Quick tension profile generation."""
    engine = TensionEngine(seed=seed)
    return engine.generate_tension_profile(
        emotion=emotion,
        section_type=section_type,
        bars=bars,
    )


def analyze_chord(chord: str, function: str = "") -> HarmonicTension:
    """Quick chord tension analysis."""
    engine = TensionEngine()
    return engine.analyze_chord_tension(chord, function)


def get_tension_levels() -> List[str]:
    """List available tension levels."""
    return [t.name for t in TensionLevel]


def get_tension_techniques() -> List[str]:
    """List available tension techniques."""
    return [t.value for t in TensionTechnique]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║      KELLY TENSION ENGINE - TIER 3       ║")
    print("╚══════════════════════════════════════════╝\n")
    
    engine = TensionEngine(seed=42)
    
    # Test different emotions
    test_cases = [
        ("grief", "verse"),
        ("rage", "build"),
        ("hope", "chorus"),
        ("fear", "breakdown"),
        ("anxiety", "verse"),
    ]
    
    for emotion, section in test_cases:
        print(f"\n{'='*60}")
        
        profile = engine.generate_tension_profile(
            emotion=emotion,
            section_type=section,
            bars=8,
        )
        
        print(engine.print_tension(profile))
    
    # Test chord analysis
    print(f"\n{'='*60}")
    print("CHORD TENSION ANALYSIS")
    print('='*60)
    
    test_chords = [
        ("C", "I"),
        ("G7", "V7"),
        ("Dm7", "ii7"),
        ("Bdim7", "vii°7"),
        ("Db", "bII"),
        ("Gsus4", "Vsus4"),
    ]
    
    for chord, func in test_chords:
        analysis = engine.analyze_chord_tension(chord, func)
        techs = ", ".join(t.value for t in analysis.techniques_active) or "none"
        print(f"  {chord:8s} ({func:6s}): {analysis.tension_level.name:10s} | Techniques: {techs}")
        if analysis.resolution_target:
            print(f"           → Wants to resolve to: {analysis.resolution_target}")
