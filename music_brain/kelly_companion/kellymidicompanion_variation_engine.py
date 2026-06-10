"""
Variation Engine - Motif development and thematic transformation.

Part of Kelly MIDI Companion Tier 3.

Philosophy: Repetition without variation is monotony. Variation without
repetition is chaos. The magic is in the balance - recognizable themes
that evolve, melodies that grow, patterns that transform. Grief might
circle back to the same phrase, unable to move on. Joy might spin endless
variations, each more playful than the last. The variation IS the journey.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
import random
import copy

# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================


class VariationType(Enum):
    """Types of musical variation."""

    # Melodic
    TRANSPOSITION = "transposition"  # Move to different pitch
    INVERSION = "inversion"  # Flip melodic contour
    RETROGRADE = "retrograde"  # Reverse direction
    AUGMENTATION = "augmentation"  # Stretch durations
    DIMINUTION = "diminution"  # Compress durations
    ORNAMENTATION = "ornamentation"  # Add decorations
    SIMPLIFICATION = "simplification"  # Strip to essentials
    SEQUENCE = "sequence"  # Repeat at different pitch
    FRAGMENTATION = "fragmentation"  # Use partial motif
    EXTENSION = "extension"  # Lengthen phrase

    # Rhythmic
    SYNCOPATION = "syncopation"  # Shift accents
    DISPLACEMENT = "displacement"  # Move in time
    SUBDIVISION = "subdivision"  # Double note density
    CONSOLIDATION = "consolidation"  # Half note density
    SWING = "swing"  # Add swing feel
    STRAIGHT = "straight"  # Remove swing

    # Harmonic
    REHARMONIZATION = "reharmonization"  # Change chords
    MODE_CHANGE = "mode_change"  # Major/minor shift
    MODAL_INTERCHANGE = "modal_interchange"  # Borrow from parallel
    CHROMATIC_ALTERATION = "chromatic_alteration"  # Add chromaticism

    # Textural
    REGISTER_SHIFT = "register_shift"  # Move octave
    DOUBLING = "doubling"  # Add parallel voice
    THINNING = "thinning"  # Remove voices
    INSTRUMENTATION = "instrumentation"  # Change timbre
    ARTICULATION = "articulation"  # Change attack/release

    # Structural
    TRUNCATION = "truncation"  # Cut short
    REPETITION = "repetition"  # Exact repeat
    DEVELOPMENT = "development"  # Complex transformation
    RECOMBINATION = "recombination"  # Mix motif elements


class VariationIntensity(Enum):
    """How much variation to apply."""

    NONE = 0  # Exact repeat
    SUBTLE = 1  # Nearly identical
    LIGHT = 2  # Small changes
    MODERATE = 3  # Noticeable changes
    SIGNIFICANT = 4  # Major changes
    DRAMATIC = 5  # Barely recognizable
    TOTAL = 6  # Complete transformation


class MotifType(Enum):
    """Types of musical motifs."""

    MELODIC = "melodic"  # Pitch sequence
    RHYTHMIC = "rhythmic"  # Rhythm pattern
    HARMONIC = "harmonic"  # Chord progression
    TEXTURAL = "textural"  # Density/timbre pattern
    CONTOUR = "contour"  # Shape (up/down/static)
    INTERVAL = "interval"  # Interval sequence


class DevelopmentStrategy(Enum):
    """Strategies for developing material across a song."""

    ADDITIVE = "additive"  # Gradually add elements
    SUBTRACTIVE = "subtractive"  # Gradually remove elements
    TRANSFORMATIVE = "transformative"  # Continuous evolution
    CYCLICAL = "cyclical"  # Return to origins
    SPIRAL = "spiral"  # Return but elevated
    LINEAR = "linear"  # One direction
    ARCH = "arch"  # Build then return


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Motif:
    """A musical motif that can be varied."""

    id: str
    motif_type: MotifType
    content: Any  # Pitch list, rhythm list, chord list, etc.
    length_beats: float

    # Analysis
    contour: List[int] = field(default_factory=list)  # -1, 0, 1 for down/same/up
    intervals: List[int] = field(default_factory=list)  # Semitone intervals
    rhythm_pattern: List[float] = field(default_factory=list)  # Note durations

    # Metadata
    source_section: str = ""
    emotional_character: str = ""
    importance: float = 1.0  # 0.0-1.0, how central this motif is


@dataclass
class VariationRule:
    """A rule for how to vary a motif."""

    variation_type: VariationType
    intensity: VariationIntensity
    probability: float = 1.0  # Chance this variation is applied
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Constraints
    preserve_contour: bool = False
    preserve_rhythm: bool = False
    preserve_intervals: bool = False


@dataclass
class VariationInstance:
    """A specific variation of a motif."""

    original_motif_id: str
    variation_types: List[VariationType]
    intensity: VariationIntensity
    result_content: Any

    # How it differs
    pitch_delta: int = 0  # Transposition amount
    duration_scale: float = 1.0  # Time stretch factor
    density_change: float = 0.0  # Note density change

    # Position
    section_index: int = 0
    bar_position: int = 0


@dataclass
class VariationPlan:
    """Plan for variations across a song section or entire song."""

    motifs: List[Motif]
    variations: List[VariationInstance]
    strategy: DevelopmentStrategy

    # Per-section intensity
    section_intensities: List[VariationIntensity] = field(default_factory=list)

    # Rules
    rules: List[VariationRule] = field(default_factory=list)

    # Metadata
    emotion: str = ""
    total_bars: int = 0


# =============================================================================
# EMOTION PROFILES
# =============================================================================

EMOTION_VARIATION_PROFILES: Dict[str, Dict] = {
    "grief": {
        "base_intensity": VariationIntensity.SUBTLE,
        "max_intensity": VariationIntensity.LIGHT,
        "strategy": DevelopmentStrategy.CYCLICAL,
        "preferred_variations": [
            VariationType.TRANSPOSITION,
            VariationType.REGISTER_SHIFT,
            VariationType.SIMPLIFICATION,
            VariationType.REPETITION,
        ],
        "avoid_variations": [
            VariationType.ORNAMENTATION,
            VariationType.SYNCOPATION,
            VariationType.SUBDIVISION,
        ],
        "variation_rate": 0.3,  # Low - grief repeats
        "preserve_contour": True,
        "preserve_rhythm": True,
        "development_arc": [1, 1, 2, 1, 1],  # Barely changes
    },
    "sadness": {
        "base_intensity": VariationIntensity.LIGHT,
        "max_intensity": VariationIntensity.MODERATE,
        "strategy": DevelopmentStrategy.ARCH,
        "preferred_variations": [
            VariationType.TRANSPOSITION,
            VariationType.INVERSION,
            VariationType.EXTENSION,
            VariationType.REGISTER_SHIFT,
        ],
        "avoid_variations": [
            VariationType.SYNCOPATION,
            VariationType.SUBDIVISION,
        ],
        "variation_rate": 0.4,
        "preserve_contour": True,
        "preserve_rhythm": False,
        "development_arc": [1, 2, 3, 2, 1],
    },
    "melancholy": {
        "base_intensity": VariationIntensity.SUBTLE,
        "max_intensity": VariationIntensity.LIGHT,
        "strategy": DevelopmentStrategy.SPIRAL,
        "preferred_variations": [
            VariationType.TRANSPOSITION,
            VariationType.MODAL_INTERCHANGE,
            VariationType.SEQUENCE,
        ],
        "avoid_variations": [
            VariationType.SYNCOPATION,
            VariationType.FRAGMENTATION,
        ],
        "variation_rate": 0.35,
        "preserve_contour": True,
        "preserve_rhythm": True,
        "development_arc": [1, 1, 2, 2, 1],
    },
    "rage": {
        "base_intensity": VariationIntensity.MODERATE,
        "max_intensity": VariationIntensity.DRAMATIC,
        "strategy": DevelopmentStrategy.ADDITIVE,
        "preferred_variations": [
            VariationType.FRAGMENTATION,
            VariationType.SUBDIVISION,
            VariationType.SYNCOPATION,
            VariationType.DOUBLING,
            VariationType.CHROMATIC_ALTERATION,
        ],
        "avoid_variations": [
            VariationType.SIMPLIFICATION,
            VariationType.AUGMENTATION,
        ],
        "variation_rate": 0.7,
        "preserve_contour": False,
        "preserve_rhythm": False,
        "development_arc": [2, 3, 4, 5, 5],
    },
    "anger": {
        "base_intensity": VariationIntensity.LIGHT,
        "max_intensity": VariationIntensity.SIGNIFICANT,
        "strategy": DevelopmentStrategy.LINEAR,
        "preferred_variations": [
            VariationType.SUBDIVISION,
            VariationType.SYNCOPATION,
            VariationType.FRAGMENTATION,
            VariationType.DISPLACEMENT,
        ],
        "avoid_variations": [
            VariationType.AUGMENTATION,
            VariationType.THINNING,
        ],
        "variation_rate": 0.6,
        "preserve_contour": False,
        "preserve_rhythm": False,
        "development_arc": [2, 3, 3, 4, 4],
    },
    "fear": {
        "base_intensity": VariationIntensity.LIGHT,
        "max_intensity": VariationIntensity.SIGNIFICANT,
        "strategy": DevelopmentStrategy.TRANSFORMATIVE,
        "preferred_variations": [
            VariationType.FRAGMENTATION,
            VariationType.DISPLACEMENT,
            VariationType.CHROMATIC_ALTERATION,
            VariationType.REGISTER_SHIFT,
        ],
        "avoid_variations": [
            VariationType.ORNAMENTATION,
            VariationType.SWING,
        ],
        "variation_rate": 0.5,
        "preserve_contour": False,
        "preserve_rhythm": True,
        "development_arc": [1, 2, 3, 4, 3],
    },
    "anxiety": {
        "base_intensity": VariationIntensity.MODERATE,
        "max_intensity": VariationIntensity.SIGNIFICANT,
        "strategy": DevelopmentStrategy.SPIRAL,
        "preferred_variations": [
            VariationType.SUBDIVISION,
            VariationType.DISPLACEMENT,
            VariationType.SEQUENCE,
            VariationType.FRAGMENTATION,
        ],
        "avoid_variations": [
            VariationType.AUGMENTATION,
            VariationType.SIMPLIFICATION,
        ],
        "variation_rate": 0.65,
        "preserve_contour": False,
        "preserve_rhythm": False,
        "development_arc": [3, 3, 4, 4, 3],
    },
    "hope": {
        "base_intensity": VariationIntensity.LIGHT,
        "max_intensity": VariationIntensity.MODERATE,
        "strategy": DevelopmentStrategy.ADDITIVE,
        "preferred_variations": [
            VariationType.TRANSPOSITION,
            VariationType.EXTENSION,
            VariationType.ORNAMENTATION,
            VariationType.SEQUENCE,
            VariationType.DOUBLING,
        ],
        "avoid_variations": [
            VariationType.FRAGMENTATION,
            VariationType.TRUNCATION,
        ],
        "variation_rate": 0.5,
        "preserve_contour": True,
        "preserve_rhythm": False,
        "development_arc": [1, 2, 3, 4, 3],
    },
    "joy": {
        "base_intensity": VariationIntensity.MODERATE,
        "max_intensity": VariationIntensity.SIGNIFICANT,
        "strategy": DevelopmentStrategy.SPIRAL,
        "preferred_variations": [
            VariationType.ORNAMENTATION,
            VariationType.SYNCOPATION,
            VariationType.SUBDIVISION,
            VariationType.TRANSPOSITION,
            VariationType.EXTENSION,
        ],
        "avoid_variations": [
            VariationType.SIMPLIFICATION,
            VariationType.TRUNCATION,
        ],
        "variation_rate": 0.6,
        "preserve_contour": False,
        "preserve_rhythm": False,
        "development_arc": [2, 3, 4, 4, 3],
    },
    "peace": {
        "base_intensity": VariationIntensity.SUBTLE,
        "max_intensity": VariationIntensity.LIGHT,
        "strategy": DevelopmentStrategy.CYCLICAL,
        "preferred_variations": [
            VariationType.TRANSPOSITION,
            VariationType.REGISTER_SHIFT,
            VariationType.AUGMENTATION,
        ],
        "avoid_variations": [
            VariationType.SYNCOPATION,
            VariationType.SUBDIVISION,
            VariationType.FRAGMENTATION,
        ],
        "variation_rate": 0.25,
        "preserve_contour": True,
        "preserve_rhythm": True,
        "development_arc": [1, 1, 1, 1, 1],
    },
    "nostalgia": {
        "base_intensity": VariationIntensity.LIGHT,
        "max_intensity": VariationIntensity.MODERATE,
        "strategy": DevelopmentStrategy.ARCH,
        "preferred_variations": [
            VariationType.TRANSPOSITION,
            VariationType.MODAL_INTERCHANGE,
            VariationType.ORNAMENTATION,
            VariationType.SEQUENCE,
        ],
        "avoid_variations": [
            VariationType.FRAGMENTATION,
            VariationType.CHROMATIC_ALTERATION,
        ],
        "variation_rate": 0.4,
        "preserve_contour": True,
        "preserve_rhythm": True,
        "development_arc": [1, 2, 3, 2, 1],
    },
    "longing": {
        "base_intensity": VariationIntensity.LIGHT,
        "max_intensity": VariationIntensity.MODERATE,
        "strategy": DevelopmentStrategy.LINEAR,
        "preferred_variations": [
            VariationType.EXTENSION,
            VariationType.SEQUENCE,
            VariationType.TRANSPOSITION,
        ],
        "avoid_variations": [
            VariationType.TRUNCATION,
            VariationType.CONSOLIDATION,
        ],
        "variation_rate": 0.45,
        "preserve_contour": True,
        "preserve_rhythm": False,
        "development_arc": [1, 2, 2, 3, 3],
    },
    "tension": {
        "base_intensity": VariationIntensity.MODERATE,
        "max_intensity": VariationIntensity.DRAMATIC,
        "strategy": DevelopmentStrategy.ADDITIVE,
        "preferred_variations": [
            VariationType.SUBDIVISION,
            VariationType.CHROMATIC_ALTERATION,
            VariationType.SEQUENCE,
            VariationType.FRAGMENTATION,
        ],
        "avoid_variations": [
            VariationType.SIMPLIFICATION,
            VariationType.AUGMENTATION,
        ],
        "variation_rate": 0.6,
        "preserve_contour": False,
        "preserve_rhythm": False,
        "development_arc": [2, 3, 4, 5, 5],
    },
    "defiance": {
        "base_intensity": VariationIntensity.MODERATE,
        "max_intensity": VariationIntensity.SIGNIFICANT,
        "strategy": DevelopmentStrategy.LINEAR,
        "preferred_variations": [
            VariationType.SYNCOPATION,
            VariationType.FRAGMENTATION,
            VariationType.DISPLACEMENT,
            VariationType.DOUBLING,
        ],
        "avoid_variations": [
            VariationType.SIMPLIFICATION,
            VariationType.THINNING,
        ],
        "variation_rate": 0.55,
        "preserve_contour": False,
        "preserve_rhythm": False,
        "development_arc": [2, 3, 4, 4, 4],
    },
    "vulnerability": {
        "base_intensity": VariationIntensity.SUBTLE,
        "max_intensity": VariationIntensity.LIGHT,
        "strategy": DevelopmentStrategy.CYCLICAL,
        "preferred_variations": [
            VariationType.SIMPLIFICATION,
            VariationType.REGISTER_SHIFT,
            VariationType.THINNING,
        ],
        "avoid_variations": [
            VariationType.SUBDIVISION,
            VariationType.ORNAMENTATION,
            VariationType.DOUBLING,
        ],
        "variation_rate": 0.3,
        "preserve_contour": True,
        "preserve_rhythm": True,
        "development_arc": [1, 1, 2, 1, 1],
    },
    "euphoria": {
        "base_intensity": VariationIntensity.MODERATE,
        "max_intensity": VariationIntensity.DRAMATIC,
        "strategy": DevelopmentStrategy.ADDITIVE,
        "preferred_variations": [
            VariationType.ORNAMENTATION,
            VariationType.SUBDIVISION,
            VariationType.DOUBLING,
            VariationType.EXTENSION,
            VariationType.TRANSPOSITION,
        ],
        "avoid_variations": [
            VariationType.SIMPLIFICATION,
            VariationType.TRUNCATION,
        ],
        "variation_rate": 0.7,
        "preserve_contour": False,
        "preserve_rhythm": False,
        "development_arc": [2, 4, 5, 5, 4],
    },
    "emptiness": {
        "base_intensity": VariationIntensity.NONE,
        "max_intensity": VariationIntensity.SUBTLE,
        "strategy": DevelopmentStrategy.SUBTRACTIVE,
        "preferred_variations": [
            VariationType.SIMPLIFICATION,
            VariationType.THINNING,
            VariationType.AUGMENTATION,
            VariationType.REPETITION,
        ],
        "avoid_variations": [
            VariationType.ORNAMENTATION,
            VariationType.SUBDIVISION,
            VariationType.DOUBLING,
        ],
        "variation_rate": 0.15,
        "preserve_contour": True,
        "preserve_rhythm": True,
        "development_arc": [1, 1, 0, 0, 0],
    },
}


# =============================================================================
# VARIATION TRANSFORMATIONS
# =============================================================================


class VariationTransforms:
    """Collection of transformation functions."""

    @staticmethod
    def transpose(pitches: List[int], semitones: int) -> List[int]:
        """Transpose pitches by semitones."""
        return [p + semitones for p in pitches]

    @staticmethod
    def invert(pitches: List[int], axis: Optional[int] = None) -> List[int]:
        """Invert melodic contour around axis."""
        if not pitches:
            return pitches
        if axis is None:
            axis = pitches[0]
        return [axis - (p - axis) for p in pitches]

    @staticmethod
    def retrograde(sequence: List[Any]) -> List[Any]:
        """Reverse sequence."""
        return list(reversed(sequence))

    @staticmethod
    def augment(durations: List[float], factor: float = 2.0) -> List[float]:
        """Stretch durations."""
        return [d * factor for d in durations]

    @staticmethod
    def diminish(durations: List[float], factor: float = 0.5) -> List[float]:
        """Compress durations."""
        return [d * factor for d in durations]

    @staticmethod
    def ornament(pitches: List[int], density: float = 0.3) -> List[int]:
        """Add neighbor tones and passing tones."""
        if len(pitches) < 2:
            return pitches

        result = []
        for i, p in enumerate(pitches):
            result.append(p)
            if i < len(pitches) - 1 and random.random() < density:
                # Add passing/neighbor tone
                next_p = pitches[i + 1]
                diff = next_p - p
                if abs(diff) > 2:
                    # Passing tone
                    result.append(p + diff // 2)
                else:
                    # Neighbor tone
                    result.append(p + random.choice([-1, 1, 2, -2]))

        return result

    @staticmethod
    def simplify(pitches: List[int], keep_ratio: float = 0.5) -> List[int]:
        """Keep only essential notes."""
        if len(pitches) <= 2:
            return pitches

        # Always keep first and last
        result = [pitches[0]]

        # Keep some middle notes
        for p in pitches[1:-1]:
            if random.random() < keep_ratio:
                result.append(p)

        result.append(pitches[-1])
        return result

    @staticmethod
    def sequence(pitches: List[int], interval: int, repetitions: int = 2) -> List[int]:
        """Create sequential repetitions."""
        result = list(pitches)
        for i in range(1, repetitions):
            transposed = [p + interval * i for p in pitches]
            result.extend(transposed)
        return result

    @staticmethod
    def fragment(
        sequence: List[Any], start_ratio: float = 0.0, end_ratio: float = 0.5
    ) -> List[Any]:
        """Extract a fragment."""
        if not sequence:
            return sequence
        start = int(len(sequence) * start_ratio)
        end = int(len(sequence) * end_ratio)
        return sequence[start : max(start + 1, end)]

    @staticmethod
    def extend(sequence: List[Any], extension_ratio: float = 0.5) -> List[Any]:
        """Extend by repeating end portion."""
        if not sequence:
            return sequence
        extension_len = max(1, int(len(sequence) * extension_ratio))
        extension = sequence[-extension_len:]
        return sequence + extension

    @staticmethod
    def syncopate(onsets: List[float], shift: float = 0.25) -> List[float]:
        """Shift onsets to create syncopation."""
        return [max(0, o - shift) if i % 2 == 1 else o for i, o in enumerate(onsets)]

    @staticmethod
    def displace(onsets: List[float], shift: float = 0.5) -> List[float]:
        """Shift all onsets."""
        return [o + shift for o in onsets]

    @staticmethod
    def subdivide(durations: List[float]) -> List[float]:
        """Double the number of notes."""
        result = []
        for d in durations:
            result.extend([d / 2, d / 2])
        return result

    @staticmethod
    def consolidate(durations: List[float]) -> List[float]:
        """Halve the number of notes."""
        result = []
        for i in range(0, len(durations), 2):
            if i + 1 < len(durations):
                result.append(durations[i] + durations[i + 1])
            else:
                result.append(durations[i])
        return result

    @staticmethod
    def shift_register(pitches: List[int], octaves: int = 1) -> List[int]:
        """Shift by octaves."""
        return [p + 12 * octaves for p in pitches]

    @staticmethod
    def double(pitches: List[int], interval: int = 12) -> List[Tuple[int, int]]:
        """Add parallel voice."""
        return [(p, p + interval) for p in pitches]


# =============================================================================
# VARIATION ENGINE
# =============================================================================


class VariationEngine:
    """
    Generates variations and developments of musical material.

    Ensures musical coherence while preventing monotony through
    emotion-appropriate transformations.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.transforms = VariationTransforms()

    def create_motif(
        self,
        pitches: Optional[List[int]] = None,
        durations: Optional[List[float]] = None,
        motif_type: MotifType = MotifType.MELODIC,
        motif_id: Optional[str] = None,
    ) -> Motif:
        """
        Create a motif from musical material.

        Args:
            pitches: List of MIDI pitches
            durations: List of note durations
            motif_type: Type of motif
            motif_id: Optional identifier

        Returns:
            Analyzed Motif object
        """
        if motif_id is None:
            motif_id = f"motif_{self.rng.randint(1000, 9999)}"

        # Calculate contour
        contour = []
        if pitches and len(pitches) > 1:
            for i in range(1, len(pitches)):
                diff = pitches[i] - pitches[i - 1]
                if diff > 0:
                    contour.append(1)
                elif diff < 0:
                    contour.append(-1)
                else:
                    contour.append(0)

        # Calculate intervals
        intervals = []
        if pitches and len(pitches) > 1:
            for i in range(1, len(pitches)):
                intervals.append(pitches[i] - pitches[i - 1])

        # Calculate length
        length_beats = sum(durations) if durations else 4.0

        return Motif(
            id=motif_id,
            motif_type=motif_type,
            content={"pitches": pitches or [], "durations": durations or []},
            length_beats=length_beats,
            contour=contour,
            intervals=intervals,
            rhythm_pattern=durations or [],
        )

    def generate_variation_plan(
        self,
        emotion: str,
        num_sections: int,
        motifs: Optional[List[Motif]] = None,
        bars_per_section: int = 8,
    ) -> VariationPlan:
        """
        Generate a plan for how material should vary across sections.

        Args:
            emotion: Primary emotion
            num_sections: Number of sections in song
            motifs: Optional existing motifs
            bars_per_section: Bars per section

        Returns:
            VariationPlan with rules and instances
        """
        emotion_lower = emotion.lower()
        profile = EMOTION_VARIATION_PROFILES.get(
            emotion_lower, EMOTION_VARIATION_PROFILES["sadness"]
        )

        # Create default motifs if none provided
        if not motifs:
            motifs = [
                self.create_motif(
                    pitches=[60, 62, 64, 65, 67],
                    durations=[1.0, 1.0, 0.5, 0.5, 2.0],
                    motif_id="main_motif",
                )
            ]

        # Get development arc
        dev_arc = profile["development_arc"]
        if len(dev_arc) < num_sections:
            dev_arc = dev_arc + [dev_arc[-1]] * (num_sections - len(dev_arc))

        # Map arc to intensities
        section_intensities = [
            VariationIntensity(min(dev_arc[i], profile["max_intensity"].value))
            for i in range(num_sections)
        ]

        # Generate rules
        rules = []
        for var_type in profile["preferred_variations"]:
            rule = VariationRule(
                variation_type=var_type,
                intensity=profile["base_intensity"],
                probability=profile["variation_rate"],
                preserve_contour=profile["preserve_contour"],
                preserve_rhythm=profile["preserve_rhythm"],
            )
            rules.append(rule)

        # Generate variation instances
        variations = []
        for section_idx in range(num_sections):
            intensity = section_intensities[section_idx]

            for motif in motifs:
                if self.rng.random() < profile["variation_rate"]:
                    # Select variations to apply
                    num_variations = min(intensity.value, len(profile["preferred_variations"]))
                    selected_types = self.rng.sample(
                        profile["preferred_variations"],
                        k=min(num_variations, len(profile["preferred_variations"])),
                    )

                    # Create variation
                    var_content = self._apply_variations(
                        motif.content,
                        selected_types,
                        intensity,
                        profile,
                    )

                    variation = VariationInstance(
                        original_motif_id=motif.id,
                        variation_types=selected_types,
                        intensity=intensity,
                        result_content=var_content,
                        section_index=section_idx,
                    )
                    variations.append(variation)

        return VariationPlan(
            motifs=motifs,
            variations=variations,
            strategy=profile["strategy"],
            section_intensities=section_intensities,
            rules=rules,
            emotion=emotion_lower,
            total_bars=num_sections * bars_per_section,
        )

    def vary_motif(
        self,
        motif: Motif,
        variation_types: List[VariationType],
        intensity: VariationIntensity = VariationIntensity.MODERATE,
        preserve_contour: bool = False,
        preserve_rhythm: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply variations to a motif.

        Args:
            motif: Source motif
            variation_types: Types of variation to apply
            intensity: How intense the variation should be
            preserve_contour: Keep melodic shape
            preserve_rhythm: Keep rhythm pattern

        Returns:
            Modified content dictionary
        """
        content = copy.deepcopy(motif.content)
        pitches = content.get("pitches", [])
        durations = content.get("durations", [])

        for var_type in variation_types:
            # Apply based on type
            if var_type == VariationType.TRANSPOSITION:
                semitones = self.rng.choice([-7, -5, -4, -3, 3, 4, 5, 7])
                semitones = int(semitones * (intensity.value / 3))
                pitches = self.transforms.transpose(pitches, semitones)
                content["pitch_delta"] = semitones

            elif var_type == VariationType.INVERSION:
                if not preserve_contour:
                    pitches = self.transforms.invert(pitches)

            elif var_type == VariationType.RETROGRADE:
                if not preserve_contour:
                    pitches = self.transforms.retrograde(pitches)
                if not preserve_rhythm:
                    durations = self.transforms.retrograde(durations)

            elif var_type == VariationType.AUGMENTATION:
                if not preserve_rhythm:
                    factor = 1.5 + (intensity.value * 0.25)
                    durations = self.transforms.augment(durations, factor)

            elif var_type == VariationType.DIMINUTION:
                if not preserve_rhythm:
                    factor = max(0.25, 1.0 - (intensity.value * 0.15))
                    durations = self.transforms.diminish(durations, factor)

            elif var_type == VariationType.ORNAMENTATION:
                density = 0.2 + (intensity.value * 0.1)
                pitches = self.transforms.ornament(pitches, density)
                # Adjust durations to match
                while len(durations) < len(pitches):
                    durations.append(durations[-1] if durations else 0.5)

            elif var_type == VariationType.SIMPLIFICATION:
                keep_ratio = max(0.3, 1.0 - (intensity.value * 0.15))
                pitches = self.transforms.simplify(pitches, keep_ratio)
                durations = durations[: len(pitches)]

            elif var_type == VariationType.SEQUENCE:
                interval = self.rng.choice([2, 3, 4, 5, -2, -3, -4, -5])
                repetitions = min(2 + intensity.value // 2, 4)
                pitches = self.transforms.sequence(pitches, interval, repetitions)
                durations = durations * repetitions

            elif var_type == VariationType.FRAGMENTATION:
                end_ratio = max(0.3, 0.8 - (intensity.value * 0.1))
                pitches = self.transforms.fragment(pitches, 0.0, end_ratio)
                durations = self.transforms.fragment(durations, 0.0, end_ratio)

            elif var_type == VariationType.EXTENSION:
                ext_ratio = 0.3 + (intensity.value * 0.1)
                pitches = self.transforms.extend(pitches, ext_ratio)
                durations = self.transforms.extend(durations, ext_ratio)

            elif var_type == VariationType.SYNCOPATION:
                if not preserve_rhythm:
                    # This would need onset times, simplified here
                    pass

            elif var_type == VariationType.SUBDIVISION:
                if not preserve_rhythm:
                    durations = self.transforms.subdivide(durations)
                    # Double pitches
                    new_pitches = []
                    for p in pitches:
                        new_pitches.extend([p, p])
                    pitches = new_pitches

            elif var_type == VariationType.CONSOLIDATION:
                if not preserve_rhythm:
                    durations = self.transforms.consolidate(durations)
                    pitches = pitches[::2]

            elif var_type == VariationType.REGISTER_SHIFT:
                octaves = self.rng.choice([-2, -1, 1, 2])
                octaves = max(-2, min(2, int(octaves * intensity.value / 3)))
                pitches = self.transforms.shift_register(pitches, octaves)

        content["pitches"] = pitches
        content["durations"] = durations
        return content

    def _apply_variations(
        self,
        content: Dict[str, Any],
        variation_types: List[VariationType],
        intensity: VariationIntensity,
        profile: Dict,
    ) -> Dict[str, Any]:
        """Apply variations using profile settings."""

        result = copy.deepcopy(content)
        pitches = result.get("pitches", [])
        durations = result.get("durations", [])

        for var_type in variation_types:
            if var_type == VariationType.TRANSPOSITION:
                semitones = self.rng.choice([-5, -4, -3, 3, 4, 5, 7])
                pitches = self.transforms.transpose(pitches, semitones)

            elif var_type == VariationType.INVERSION and not profile["preserve_contour"]:
                pitches = self.transforms.invert(pitches)

            elif var_type == VariationType.ORNAMENTATION:
                pitches = self.transforms.ornament(pitches, 0.3)
                while len(durations) < len(pitches):
                    durations.append(0.25)

            elif var_type == VariationType.SIMPLIFICATION:
                pitches = self.transforms.simplify(pitches, 0.5)
                durations = durations[: len(pitches)]

            elif var_type == VariationType.REGISTER_SHIFT:
                octaves = self.rng.choice([-1, 1])
                pitches = self.transforms.shift_register(pitches, octaves)

            elif var_type == VariationType.EXTENSION:
                pitches = self.transforms.extend(pitches, 0.5)
                durations = self.transforms.extend(durations, 0.5)

            elif var_type == VariationType.FRAGMENTATION:
                pitches = self.transforms.fragment(pitches, 0, 0.6)
                durations = self.transforms.fragment(durations, 0, 0.6)

            elif var_type == VariationType.SUBDIVISION and not profile["preserve_rhythm"]:
                durations = self.transforms.subdivide(durations)
                pitches = [p for p in pitches for _ in range(2)]

            elif var_type == VariationType.SEQUENCE:
                pitches = self.transforms.sequence(pitches, 2, 2)
                durations = durations * 2

        result["pitches"] = pitches
        result["durations"] = durations
        return result

    def get_intensity_for_section(
        self,
        plan: VariationPlan,
        section_index: int,
    ) -> VariationIntensity:
        """Get variation intensity for a section."""
        if section_index < len(plan.section_intensities):
            return plan.section_intensities[section_index]
        return VariationIntensity.MODERATE

    def print_variation_plan(self, plan: VariationPlan) -> str:
        """Generate visual representation of variation plan."""

        lines = []
        lines.append(f"═══ VARIATION PLAN: {plan.emotion.upper()} ═══")
        lines.append(f"Strategy: {plan.strategy.value}")
        lines.append(f"Motifs: {len(plan.motifs)} | Variations: {len(plan.variations)}")
        lines.append("")

        # Section intensities
        lines.append("Section Development Arc:")
        intensity_chars = ["·", "░", "▒", "▓", "█", "▆", "▇"]
        arc_line = "  "
        for i, intensity in enumerate(plan.section_intensities):
            char = intensity_chars[min(intensity.value, len(intensity_chars) - 1)]
            arc_line += f"[{char}]"
        lines.append(arc_line)

        labels = "  "
        for i in range(len(plan.section_intensities)):
            labels += f" {i+1} "
        lines.append(labels)
        lines.append("")

        # Motifs
        lines.append("Motifs:")
        for motif in plan.motifs:
            pitches = motif.content.get("pitches", [])
            pitch_str = ",".join(str(p) for p in pitches[:5])
            if len(pitches) > 5:
                pitch_str += "..."
            lines.append(f"  [{motif.id}] {motif.motif_type.value}: [{pitch_str}]")
        lines.append("")

        # Variations by section
        lines.append("Variations by Section:")
        section_vars = {}
        for var in plan.variations:
            idx = var.section_index
            if idx not in section_vars:
                section_vars[idx] = []
            section_vars[idx].append(var)

        for section_idx in sorted(section_vars.keys()):
            vars_in_section = section_vars[section_idx]
            intensity = (
                plan.section_intensities[section_idx]
                if section_idx < len(plan.section_intensities)
                else VariationIntensity.NONE
            )
            lines.append(f"  Section {section_idx + 1} ({intensity.name}):")
            for var in vars_in_section:
                var_types = ", ".join(vt.value[:6] for vt in var.variation_types[:3])
                lines.append(f"    • {var.original_motif_id}: {var_types}")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_variation_plan(
    emotion: str,
    num_sections: int = 5,
    seed: Optional[int] = None,
) -> VariationPlan:
    """Quick variation plan generation."""
    engine = VariationEngine(seed=seed)
    return engine.generate_variation_plan(
        emotion=emotion,
        num_sections=num_sections,
    )


def get_variation_types() -> List[str]:
    """List available variation types."""
    return [v.value for v in VariationType]


def get_development_strategies() -> List[str]:
    """List available development strategies."""
    return [s.value for s in DevelopmentStrategy]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║     KELLY VARIATION ENGINE - TIER 3      ║")
    print("╚══════════════════════════════════════════╝\n")

    engine = VariationEngine(seed=42)

    # Test different emotions
    test_emotions = ["grief", "rage", "hope", "anxiety", "peace"]

    for emotion in test_emotions:
        print(f"\n{'='*60}")

        # Create a test motif
        motif = engine.create_motif(
            pitches=[60, 62, 64, 65, 67, 65, 64],
            durations=[0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0],
            motif_id="test_melody",
        )

        plan = engine.generate_variation_plan(
            emotion=emotion,
            num_sections=5,
            motifs=[motif],
        )

        print(engine.print_variation_plan(plan))

    # Demo specific variations
    print(f"\n{'='*60}")
    print("TRANSFORMATION EXAMPLES")
    print("=" * 60)

    original = [60, 62, 64, 65, 67]
    print(f"\nOriginal pitches: {original}")
    print(f"Transposed +5:    {VariationTransforms.transpose(original, 5)}")
    print(f"Inverted:         {VariationTransforms.invert(original)}")
    print(f"Retrograde:       {VariationTransforms.retrograde(original)}")
    print(f"Ornamented:       {VariationTransforms.ornament(original, 0.5)}")
    print(f"Simplified:       {VariationTransforms.simplify(original, 0.5)}")
    print(f"Sequenced:        {VariationTransforms.sequence(original, 2, 2)}")
