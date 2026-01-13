"""
Song Intent Schema - Structured deep interrogation for songwriting.

Implements the three-phase interrogation model:
- Phase 0: Core Wound/Desire (deep interrogation)
- Phase 1: Emotional & Intent (validation)
- Phase 2: Technical Constraints (implementation)

Plus comprehensive rule-breaking enums for intentional creative choices.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path


# =================================================================
# VALIDATION CONSTANTS: Enum Values from YAML Schema
# =================================================================

# Mood/Emotion Primaries (17 options from YAML schema)
VALID_MOOD_PRIMARY_OPTIONS = [
    "Grief",
    "Joy",
    "Nervousness",
    "Defiance",
    "Liberation",
    "Longing",
    "Rage",
    "Acceptance",
    "Nostalgia",
    "Dissociation",
    "Triumphant Hope",
    "Bittersweet",
    "Melancholy",
    "Euphoria",
    "Desperation",
    "Serenity",
    "Confusion",
    "Determination",
]

# Imagery Textures (15 options from YAML schema)
VALID_IMAGERY_TEXTURE_OPTIONS = [
    "Sharp Edges",
    "Muffled",
    "Open/Vast",
    "Claustrophobic",
    "Hazy/Dreamy",
    "Crystalline",
    "Muddy/Thick",
    "Sparse/Empty",
    "Chaotic",
    "Flowing/Liquid",
    "Fractured",
    "Warm/Enveloping",
    "Cold/Distant",
    "Blinding Light",
    "Deep Shadow",
]

# Vulnerability Scale
VALID_VULNERABILITY_SCALE_OPTIONS = ["Low", "Medium", "High"]

# Narrative Arc Options (8 options from YAML schema)
VALID_NARRATIVE_ARC_OPTIONS = [
    "Climb-to-Climax",
    "Slow Reveal",
    "Repetitive Despair",
    "Static Reflection",
    "Sudden Shift",
    "Descent",
    "Rise and Fall",
    "Spiral",
]

# Core Stakes Options (6 options from YAML schema)
VALID_CORE_STAKES_OPTIONS = [
    "Personal",
    "Relational",
    "Existential",
    "Survival",
    "Creative",
    "Moral",
]

# Genre Options (15 options from YAML schema)
VALID_GENRE_OPTIONS = [
    "Cinematic Neo-Soul",
    "Lo-Fi Bedroom",
    "Industrial Pop",
    "Synthwave",
    "Confessional Acoustic",
    "Art Rock",
    "Indie Folk",
    "Post-Punk",
    "Chamber Pop",
    "Electronic",
    "Hip-Hop",
    "R&B",
    "Alternative",
    "Shoegaze",
    "Dream Pop",
]

# Groove Feel Options (8 options from YAML schema)
VALID_GROOVE_FEEL_OPTIONS = [
    "Straight/Driving",
    "Laid Back",
    "Swung",
    "Syncopated",
    "Rubato/Free",
    "Mechanical",
    "Organic/Breathing",
    "Push-Pull",
]


# =================================================================
# ENUMS: Rule Breaking Categories
# =================================================================


class HarmonyRuleBreak(Enum):
    """Harmony rules to intentionally break."""

    AVOID_TONIC_RESOLUTION = "HARMONY_AvoidTonicResolution"
    PARALLEL_MOTION = "HARMONY_ParallelMotion"
    MODAL_INTERCHANGE = "HARMONY_ModalInterchange"
    TRITONE_SUBSTITUTION = "HARMONY_TritoneSubstitution"
    POLYTONALITY = "HARMONY_Polytonality"
    UNRESOLVED_DISSONANCE = "HARMONY_UnresolvedDissonance"


class RhythmRuleBreak(Enum):
    """Rhythm rules to intentionally break."""

    CONSTANT_DISPLACEMENT = "RHYTHM_ConstantDisplacement"
    TEMPO_FLUCTUATION = "RHYTHM_TempoFluctuation"
    METRIC_MODULATION = "RHYTHM_MetricModulation"
    POLYRHYTHMIC_LAYERS = "RHYTHM_PolyrhythmicLayers"
    DROPPED_BEATS = "RHYTHM_DroppedBeats"


class ArrangementRuleBreak(Enum):
    """Arrangement rules to intentionally break."""

    UNBALANCED_DYNAMICS = "ARRANGEMENT_UnbalancedDynamics"
    STRUCTURAL_MISMATCH = "ARRANGEMENT_StructuralMismatch"
    BURIED_VOCALS = "ARRANGEMENT_BuriedVocals"
    EXTREME_DYNAMIC_RANGE = "ARRANGEMENT_ExtremeDynamicRange"
    PREMATURE_CLIMAX = "ARRANGEMENT_PrematureClimax"


class ProductionRuleBreak(Enum):
    """Production rules to intentionally break."""

    EXCESSIVE_MUD = "PRODUCTION_ExcessiveMud"
    PITCH_IMPERFECTION = "PRODUCTION_PitchImperfection"
    ROOM_NOISE = "PRODUCTION_RoomNoise"
    DISTORTION = "PRODUCTION_Distortion"
    MONO_COLLAPSE = "PRODUCTION_MonoCollapse"


class VulnerabilityScale(Enum):
    """Vulnerability level for emotional exposure."""

    LOW = "Low"  # Guarded, protective
    MEDIUM = "Medium"  # Honest but controlled
    HIGH = "High"  # Raw, exposed


class NarrativeArc(Enum):
    """Structural emotional arc."""

    CLIMB_TO_CLIMAX = "Climb-to-Climax"
    SLOW_REVEAL = "Slow Reveal"
    REPETITIVE_DESPAIR = "Repetitive Despair"
    STATIC_REFLECTION = "Static Reflection"
    SUDDEN_SHIFT = "Sudden Shift"
    DESCENT = "Descent"
    RISE_AND_FALL = "Rise and Fall"
    SPIRAL = "Spiral"


class CoreStakes(Enum):
    """What's at stake in the song."""

    PERSONAL = "Personal"  # Individual identity
    RELATIONAL = "Relational"  # Connections
    EXISTENTIAL = "Existential"  # Meaning/purpose
    SURVIVAL = "Survival"  # Life/safety
    CREATIVE = "Creative"  # Expression
    MORAL = "Moral"  # Right/wrong


class GrooveFeel(Enum):
    """Rhythmic feel."""

    STRAIGHT_DRIVING = "Straight/Driving"
    LAID_BACK = "Laid Back"
    SWUNG = "Swung"
    SYNCOPATED = "Syncopated"
    RUBATO_FREE = "Rubato/Free"
    MECHANICAL = "Mechanical"
    ORGANIC_BREATHING = "Organic/Breathing"
    PUSH_PULL = "Push-Pull"


# =================================================================
# RULE BREAKING DEFINITIONS
# =================================================================

RULE_BREAKING_EFFECTS = {
    # Harmony
    "HARMONY_AvoidTonicResolution": {
        "description": "Resolve to IV or VI instead of I",
        "effect": "Unresolved, yearning feeling",
        "use_when": "Song shouldn't feel 'finished' or 'answered'",
        "example_emotions": ["longing", "grief", "uncertainty"],
    },
    "HARMONY_ParallelMotion": {
        "description": "Use forbidden parallel 5ths/octaves",
        "effect": "Vintage, punk, or medieval sound",
        "use_when": "Defiance, raw power, or historical evocation",
        "example_emotions": ["defiance", "anger", "power"],
    },
    "HARMONY_ModalInterchange": {
        "description": "Borrow chord from unrelated key",
        "effect": "Unexpected color, emotional shift",
        "use_when": "Making emotions feel 'earned' or complex",
        "example_emotions": ["bittersweet", "nostalgia", "hope"],
    },
    "HARMONY_TritoneSubstitution": {
        "description": "Replace V7 with bII7",
        "effect": "Jazz sophistication, chromatic movement",
        "use_when": "Adding depth without cliché resolution",
        "example_emotions": ["sophistication", "complexity"],
    },
    "HARMONY_Polytonality": {
        "description": "Stack chords from different keys",
        "effect": "Tension, complexity, disorientation",
        "use_when": "Representing internal conflict or chaos",
        "example_emotions": ["confusion", "conflict", "chaos"],
    },
    "HARMONY_UnresolvedDissonance": {
        "description": "Leave 7ths, 9ths, tritones unresolved",
        "effect": "Lingering tension, incompleteness",
        "use_when": "Grief, longing, or open questions",
        "example_emotions": ["grief", "longing", "uncertainty"],
    },
    # Rhythm
    "RHYTHM_ConstantDisplacement": {
        "description": "Shift pattern one 16th note late/early",
        "effect": "Perpetually off-kilter, unsettling",
        "use_when": "Anxiety, instability, before a shift",
        "example_emotions": ["anxiety", "unease", "anticipation"],
    },
    "RHYTHM_TempoFluctuation": {
        "description": "Gradual ±5 BPM drift over phrase",
        "effect": "Organic breathing, tension/release",
        "use_when": "Human feel, emotional intensity",
        "example_emotions": ["intimacy", "vulnerability"],
    },
    "RHYTHM_MetricModulation": {
        "description": "Temporarily change implied time signature",
        "effect": "Disorientation, complexity",
        "use_when": "Representing mental state change",
        "example_emotions": ["confusion", "transformation"],
    },
    "RHYTHM_PolyrhythmicLayers": {
        "description": "Layer conflicting rhythmic patterns",
        "effect": "Complexity, tension, richness",
        "use_when": "Multiple emotions simultaneously",
        "example_emotions": ["complexity", "internal conflict"],
    },
    "RHYTHM_DroppedBeats": {
        "description": "Remove expected beats/hits",
        "effect": "Surprise, space, emphasis",
        "use_when": "Creating impact through absence",
        "example_emotions": ["shock", "emphasis", "breath"],
    },
    # Arrangement
    "ARRANGEMENT_UnbalancedDynamics": {
        "description": "Keep element too loud/quiet for standard",
        "effect": "Intentional imbalance, focus shift",
        "use_when": "Drawing attention or creating discomfort",
        "example_emotions": ["obsession", "imbalance"],
    },
    "ARRANGEMENT_StructuralMismatch": {
        "description": "Use unexpected structure for genre",
        "effect": "Subverted expectations, uniqueness",
        "use_when": "Story requires non-standard form",
        "example_emotions": ["defiance", "uniqueness"],
    },
    "ARRANGEMENT_BuriedVocals": {
        "description": "Place vocals below instruments",
        "effect": "Intimacy, dissociation, texture",
        "use_when": "Vulnerability, dream states",
        "example_emotions": ["dissociation", "intimacy", "dreams"],
    },
    "ARRANGEMENT_ExtremeDynamicRange": {
        "description": "Exceed normal dynamic limits",
        "effect": "Dramatic impact, contrast",
        "use_when": "Emotional crescendos, reveals",
        "example_emotions": ["catharsis", "revelation"],
    },
    "ARRANGEMENT_PrematureClimax": {
        "description": "Put peak earlier than expected",
        "effect": "Subversion, reflection time",
        "use_when": "Aftermath is the point",
        "example_emotions": ["aftermath", "reflection"],
    },
    # Production
    "PRODUCTION_ExcessiveMud": {
        "description": "Leave 200-400Hz buildup",
        "effect": "Weight, claustrophobia, heaviness",
        "use_when": "Trapped feelings, density",
        "example_emotions": ["trapped", "heavy", "suffocating"],
    },
    "PRODUCTION_PitchImperfection": {
        "description": "Leave natural pitch drift",
        "effect": "Emotional honesty, vulnerability",
        "use_when": "Raw emotional delivery",
        "example_emotions": ["vulnerability", "honesty", "rawness"],
    },
    "PRODUCTION_RoomNoise": {
        "description": "Keep ambient room sound",
        "effect": "Authenticity, intimacy, place",
        "use_when": "Lo-fi aesthetic, presence",
        "example_emotions": ["intimacy", "authenticity"],
    },
    "PRODUCTION_Distortion": {
        "description": "Allow clipping/saturation",
        "effect": "Aggression, urgency, damage",
        "use_when": "Anger, intensity, decay",
        "example_emotions": ["anger", "damage", "intensity"],
    },
    "PRODUCTION_MonoCollapse": {
        "description": "Intentionally narrow stereo field",
        "effect": "Claustrophobia, focus, intimacy",
        "use_when": "Internal monologue, pressure",
        "example_emotions": ["pressure", "focus", "isolation"],
    },
}


# =================================================================
# DATA CLASSES: Song Intent Structure
# =================================================================


@dataclass
class SongRoot:
    """
    Phase 0: The Core Wound/Desire

    Deep interrogation to find what the song NEEDS to express.
    """

    core_event: str = ""  # The inciting moment/realization
    core_resistance: str = ""  # What's holding you back
    core_longing: str = ""  # What you ultimately want to feel
    core_stakes: str = ""  # What's at risk
    core_transformation: str = ""  # How you want to feel when done


@dataclass
class SongIntent:
    """
    Phase 1: Emotional & Intent

    Validated by Phase 0, guides all technical decisions.
    """

    mood_primary: str = ""  # Primary emotion
    mood_secondary_tension: float = 0.5  # Tension level 0.0-1.0
    imagery_texture: str = ""  # Visual/tactile quality
    vulnerability_scale: str = "Medium"  # Low/Medium/High
    narrative_arc: str = ""  # Structural emotion


@dataclass
class TechnicalConstraints:
    """
    Phase 2: Technical Constraints

    Implementation of intent into concrete musical decisions.
    """

    technical_genre: str = ""
    technical_tempo_range: Tuple[int, int] = (80, 120)
    technical_key: str = ""
    technical_mode: str = ""
    technical_groove_feel: str = ""
    technical_rule_to_break: str = ""
    rule_breaking_justification: str = ""


@dataclass
class SystemDirective:
    """What DAiW should generate."""

    output_target: str = ""  # What to generate
    output_feedback_loop: str = ""  # Which modules to iterate


@dataclass
class CompleteSongIntent:
    """
    Complete song intent combining all phases.

    This is the full specification for a song that DAiW
    uses to generate meaningful, emotionally-aligned output.
    """

    # Phase 0
    song_root: SongRoot = field(default_factory=SongRoot)

    # Phase 1
    song_intent: SongIntent = field(default_factory=SongIntent)

    # Phase 2
    technical_constraints: TechnicalConstraints = field(default_factory=TechnicalConstraints)

    # System
    system_directive: SystemDirective = field(default_factory=SystemDirective)

    # Reasoning Engine Outputs
    midi_plan: Optional[Dict[str, Any]] = None
    image_prompt: Optional[str] = None
    image_style_constraints: Optional[str] = None
    audio_texture_prompt: Optional[str] = None
    explanation: Optional[str] = None
    rule_breaking_logic: Optional[str] = None
    generated_image_data: Optional[Dict[str, Any]] = None
    generated_audio_data: Optional[Dict[str, Any]] = None

    # Meta
    title: str = ""
    created: str = ""

    def __init__(
        self,
        core_event: str = "",
        core_resistance: str = "",
        core_longing: str = "",
        core_stakes: str = "",
        core_transformation: str = "",
        mood_primary: str = "",
        mood_secondary_tension: str = "",
        imagery_texture: str = "",
        vulnerability_scale: float = 0.0,
        narrative_arc: str = "",
        technical_genre: str = "",
        technical_tempo_range: tuple = (60, 140),
        technical_key: str = "",
        technical_mode: str = "",
        technical_groove_feel: str = "",
        technical_rule_to_break: str = "",
        rule_breaking_justification: str = "",
        output_target: str = "",
        output_feedback_loop: str = "",
        title: str = "",
        created: str = "",
        midi_plan: Optional[Dict[str, Any]] = None,
        image_prompt: Optional[str] = None,
        image_style_constraints: Optional[str] = None,
        audio_texture_prompt: Optional[str] = None,
        explanation: Optional[str] = None,
        rule_breaking_logic: Optional[str] = None,
        generated_image_data: Optional[Dict[str, Any]] = None,
        generated_audio_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.song_root = SongRoot(
            core_event=core_event,
            core_resistance=core_resistance,
            core_longing=core_longing,
            core_stakes=core_stakes,
            core_transformation=core_transformation,
        )
        try:
            tension_val = float(mood_secondary_tension)
        except Exception:
            tension_val = 0.5
        try:
            vuln_val = float(vulnerability_scale)
        except Exception:
            vuln_val = 0.5

        self.song_intent = SongIntent(
            mood_primary=mood_primary,
            mood_secondary_tension=tension_val,
            imagery_texture=imagery_texture,
            vulnerability_scale=vuln_val,
            narrative_arc=narrative_arc or "",
        )
        self.technical_constraints = TechnicalConstraints(
            technical_genre=technical_genre,
            technical_tempo_range=technical_tempo_range,
            technical_key=technical_key,
            technical_mode=technical_mode,
            technical_groove_feel=technical_groove_feel,
            technical_rule_to_break=technical_rule_to_break,
            rule_breaking_justification=rule_breaking_justification,
        )
        self.system_directive = SystemDirective(
            output_target=output_target,
            output_feedback_loop=output_feedback_loop,
        )
        self.title = title
        self.created = created
        self.midi_plan = midi_plan
        self.image_prompt = image_prompt
        self.image_style_constraints = image_style_constraints
        self.audio_texture_prompt = audio_texture_prompt
        self.explanation = explanation
        self.rule_breaking_logic = rule_breaking_logic
        self.generated_image_data = generated_image_data
        self.generated_audio_data = generated_audio_data

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "created": self.created,
            "song_root": {
                "core_event": self.song_root.core_event,
                "core_resistance": self.song_root.core_resistance,
                "core_longing": self.song_root.core_longing,
                "core_stakes": self.song_root.core_stakes,
                "core_transformation": self.song_root.core_transformation,
            },
            "song_intent": {
                "mood_primary": self.song_intent.mood_primary,
                "mood_secondary_tension": self.song_intent.mood_secondary_tension,
                "imagery_texture": self.song_intent.imagery_texture,
                "vulnerability_scale": self.song_intent.vulnerability_scale,
                "narrative_arc": self.song_intent.narrative_arc,
            },
            "technical_constraints": {
                "technical_genre": self.technical_constraints.technical_genre,
                "technical_tempo_range": list(self.technical_constraints.technical_tempo_range),
                "technical_key": self.technical_constraints.technical_key,
                "technical_mode": self.technical_constraints.technical_mode,
                "technical_groove_feel": self.technical_constraints.technical_groove_feel,
                "technical_rule_to_break": self.technical_constraints.technical_rule_to_break,
                "rule_breaking_justification": self.technical_constraints.rule_breaking_justification,
            },
            "system_directive": {
                "output_target": self.system_directive.output_target,
                "output_feedback_loop": self.system_directive.output_feedback_loop,
            },
            "midi_plan": self.midi_plan,
            "image_prompt": self.image_prompt,
            "image_style_constraints": self.image_style_constraints,
            "audio_texture_prompt": self.audio_texture_prompt,
            "explanation": self.explanation,
            "rule_breaking_logic": self.rule_breaking_logic,
            "generated_image_data": self.generated_image_data,
            "generated_audio_data": self.generated_audio_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CompleteSongIntent":
        """Create from dictionary."""
        intent = cls()
        intent.title = data.get("title", "")
        intent.created = data.get("created", "")

        if "song_root" in data:
            root = data["song_root"]
            intent.song_root = SongRoot(
                core_event=root.get("core_event", ""),
                core_resistance=root.get("core_resistance", ""),
                core_longing=root.get("core_longing", ""),
                core_stakes=root.get("core_stakes", ""),
                core_transformation=root.get("core_transformation", ""),
            )

        if "song_intent" in data:
            si = data["song_intent"]
            intent.song_intent = SongIntent(
                mood_primary=si.get("mood_primary", ""),
                mood_secondary_tension=si.get("mood_secondary_tension", 0.5),
                imagery_texture=si.get("imagery_texture", ""),
                vulnerability_scale=si.get("vulnerability_scale", "Medium"),
                narrative_arc=si.get("narrative_arc", ""),
            )

        if "technical_constraints" in data:
            tc = data["technical_constraints"]
            tempo = tc.get("technical_tempo_range", [80, 120])
            intent.technical_constraints = TechnicalConstraints(
                technical_genre=tc.get("technical_genre", ""),
                technical_tempo_range=tuple(tempo) if isinstance(tempo, list) else tempo,
                technical_key=tc.get("technical_key", ""),
                technical_mode=tc.get("technical_mode", ""),
                technical_groove_feel=tc.get("technical_groove_feel", ""),
                technical_rule_to_break=tc.get("technical_rule_to_break", ""),
                rule_breaking_justification=tc.get("rule_breaking_justification", ""),
            )

        if "system_directive" in data:
            sd = data["system_directive"]
            intent.system_directive = SystemDirective(
                output_target=sd.get("output_target", ""),
                output_feedback_loop=sd.get("output_feedback_loop", ""),
            )

        intent.midi_plan = data.get("midi_plan", None)
        intent.image_prompt = data.get("image_prompt", None)
        intent.image_style_constraints = data.get("image_style_constraints", None)
        intent.audio_texture_prompt = data.get("audio_texture_prompt", None)
        intent.explanation = data.get("explanation", None)
        intent.rule_breaking_logic = data.get("rule_breaking_logic", None)
        intent.generated_image_data = data.get("generated_image_data", None)
        intent.generated_audio_data = data.get("generated_audio_data", None)

        return intent

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CompleteSongIntent":
        """Load from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# =================================================================
# HELPER FUNCTIONS
# =================================================================


def suggest_rule_break(emotion: str) -> List[Dict]:
    """
    Suggest appropriate rules to break based on target emotion.

    Args:
        emotion: Target emotion (grief, defiance, etc.)

    Returns:
        List of rule-breaking suggestions with justifications
    """
    emotion_lower = emotion.lower()
    suggestions = []

    for rule_key, rule_data in RULE_BREAKING_EFFECTS.items():
        if any(e in emotion_lower for e in rule_data.get("example_emotions", [])):
            suggestions.append(
                {
                    "rule": rule_key,
                    "description": rule_data["description"],
                    "effect": rule_data["effect"],
                    "use_when": rule_data["use_when"],
                }
            )

    return suggestions


def get_rule_breaking_info(rule_key: str) -> Optional[Dict]:
    """Get detailed info about a rule-breaking option."""
    return RULE_BREAKING_EFFECTS.get(rule_key)


# Affect mapping for musical parameters based on mood
AFFECT_MAPPINGS: Dict[str, Dict] = {
    "Grief": {
        "modes": ["aeolian", "phrygian", "dorian"],
        "tempo_range": (50, 80),
        "key_preference": "minor",
    },
    "Joy": {
        "modes": ["ionian", "lydian", "mixolydian"],
        "tempo_range": (100, 140),
        "key_preference": "major",
    },
    "Nervousness": {
        "modes": ["locrian", "phrygian"],
        "tempo_range": (90, 130),
        "key_preference": "minor",
    },
    "Defiance": {
        "modes": ["mixolydian", "dorian"],
        "tempo_range": (110, 150),
        "key_preference": "major",
    },
    "Liberation": {
        "modes": ["lydian", "ionian"],
        "tempo_range": (100, 130),
        "key_preference": "major",
    },
    "Longing": {"modes": ["dorian", "aeolian"], "tempo_range": (60, 90), "key_preference": "minor"},
    "Rage": {
        "modes": ["phrygian", "locrian"],
        "tempo_range": (130, 180),
        "key_preference": "minor",
    },
    "Acceptance": {
        "modes": ["ionian", "mixolydian"],
        "tempo_range": (70, 100),
        "key_preference": "major",
    },
    "Nostalgia": {
        "modes": ["dorian", "mixolydian"],
        "tempo_range": (70, 100),
        "key_preference": "major",
    },
    "Dissociation": {
        "modes": ["locrian", "phrygian"],
        "tempo_range": (60, 90),
        "key_preference": "minor",
    },
    "Triumphant Hope": {
        "modes": ["lydian", "ionian"],
        "tempo_range": (100, 130),
        "key_preference": "major",
    },
    "Bittersweet": {
        "modes": ["dorian", "mixolydian"],
        "tempo_range": (70, 100),
        "key_preference": "minor",
    },
    "Melancholy": {
        "modes": ["aeolian", "dorian"],
        "tempo_range": (50, 80),
        "key_preference": "minor",
    },
    "Euphoria": {
        "modes": ["lydian", "ionian"],
        "tempo_range": (120, 150),
        "key_preference": "major",
    },
    "Desperation": {
        "modes": ["phrygian", "aeolian"],
        "tempo_range": (90, 130),
        "key_preference": "minor",
    },
    "Serenity": {"modes": ["ionian", "lydian"], "tempo_range": (60, 90), "key_preference": "major"},
    "Confusion": {
        "modes": ["locrian", "phrygian"],
        "tempo_range": (80, 120),
        "key_preference": "minor",
    },
    "Determination": {
        "modes": ["dorian", "mixolydian"],
        "tempo_range": (100, 130),
        "key_preference": "major",
    },
}


def get_affect_mapping(mood_primary: str) -> Optional[Dict]:
    """
    Get musical parameters (modes, tempo, key) based on mood.

    Args:
        mood_primary: The primary mood from VALID_MOOD_PRIMARY_OPTIONS

    Returns:
        Dict with 'modes', 'tempo_range', 'key_preference' or None if not found
    """
    return AFFECT_MAPPINGS.get(mood_primary)


def validate_intent(intent: CompleteSongIntent) -> List[str]:
    """
    Validate a song intent for completeness and consistency.

    Returns list of issues found (empty = valid).
    """
    issues: List[str] = []

    # Phase 0 checks
    if not intent.song_root.core_event:
        issues.append("Phase 0: Missing core_event - what happened?")
    if not intent.song_root.core_longing:
        issues.append("Phase 0: Missing core_longing - what do you want to feel?")

    # Validate core_stakes enum
    if (
        intent.song_root.core_stakes
        and intent.song_root.core_stakes not in VALID_CORE_STAKES_OPTIONS
    ):
        issues.append(
            f"Phase 0: Invalid core_stakes '{intent.song_root.core_stakes}'. Must be one of: {', '.join(VALID_CORE_STAKES_OPTIONS)}"
        )

    # Phase 1 checks
    if not intent.song_intent.mood_primary:
        issues.append("Phase 1: Missing mood_primary - what's the main emotion?")
    elif intent.song_intent.mood_primary not in VALID_MOOD_PRIMARY_OPTIONS:
        issues.append(
            f"Phase 1: Invalid mood_primary '{intent.song_intent.mood_primary}'. Must be one of: {', '.join(VALID_MOOD_PRIMARY_OPTIONS)}"
        )

    if (
        intent.song_intent.mood_secondary_tension < 0
        or intent.song_intent.mood_secondary_tension > 1
    ):
        issues.append("Phase 1: mood_secondary_tension should be 0.0-1.0")

    # Validate imagery_texture enum
    if (
        intent.song_intent.imagery_texture
        and intent.song_intent.imagery_texture not in VALID_IMAGERY_TEXTURE_OPTIONS
    ):
        issues.append(
            f"Phase 1: Invalid imagery_texture '{intent.song_intent.imagery_texture}'. Must be one of: {', '.join(VALID_IMAGERY_TEXTURE_OPTIONS)}"
        )

    # Validate vulnerability_scale enum
    if isinstance(intent.song_intent.vulnerability_scale, str):
        if intent.song_intent.vulnerability_scale not in VALID_VULNERABILITY_SCALE_OPTIONS:
            issues.append(
                f"Phase 1: Invalid vulnerability_scale '{intent.song_intent.vulnerability_scale}'. Must be one of: {', '.join(VALID_VULNERABILITY_SCALE_OPTIONS)}"
            )

    # Validate narrative_arc enum
    if (
        intent.song_intent.narrative_arc
        and intent.song_intent.narrative_arc not in VALID_NARRATIVE_ARC_OPTIONS
    ):
        issues.append(
            f"Phase 1: Invalid narrative_arc '{intent.song_intent.narrative_arc}'. Must be one of: {', '.join(VALID_NARRATIVE_ARC_OPTIONS)}"
        )

    # Phase 2 checks
    if intent.technical_constraints.technical_rule_to_break:
        if not intent.technical_constraints.rule_breaking_justification:
            issues.append(
                "Phase 2: Rule to break specified without justification - WHY break this rule?"
            )

    # Validate genre enum
    if (
        intent.technical_constraints.technical_genre
        and intent.technical_constraints.technical_genre not in VALID_GENRE_OPTIONS
    ):
        issues.append(
            f"Phase 2: Invalid technical_genre '{intent.technical_constraints.technical_genre}'. Must be one of: {', '.join(VALID_GENRE_OPTIONS)}"
        )

    # Validate groove_feel enum
    if (
        intent.technical_constraints.technical_groove_feel
        and intent.technical_constraints.technical_groove_feel not in VALID_GROOVE_FEEL_OPTIONS
    ):
        issues.append(
            f"Phase 2: Invalid technical_groove_feel '{intent.technical_constraints.technical_groove_feel}'. Must be one of: {', '.join(VALID_GROOVE_FEEL_OPTIONS)}"
        )

    # Consistency checks
    vuln_scale = intent.song_intent.vulnerability_scale
    if isinstance(vuln_scale, str) and vuln_scale == "High":
        if intent.song_intent.mood_secondary_tension < 0.3:
            issues.append(
                "Consistency: High vulnerability usually implies some tension (tension is very low)"
            )

    return issues


def list_all_rules() -> Dict[str, List[str]]:
    """Get all available rule-breaking options by category."""
    return {
        "Harmony": [e.value for e in HarmonyRuleBreak],
        "Rhythm": [e.value for e in RhythmRuleBreak],
        "Arrangement": [e.value for e in ArrangementRuleBreak],
        "Production": [e.value for e in ProductionRuleBreak],
    }
