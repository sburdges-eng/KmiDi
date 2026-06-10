"""
Song Intent Schema - Structured deep interrogation for songwriting.

Implements the three-phase interrogation model:
- Phase 0: Core Wound/Desire (deep interrogation)
- Phase 1: Emotional & Intent (validation)
- Phase 2: Technical Constraints (implementation)

Plus comprehensive rule-breaking enums for intentional creative choices.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json


def _coerce_text(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


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
        **kwargs,
    ):
        self.song_root = SongRoot(
            core_event=_coerce_text(core_event),
            core_resistance=_coerce_text(core_resistance),
            core_longing=_coerce_text(core_longing),
            core_stakes=_coerce_text(core_stakes),
            core_transformation=_coerce_text(core_transformation),
        )
        # Validate and clamp mood_secondary_tension to [0.0, 1.0]
        try:
            tension_val = float(mood_secondary_tension)
            tension_val = max(0.0, min(1.0, tension_val))  # Clamp to valid range
        except (ValueError, TypeError):
            # ValueError: invalid literal for float()
            # TypeError: float() argument must be a string or a number
            tension_val = 0.5

        # Validate vulnerability_scale - normalize to string enum
        if isinstance(vulnerability_scale, (int, float)):
            # Convert float to string enum
            vuln_float = float(vulnerability_scale)
            if vuln_float < 0.33:
                vuln_str = "Low"
            elif vuln_float < 0.67:
                vuln_str = "Medium"
            else:
                vuln_str = "High"
        else:
            # Validate string enum
            vuln_str = str(vulnerability_scale).strip().title()
            if vuln_str not in {"Low", "Medium", "High"}:
                vuln_str = "Medium"

        self.song_intent = SongIntent(
            mood_primary=_coerce_text(mood_primary),
            mood_secondary_tension=tension_val,
            imagery_texture=_coerce_text(imagery_texture),
            vulnerability_scale=vuln_str,
            narrative_arc=_coerce_text(narrative_arc),
        )
        self.technical_constraints = TechnicalConstraints(
            technical_genre=_coerce_text(technical_genre),
            technical_tempo_range=technical_tempo_range,
            technical_key=_coerce_text(technical_key),
            technical_mode=_coerce_text(technical_mode),
            technical_groove_feel=_coerce_text(technical_groove_feel),
            technical_rule_to_break=_coerce_text(technical_rule_to_break),
            rule_breaking_justification=_coerce_text(rule_breaking_justification),
        )
        self.system_directive = SystemDirective(
            output_target=_coerce_text(output_target),
            output_feedback_loop=_coerce_text(output_feedback_loop),
        )
        self.title = _coerce_text(title)
        self.created = _coerce_text(created)

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
                "rule_breaking_justification": self.technical_constraints.rule_breaking_justification,  # noqa: E501
            },
            "system_directive": {
                "output_target": self.system_directive.output_target,
                "output_feedback_loop": self.system_directive.output_feedback_loop,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CompleteSongIntent":
        """Create from dictionary."""
        intent = cls()
        intent.title = _coerce_text(data.get("title", ""))
        intent.created = _coerce_text(data.get("created", ""))

        if "song_root" in data:
            root = data.get("song_root", {})
            intent.song_root = SongRoot(
                core_event=_coerce_text(root.get("core_event", "")),
                core_resistance=_coerce_text(root.get("core_resistance", "")),
                core_longing=_coerce_text(root.get("core_longing", "")),
                core_stakes=_coerce_text(root.get("core_stakes", "")),
                core_transformation=_coerce_text(root.get("core_transformation", "")),
            )

        if "song_intent" in data:
            si = data.get("song_intent", {})
            # Validate mood_secondary_tension is in range [0.0, 1.0]
            tension = si.get("mood_secondary_tension", 0.5)
            try:
                tension_val = float(tension)
                tension_val = max(0.0, min(1.0, tension_val))  # Clamp to valid range
            except (ValueError, TypeError):
                tension_val = 0.5

            # Validate vulnerability_scale is valid enum value
            vuln = si.get("vulnerability_scale", "Medium")
            _valid_vuln = {"Low", "Medium", "High", "LOW", "MEDIUM", "HIGH"}  # noqa: F841
            if isinstance(vuln, str):
                # Normalize to title case
                vuln_normalized = vuln.strip().title()
                if vuln_normalized not in {"Low", "Medium", "High"}:
                    vuln_normalized = "Medium"  # Default if invalid
                vuln = vuln_normalized
            else:
                vuln = "Medium"

            intent.song_intent = SongIntent(
                mood_primary=_coerce_text(si.get("mood_primary", "")),
                mood_secondary_tension=tension_val,
                imagery_texture=_coerce_text(si.get("imagery_texture", "")),
                vulnerability_scale=vuln,
                narrative_arc=_coerce_text(si.get("narrative_arc", "")),
            )

        if "technical_constraints" in data:
            tc = data.get("technical_constraints", {})
            tempo = tc.get("technical_tempo_range", [80, 120])
            intent.technical_constraints = TechnicalConstraints(
                technical_genre=_coerce_text(tc.get("technical_genre", "")),
                technical_tempo_range=tuple(tempo) if isinstance(tempo, list) else tempo,
                technical_key=_coerce_text(tc.get("technical_key", "")),
                technical_mode=_coerce_text(tc.get("technical_mode", "")),
                technical_groove_feel=_coerce_text(tc.get("technical_groove_feel", "")),
                technical_rule_to_break=_coerce_text(tc.get("technical_rule_to_break", "")),
                rule_breaking_justification=_coerce_text(tc.get("rule_breaking_justification", "")),
            )

        if "system_directive" in data:
            sd = data.get("system_directive", {})
            intent.system_directive = SystemDirective(
                output_target=_coerce_text(sd.get("output_target", "")),
                output_feedback_loop=_coerce_text(sd.get("output_feedback_loop", "")),
            )

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

    # Phase 1 checks
    if not intent.song_intent.mood_primary:
        issues.append("Phase 1: Missing mood_primary - what's the main emotion?")
    if (
        intent.song_intent.mood_secondary_tension < 0
        or intent.song_intent.mood_secondary_tension > 1
    ):  # noqa: E501

        issues.append("Phase 1: mood_secondary_tension should be 0.0-1.0")

    # Phase 2 checks
    if intent.technical_constraints.technical_rule_to_break:
        if not intent.technical_constraints.rule_breaking_justification:
            issues.append(
                "Phase 2: Rule to break specified without justification - WHY break this rule?"
            )

    # Consistency checks
    if intent.song_intent.vulnerability_scale == "High":
        if intent.song_intent.mood_secondary_tension < 0.3:
            issues.append(
                "Consistency: High vulnerability usually implies some tension (tension is very low)"
            )  # noqa: E501

    return issues


def list_all_rules() -> Dict[str, List[str]]:
    """Get all available rule-breaking options by category."""
    return {
        "Harmony": [e.value for e in HarmonyRuleBreak],
        "Rhythm": [e.value for e in RhythmRuleBreak],
        "Arrangement": [e.value for e in ArrangementRuleBreak],
        "Production": [e.value for e in ProductionRuleBreak],
    }
