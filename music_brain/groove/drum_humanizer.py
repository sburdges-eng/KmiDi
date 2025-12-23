"""
Drum humanization scaffold using Production_Workflows guide rules.

Intended to sit between drum analysis and the existing groove engine:
- Consume DrumTechniqueProfile (from drum_analysis)
- Apply guide-derived presets (Drum Programming Guide, Humanization Cheat Sheet)
- Output GrooveSettings or a future humanization plan

TODO:
- Ingest markdown rules (velocity, timing, ghost notes, swing) as data.
- Add section-aware presets (verse/chorus/bridge) from Dynamics guide.
- Connect to groove_engine.humanize_drums for full application.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from music_brain.groove.drum_analysis import DrumTechniqueProfile
from music_brain.groove.groove_engine import GrooveSettings


@dataclass
class GuideRuleSet:
    """Simplified representation of guide-derived parameters."""

    swing: float = 0.0
    timing_shift_ms: float = 0.0
    ghost_rate: float = 0.0
    velocity_variation: float = 0.1
    notes: List[str] = field(default_factory=list)


class DrumHumanizer:
    """Placeholder humanizer that will apply Production_Workflows rules."""

    def __init__(self, default_style: str = "standard") -> None:
        self.default_style = default_style
        self.guide_rules: Dict[str, GuideRuleSet] = self._build_default_rules()

    def _build_default_rules(self) -> Dict[str, GuideRuleSet]:
        """Seed a few rule presets until markdown ingestion is wired up."""
        return {
            "standard": GuideRuleSet(
                swing=0.0,
                timing_shift_ms=5.0,
                ghost_rate=0.05,
                velocity_variation=0.12,
                notes=["Baseline humanization; replace with guide-driven data."],
            ),
            "jazzy": GuideRuleSet(
                swing=0.58,
                timing_shift_ms=12.0,
                ghost_rate=0.12,
                velocity_variation=0.18,
                notes=["Pulled-back snare, heavy ghost notes, ride accenting."],
            ),
            "heavy": GuideRuleSet(
                swing=0.0,
                timing_shift_ms=3.0,
                ghost_rate=0.02,
                velocity_variation=0.1,
                notes=["Tight kicks/snares; velocity accents drive impact."],
            ),
            "laid_back": GuideRuleSet(
                swing=0.54,
                timing_shift_ms=18.0,
                ghost_rate=0.1,
                velocity_variation=0.15,
                notes=["Behind-the-beat feel; suitable for R&B/lo-fi pockets."],
            ),
        }

    def create_preset_from_guide(self, style: Optional[str] = None) -> GrooveSettings:
        """
        Create a GrooveSettings object seeded by guide presets.
        """
        style_key = style or self.default_style
        rules = self.guide_rules.get(style_key, self.guide_rules["standard"])
        settings = GrooveSettings()

        # Map guide cues onto GrooveSettings heuristically.
        settings.complexity = 0.55 + min(0.2, rules.ghost_rate)
        settings.vulnerability = 0.5 + (rules.velocity_variation * 0.5)
        settings.ghost_note_probability = rules.ghost_rate

        # TODO: map swing/timing_shift_ms into groove_engine jitter parameters.
        return settings

    def apply_guide_rules(
        self,
        midi: Any,
        technique_profile: Optional[DrumTechniqueProfile] = None,
        style: Optional[str] = None,
    ) -> Any:
        """
        Stub for applying guide-informed humanization to a MIDI object.

        Currently returns the input unchanged but attaches TODOs in notes.
        """
        _ = technique_profile  # placeholder until analysis-to-rules is wired
        _ = style
        # TODO: call groove_engine.humanize_drums with derived GrooveSettings.
        return midi

    def to_plan(
        self,
        technique_profile: Optional[DrumTechniqueProfile],
        style: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return a JSON-serializable plan that other layers can consume.
        """
        rules = self.guide_rules.get(style or self.default_style, self.guide_rules["standard"])
        return {
            "style": style or self.default_style,
            "swing": rules.swing,
            "timing_shift_ms": rules.timing_shift_ms,
            "ghost_rate": rules.ghost_rate,
            "velocity_variation": rules.velocity_variation,
            "techniques": technique_profile.__dict__ if technique_profile else {},
            "notes": rules.notes
            + ["TODO: merge Drum Programming Guide + Humanization Cheat Sheet."],
        }

