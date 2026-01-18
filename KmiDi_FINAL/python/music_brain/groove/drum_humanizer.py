"""
Drum humanization scaffold using Production_Workflows guide rules.

Intended to sit between drum analysis and the existing groove engine:
- Consume DrumTechniqueProfile (from drum_analysis)
- Apply guide-derived presets (Drum Programming Guide, Humanization
  Cheat Sheet)
- Output GrooveSettings or a future humanization plan

- Ingest markdown rules (velocity, timing, ghost notes, swing) as data.
- Add section-aware presets (verse/chorus/bridge) from Dynamics guide.
- Connect to groove_engine.humanize_drums for full application.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from music_brain.groove.drum_analysis import AnalysisConfig, DrumAnalyzer, DrumTechniqueProfile
from music_brain.groove.groove_engine import GrooveSettings, humanize_drums
from music_brain.groove.guide_parser import DrumGuideParser
from music_brain.groove.fan_feedback import FanProfile


@dataclass
class HumanizerConfig:
    """Config for analysis defaults."""

    ppq: int = 480
    bpm: float = 120.0


@dataclass
class GuideRuleSet:
    """Simplified representation of guide-derived parameters."""

    swing: float = 0.0
    timing_shift_ms: float = 0.0
    ghost_rate: float = 0.0
    velocity_variation: float = 0.1
    notes: List[str] = field(default_factory=list)
    # Parsed instrument-specific rules
    hihat_velocity_range: Optional[Tuple[int, int]] = None
    hihat_timing_range: Optional[Tuple[int, int]] = None
    snare_ghost_velocity: Optional[Tuple[int, int]] = None
    snare_main_velocity: Optional[Tuple[int, int]] = None
    kick_velocity_range: Optional[Tuple[int, int]] = None


class DrumHumanizer:
    """Placeholder humanizer that will apply Production_Workflows rules."""

    def __init__(
        self,
        default_style: str = "standard",
        config: Optional[HumanizerConfig] = None,
        analyzer: Optional[DrumAnalyzer] = None,
        config_path: Optional[Path] = None,
        guide_path: Optional[Path] = None,
    ) -> None:
        self.default_style = default_style

        # Locate guide if not provided
        if not guide_path:
            # Try to find vault relative to this file
            # music_brain/groove/drum_humanizer.py -> .../KmiDi/
            root = Path(__file__).resolve().parent.parent.parent
            guide_path = root / "vault" / "Production_Guides" / "Drum Programming Guide.md"

        self.guide_parser = DrumGuideParser(guide_path)
        self.guide_rules: Dict[str, GuideRuleSet] = self._build_default_rules()

        loaded_cfg, loaded_analysis, style_override = self._load_config(
            config_path)
        self.config = config or loaded_cfg
        if style_override:
            self.default_style = style_override
        # Analyzer is lazily cloned per-call when bpm/ppq override is provided.
        self.analyzer = analyzer or DrumAnalyzer(
            ppq=self.config.ppq,
            bpm=self.config.bpm,
            config=loaded_analysis,
        )

    def _build_default_rules(self) -> Dict[str, GuideRuleSet]:
        """Build ruleset by parsing the guide and adding legacy fallbacks."""
        parsed_data = self.guide_parser.parse()

        # Extract global instrument rules
        hihat_vel = parsed_data["hihat"].get("velocity_variation")
        hihat_time = parsed_data["hihat"].get("timing_variation_ms")
        snare_ghost = parsed_data["snare"].get("ghost_velocity")
        snare_main = parsed_data["snare"].get("main_velocity")
        kick_vel = parsed_data["kick"].get("velocity_range")

        rules: Dict[str, GuideRuleSet] = {}

        # 1. Create rules from parsed genres
        for genre_name, genre_data in parsed_data.get("genres", {}).items():
            # Normalize genre name to snake_case for keys
            key = genre_name.lower().replace(" ", "_").replace("-", "_")

            rules[key] = GuideRuleSet(
                swing=genre_data.get("swing", 0.0),
                timing_shift_ms=genre_data.get("timing_shift", 0.0),
                ghost_rate=0.1,  # Default, could be refined by parsing notes
                velocity_variation=0.1,
                notes=genre_data.get("notes", []),
                hihat_velocity_range=hihat_vel,
                hihat_timing_range=hihat_time,
                snare_ghost_velocity=snare_ghost,
                snare_main_velocity=snare_main,
                kick_velocity_range=kick_vel
            )

        # 2. Add legacy/fallback styles if they weren't parsed
        legacy_defaults = {
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
            "technical": GuideRuleSet(
                swing=0.02,
                timing_shift_ms=6.0,
                ghost_rate=0.15,
                velocity_variation=0.2,
                notes=["Stick control focus; buzz/drag textures emphasized."],
            ),
            "laid_back": GuideRuleSet(
                swing=0.54,
                timing_shift_ms=18.0,
                ghost_rate=0.1,
                velocity_variation=0.15,
                notes=["Behind-the-beat feel; suitable for R&B/lo-fi pockets."],
            ),
        }

        for key, rule in legacy_defaults.items():
            if key not in rules:
                # Inject global instrument rules into legacy presets too
                rule.hihat_velocity_range = hihat_vel
                rule.hihat_timing_range = hihat_time
                rule.snare_ghost_velocity = snare_ghost
                rule.snare_main_velocity = snare_main
                rule.kick_velocity_range = kick_vel
                rules[key] = rule

        return rules

    def create_preset_from_guide(
        self,
        style: Optional[str] = None,
        technique_profile: Optional[DrumTechniqueProfile] = None,
        fan_profile: Optional[FanProfile] = None,
    ) -> GrooveSettings:
        """
        Create a GrooveSettings object seeded by guide presets.
        """
        style_key = style or self._style_from_profile(technique_profile)
        rules = self.guide_rules.get(style_key, self.guide_rules["standard"])
        settings = GrooveSettings()

        # Apply Fan Profile Modifiers
        fp = fan_profile or FanProfile()  # Default neutral profile

        # Map guide cues onto GrooveSettings heuristically.
        # Base values
        base_ghost = rules.ghost_rate
        base_vel_var = rules.velocity_variation
        base_swing = rules.swing

        # Modulated values
        settings.complexity = 0.55 + min(0.2, base_ghost)

        # Velocity variation is affected by slop multiplier (looser = more variation)
        settings.vulnerability = (
            0.5 + (base_vel_var * 0.5)) * fp.timing_slop_multiplier
        settings.ghost_note_probability = base_ghost

        # Swing modulation
        # GrooveSettings doesn't have explicit swing field in this stub,
        # but we would apply it here.
        # settings.swing = fp.apply_swing(base_swing)

        return settings

    def apply_guide_rules(
        self,
        midi: Any,
        technique_profile: Optional[DrumTechniqueProfile] = None,
        style: Optional[str] = None,
        notes: Optional[Sequence[Any]] = None,
        bpm: Optional[float] = None,
        ppq: Optional[int] = None,
        seed: Optional[int] = None,
        fan_profile: Optional[FanProfile] = None,
    ) -> Any:
        """
        Stub for applying guide-informed humanization to a MIDI object.

        Currently returns the input unchanged but attaches TODOs in notes.
        """
        profile = technique_profile or self._analyze_notes_if_possible(
            notes=notes,
            bpm=bpm,
            ppq=ppq,
        )
        inferred_style = style or self._style_from_profile(profile)
        settings = self.create_preset_from_guide(
            style=inferred_style,
            technique_profile=profile,
            fan_profile=fan_profile,
        )
        derived_ppq = ppq or self.config.ppq

        try:
            events = list(midi)
        except TypeError:
            return midi

        return humanize_drums(
            events=events,
            complexity=settings.complexity,
            vulnerability=settings.vulnerability,
            ppq=derived_ppq,
            settings=settings,
            seed=seed,
        )

    def to_plan(
        self,
        technique_profile: Optional[DrumTechniqueProfile],
        style: Optional[str] = None,
        notes: Optional[Sequence[Any]] = None,
        bpm: Optional[float] = None,
        ppq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Return a JSON-serializable plan that other layers can consume.
        """
        profile = technique_profile or self._analyze_notes_if_possible(
            notes=notes,
            bpm=bpm,
            ppq=ppq,
        )
        style_key = style or self._style_from_profile(profile)
        rules = self.guide_rules.get(
            style_key,
            self.guide_rules["standard"],
        )
        return {
            "style": style_key,
            "swing": rules.swing,
            "timing_shift_ms": rules.timing_shift_ms,
            "ghost_rate": rules.ghost_rate,
            "velocity_variation": rules.velocity_variation,
            "techniques": (
                profile.__dict__ if profile else {}
            ),
            "notes": rules.notes
            + [
                "TODO: merge Drum Programming Guide + Humanization "
                "Cheat Sheet."
            ],
        }

    def analyze_notes(
        self,
        notes: Sequence[Any],
        bpm: Optional[float] = None,
        ppq: Optional[int] = None,
    ) -> DrumTechniqueProfile:
        """Analyze a sequence of note-like objects using DrumAnalyzer."""
        analyzer = (
            self.analyzer
            if bpm is None and ppq is None
            else DrumAnalyzer(ppq=ppq or self.config.ppq, bpm=bpm or self.config.bpm)
        )
        return analyzer.analyze(list(notes), bpm=bpm or analyzer.bpm)

    def _load_config(self, config_path: Optional[Path]) -> Tuple[HumanizerConfig, Optional[AnalysisConfig], Optional[str]]:
        """Load HumanizerConfig and AnalysisConfig overrides from JSON if provided."""
        if not config_path:
            return HumanizerConfig(), None, None
        try:
            data = json.loads(Path(config_path).read_text())
            analysis_cfg = AnalysisConfig.from_dict(
                data["analysis"]) if "analysis" in data else None
            config = HumanizerConfig(
                **{k: v for k, v in data.items() if k in {"ppq", "bpm"}})
            style = data.get("default_style")
            return config, analysis_cfg, style
        except FileNotFoundError:
            return HumanizerConfig(), None, None
        except Exception:
            # On malformed config, fall back to defaults.
            return HumanizerConfig(), None, None

    def _style_from_profile(
        self, profile: Optional[DrumTechniqueProfile]
    ) -> str:
        """Pick a guide preset based on detected technique."""
        if profile and profile.snare.primary_technique in self.guide_rules:
            return profile.snare.primary_technique
        return self.default_style

    def _analyze_notes_if_possible(
        self,
        notes: Optional[Sequence[Any]],
        bpm: Optional[float],
        ppq: Optional[int],
    ) -> Optional[DrumTechniqueProfile]:
        if not notes:
            return None
        return self.analyze_notes(notes=notes, bpm=bpm, ppq=ppq)
