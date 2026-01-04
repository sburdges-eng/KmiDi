"""
Humanization layer that applies the Drum Programming Guide to MIDI events.
"""

from typing import Any, Dict, List, Optional

from music_brain.groove.groove_engine import GrooveSettings, humanize_drums
from music_brain.groove.drum_analysis import AnalysisConfig, DrumAnalyzer, DrumTechniqueProfile


class DrumHumanizer:
    """
    Apply guide-informed presets to drum MIDI events.

    This class is intentionally lightweight: it builds `GrooveSettings` from
    guide-inspired style presets and optionally refines them using a
    `DrumTechniqueProfile` from `DrumAnalyzer`.
    """

    def __init__(
        self,
        analyzer: Optional[DrumAnalyzer] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.analyzer = analyzer or DrumAnalyzer()
        self._style_presets: Dict[str, Dict[str, Any]] = {
            "standard": {
                "complexity": 0.5,
                "vulnerability": 0.5,
                "enable_ghost_notes": True,
            },
            "hip-hop": {
                "complexity": 0.35,
                "vulnerability": 0.65,
                "ghost_note_probability": 0.16,
                "hihat_timing_mult": 1.25,
            },
            "rock": {
                "complexity": 0.45,
                "vulnerability": 0.45,
                "kick_timing_mult": 0.4,
                "snare_timing_mult": 0.6,
            },
            "jazzy": {
                "complexity": 0.55,
                "vulnerability": 0.6,
                "ghost_note_probability": 0.22,
            },
            "edm": {
                "complexity": 0.25,
                "vulnerability": 0.4,
                "enable_ghost_notes": False,
                "hihat_timing_mult": 1.0,
            },
            "lofi": {
                "complexity": 0.3,
                "vulnerability": 0.7,
                "ghost_note_probability": 0.28,
                "velocity_range_override": (40, 110),
            },
        }
        if config_path:
            self._load_presets(config_path)

    def create_preset_from_guide(self, style: str) -> GrooveSettings:
        """Create a `GrooveSettings` instance from a style keyword."""
        key = style.lower() if style else "standard"
        preset_kwargs = self._style_presets.get(key, self._style_presets["standard"])
        return GrooveSettings(**preset_kwargs)

    def apply_guide_rules(
        self,
        events: List[Dict[str, Any]],
        style: str = "standard",
        technique_profile: Optional[DrumTechniqueProfile] = None,
        ppq: int = 480,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply humanization to event dictionaries using guide-driven presets.

        Args:
            events: List of MIDI-like event dictionaries expected by `humanize_drums`.
            style: Guide style label (e.g., "hip-hop", "rock", "jazzy").
            technique_profile: Optional analysis profile to fine-tune settings.
            ppq: Pulses per quarter note.
            seed: Optional random seed for reproducibility.
        """
        settings = self.create_preset_from_guide(style)
        complexity = settings.complexity
        vulnerability = settings.vulnerability

        if technique_profile:
            complexity = self._clamp(
                complexity + (technique_profile.fill_density - 0.5) * 0.2
            )
            vulnerability = self._clamp(
                vulnerability + (technique_profile.tightness - 0.5) * -0.2
            )
            if (
                technique_profile.snare.has_buzz_rolls
                or technique_profile.ghost_note_density > 0.2
            ):
                settings.enable_ghost_notes = True
                settings.ghost_note_probability = max(
                    settings.ghost_note_probability, 0.12
                )

        event_copy = [dict(event) for event in events]
        return humanize_drums(
            events=event_copy,
            complexity=complexity,
            vulnerability=vulnerability,
            ppq=ppq,
            settings=settings,
            seed=seed,
        )

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _load_presets(self, path: str) -> None:
        """Optional: override style presets from JSON."""
        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "styles" in data and isinstance(data["styles"], dict):
                    self._style_presets.update(data["styles"])
                if "analysis" in data:
                    try:
                        cfg = AnalysisConfig.from_dict(data["analysis"])
                        self.analyzer = DrumAnalyzer(config=cfg)
                    except Exception:
                        pass
        except FileNotFoundError:
            return
        except Exception:
            return
