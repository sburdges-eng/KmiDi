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
        # Guide-driven style presets based on Drum Programming Guide
        # Key parameters:
        # - complexity: 0-1, affects fill density and variation
        # - vulnerability: 0-1, affects timing looseness and human feel
        # - enable_ghost_notes: whether to add quiet snare notes
        # - ghost_note_probability: 0-1, how often ghost notes appear
        # - ghost_note_velocity_range: (min, max) for ghost notes
        # - hihat_timing_mult: multiplier for hi-hat timing looseness (guide says loosest)
        # - snare_timing_mult: multiplier for snare timing
        # - kick_timing_mult: multiplier for kick timing (guide says tightest)
        # - velocity_range_override: (min, max) for main hits
        # - hihat_velocity_pattern: list of relative velocities for 8th note pattern
        # - fill_crescendo: whether fills build in velocity
        self._style_presets: Dict[str, Dict[str, Any]] = {
            "standard": {
                "complexity": 0.5,
                "vulnerability": 0.5,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.15,
                "ghost_note_velocity_range": (25, 45),
                "hihat_timing_mult": 1.5,  # Hi-hats loosest
                "snare_timing_mult": 1.0,
                "kick_timing_mult": 0.5,   # Kicks tightest
                "fill_crescendo": True,
            },
            "hip-hop": {
                "complexity": 0.35,
                "vulnerability": 0.65,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.18,
                "ghost_note_velocity_range": (30, 50),
                "hihat_timing_mult": 1.8,  # Lazy, behind-beat feel
                "snare_timing_mult": 1.2,
                "kick_timing_mult": 0.6,
                "velocity_range_override": (75, 115),
                "hihat_velocity_pattern": [70, 95, 65, 90, 68, 92, 62, 88],  # Upbeat accent
                "fill_crescendo": True,
                "swing_amount": 0.15,  # Slight swing
            },
            "rock": {
                "complexity": 0.45,
                "vulnerability": 0.45,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.12,
                "ghost_note_velocity_range": (30, 45),
                "hihat_timing_mult": 1.2,
                "snare_timing_mult": 0.8,  # Tighter snare for rock
                "kick_timing_mult": 0.4,   # Very tight kick
                "velocity_range_override": (90, 120),
                "hihat_velocity_pattern": [95, 65, 85, 60, 90, 68, 82, 58],  # Downbeat accent
                "fill_crescendo": True,
                "crash_on_one": True,
            },
            "jazzy": {
                "complexity": 0.6,
                "vulnerability": 0.65,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.25,
                "ghost_note_velocity_range": (25, 40),
                "hihat_timing_mult": 2.0,  # Very loose, breathing
                "snare_timing_mult": 1.5,
                "kick_timing_mult": 1.0,
                "velocity_range_override": (60, 100),
                "swing_amount": 0.25,  # Strong swing
                "ride_instead_of_hihat": True,
                "brush_probability": 0.3,
            },
            "edm": {
                "complexity": 0.2,
                "vulnerability": 0.25,
                "enable_ghost_notes": False,
                "ghost_note_probability": 0.0,
                "hihat_timing_mult": 0.5,  # Very tight, quantized feel
                "snare_timing_mult": 0.3,
                "kick_timing_mult": 0.2,   # Machine-tight
                "velocity_range_override": (100, 127),
                "fill_crescendo": False,
                "sidechain_kick": True,
            },
            "lofi": {
                "complexity": 0.3,
                "vulnerability": 0.75,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.28,
                "ghost_note_velocity_range": (20, 40),
                "hihat_timing_mult": 2.2,  # Very loose, dreamy
                "snare_timing_mult": 1.8,
                "kick_timing_mult": 1.2,
                "velocity_range_override": (40, 100),
                "hihat_velocity_pattern": [60, 50, 55, 45, 58, 48, 52, 42],  # Soft, muted
                "swing_amount": 0.2,
                "tape_saturation": True,
                "vinyl_noise": True,
            },
            "acoustic": {
                "complexity": 0.55,
                "vulnerability": 0.6,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.2,
                "ghost_note_velocity_range": (25, 45),
                "hihat_timing_mult": 1.6,
                "snare_timing_mult": 1.2,
                "kick_timing_mult": 0.8,
                "velocity_range_override": (65, 105),
                "room_mics": True,
                "bleed": True,
            },
            "metal": {
                "complexity": 0.7,
                "vulnerability": 0.35,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.1,
                "ghost_note_velocity_range": (35, 55),
                "hihat_timing_mult": 0.8,
                "snare_timing_mult": 0.6,
                "kick_timing_mult": 0.3,  # Very tight double kicks
                "velocity_range_override": (95, 127),
                "double_kick_enabled": True,
                "blast_beat_enabled": True,
                "fill_crescendo": True,
            },
            "funk": {
                "complexity": 0.65,
                "vulnerability": 0.55,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.3,  # Lots of ghost notes
                "ghost_note_velocity_range": (25, 45),
                "hihat_timing_mult": 1.4,
                "snare_timing_mult": 1.0,
                "kick_timing_mult": 0.6,
                "velocity_range_override": (70, 110),
                "hihat_velocity_pattern": [75, 90, 70, 95, 72, 88, 68, 92],
                "swing_amount": 0.1,
                "syncopation": 0.4,
            },
            "reggae": {
                "complexity": 0.4,
                "vulnerability": 0.6,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.15,
                "ghost_note_velocity_range": (30, 45),
                "hihat_timing_mult": 1.5,
                "snare_timing_mult": 1.3,
                "kick_timing_mult": 0.7,
                "velocity_range_override": (65, 100),
                "one_drop": True,  # Kick on 2 and 4, not 1
                "rim_click": True,
            },
            "rnb": {
                "complexity": 0.4,
                "vulnerability": 0.6,
                "enable_ghost_notes": True,
                "ghost_note_probability": 0.2,
                "ghost_note_velocity_range": (25, 40),
                "hihat_timing_mult": 1.6,
                "snare_timing_mult": 1.1,
                "kick_timing_mult": 0.8,
                "velocity_range_override": (60, 100),
                "swing_amount": 0.12,
                "finger_snaps": True,
            },
        }
        if config_path:
            self._load_presets(config_path)

    def create_preset_from_guide(self, style: str) -> GrooveSettings:
        """Create a `GrooveSettings` instance from a style keyword."""
        key = style.lower() if style else "standard"
        preset = self._style_presets.get(key, self._style_presets["standard"])
        # Filter to only GrooveSettings-compatible fields
        valid_fields = {
            "complexity", "vulnerability", "timing_sigma_override",
            "dropout_prob_override", "velocity_range_override",
            "kick_timing_mult", "snare_timing_mult", "hihat_timing_mult",
            "enable_ghost_notes", "ghost_note_probability", "ghost_note_velocity_mult",
        }
        filtered = {k: v for k, v in preset.items() if k in valid_fields}
        return GrooveSettings(**filtered)

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
