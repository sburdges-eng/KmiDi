"""
Bass humanization module using Production_Workflows guide rules.

Reads 'Bass Programming Guide.md' to apply timing shifts (pocket) and
velocity dynamics (ghost notes, accents).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from music_brain.groove.guide_parser import BassGuideParser
from music_brain.groove.fan_feedback import FanProfile


@dataclass
class BassHumanizerConfig:
    """Configuration for bass humanization."""
    style: str = "standard"
    timing_feel: str = "on_grid"  # on_grid, behind, ahead


class BassHumanizer:
    """Applies bass-specific humanization rules from the guide."""

    def __init__(
        self,
        config: Optional[BassHumanizerConfig] = None,
        guide_path: Optional[Path] = None,
    ) -> None:
        self.config = config or BassHumanizerConfig()

        # Locate guide if not provided
        if not guide_path:
            root = Path(__file__).resolve().parent.parent.parent
            guide_path = root / "vault" / "Production_Guides" / "Bass Programming Guide.md"

        self.parser = BassGuideParser(guide_path)
        self.rules = self.parser.parse()

    def get_timing_offset(self, feel: Optional[str] = None, fan_profile: Optional[FanProfile] = None) -> float:
        """Returns timing offset in ms based on feel (behind/ahead)."""
        feel = feel or self.config.timing_feel

        # Calculate base offset from guide
        base_offset = 0.0
        if feel == "behind":
            r = self.rules.get("timing", {}).get("behind", (10, 30))
            base_offset = sum(r) / 2.0
        elif feel == "ahead":
            r = self.rules.get("timing", {}).get("ahead", (5, 15))
            base_offset = -(sum(r) / 2.0)

        # Apply Fan Profile modulation
        if fan_profile:
            return fan_profile.apply_timing_offset(base_offset)

        return base_offset

    def get_velocity_range(self, note_type: str = "root", fan_profile: Optional[FanProfile] = None) -> Tuple[int, int]:
        """Returns velocity range for a given note type."""
        # root, ghost, passing
        base_range = self.rules.get("velocity", {}).get(note_type, (80, 100))

        if fan_profile:
            return fan_profile.apply_velocity(base_range[0], base_range[1])

        return base_range

    def apply(self, midi_events: List[Any], feel: Optional[str] = None, fan_profile: Optional[FanProfile] = None) -> List[Any]:
        """
        Apply humanization to a list of MIDI events.
        (Stub implementation - would modify timing/velocity in place)
        """
        offset = self.get_timing_offset(feel, fan_profile)
        # In a real implementation, we would iterate events and shift timestamps
        # and randomize velocities based on self.rules["humanize"]["velocity_range"]
        return midi_events
