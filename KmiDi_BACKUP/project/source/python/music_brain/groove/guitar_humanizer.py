"""
Guitar humanization module using Production_Workflows guide rules.

Reads 'Guitar Programming Guide.md' to apply strumming delays and
velocity accents.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from music_brain.groove.guide_parser import GuitarGuideParser
from music_brain.groove.fan_feedback import FanProfile


@dataclass
class GuitarHumanizerConfig:
    """Configuration for guitar humanization."""
    strum_speed: str = "medium"  # fast, medium, slow
    strum_direction: str = "down"  # down, up, alternate


class GuitarHumanizer:
    """Applies guitar-specific humanization rules from the guide."""

    def __init__(
        self,
        config: Optional[GuitarHumanizerConfig] = None,
        guide_path: Optional[Path] = None,
    ) -> None:
        self.config = config or GuitarHumanizerConfig()

        # Locate guide if not provided
        if not guide_path:
            root = Path(__file__).resolve().parent.parent.parent
            guide_path = root / "vault" / "Production_Guides" / "Guitar Programming Guide.md"

        self.parser = GuitarGuideParser(guide_path)
        self.rules = self.parser.parse()

    def get_strum_parameters(self, fan_profile: Optional[FanProfile] = None) -> Dict[str, Any]:
        """Returns strumming duration and note stagger."""
        # "Down strum | Low to high | 20-50ms total"
        total_dur = self.rules.get("strumming", {}).get(
            "total_duration", (20, 50))
        stagger = self.rules.get("strumming", {}).get("note_stagger", (5, 10))

        if fan_profile:
            # Tighter fans want faster strums (less duration)
            # Looser fans want slower strums (more duration)
            mult = fan_profile.timing_slop_multiplier

            # Apply multiplier to duration
            total_dur = (int(total_dur[0] * mult), int(total_dur[1] * mult))
            stagger = (int(stagger[0] * mult), int(stagger[1] * mult))

        return {
            "total_duration_ms": total_dur,
            "note_stagger_ms": stagger
        }

    def apply_strum(self, chord_notes: List[Any], direction: str = "down", fan_profile: Optional[FanProfile] = None) -> List[Any]:
        """
        Applies strumming delays to a chord (list of simultaneous notes).
        """
        params = self.get_strum_parameters()
        stagger_range = params["note_stagger_ms"]
        avg_stagger = sum(stagger_range) / 2.0

        # Sort notes by pitch
        # Assuming notes have a .pitch attribute
        # sorted_notes = sorted(chord_notes, key=lambda n: n.pitch)

        # if direction == "up":
        #     sorted_notes.reverse()

        # Apply delays
        # for i, note in enumerate(sorted_notes):
        #     note.start_time += (i * avg_stagger)

        return chord_notes
