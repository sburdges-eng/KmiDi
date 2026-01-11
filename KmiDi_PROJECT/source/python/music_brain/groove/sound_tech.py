"""
Sound Tech module.

The Sound Tech is responsible for the "Environment" and "Acoustics".
It parses EQ and Compression guides to apply mix settings, and simulates
venues (Reverb/Delay) based on the Fan Profile.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from music_brain.groove.guide_parser import EQGuideParser, CompressionGuideParser
from music_brain.groove.fan_feedback import FanProfile


@dataclass
class Venue:
    """Represents an acoustic environment."""
    name: str
    reverb_size: float  # 0.0 - 1.0
    reverb_decay: float  # seconds
    delay_time_ms: float
    delay_feedback: float  # 0.0 - 1.0
    description: str


class SoundTech:
    """
    The Sound Tech influences the environment to produce the best acoustics.
    """

    def __init__(self, vault_path: Optional[Path] = None):
        if not vault_path:
            root = Path(__file__).resolve().parent.parent.parent
            vault_path = root / "vault" / "Production_Guides"

        self.eq_parser = EQGuideParser(vault_path / "EQ Deep Dive Guide.md")
        self.comp_parser = CompressionGuideParser(
            vault_path / "Compression Deep Dive Guide.md")

        self.eq_rules = self.eq_parser.parse()
        self.comp_rules = self.comp_parser.parse()

        self.venues = {
            "studio": Venue("Studio", 0.2, 0.5, 0.0, 0.0, "Tight, dry, controlled."),
            "club": Venue("Small Club", 0.5, 1.2, 120.0, 0.2, "Intimate, sweaty, energetic."),
            "arena": Venue("Arena", 0.9, 3.5, 350.0, 0.4, "Huge, epic, booming."),
            "outdoor": Venue("Outdoor Festival", 0.1, 0.2, 250.0, 0.1, "Open air, slapback delay."),
            "basement": Venue("Basement", 0.3, 0.8, 40.0, 0.6, "Muddy, resonant, raw.")
        }

    def get_venue_for_fan(self, fan_profile: FanProfile) -> Venue:
        """Selects the best venue based on the fan profile."""
        # Heuristic mapping
        if fan_profile.name.lower() == "metalhead":
            # Metalheads like it tight (Studio) or Epic (Arena)
            # If they want it "ahead" (rushed), maybe Studio is better for tightness.
            if fan_profile.timing_slop_multiplier < 0.8:
                return self.venues["studio"]
            return self.venues["arena"]

        elif fan_profile.name.lower() == "reggae fan":
            # Reggae needs space and delay -> Outdoor or Club
            return self.venues["outdoor"]

        elif fan_profile.name.lower() == "jazz cat":
            # Jazz needs intimacy -> Club
            return self.venues["club"]

        # Default
        return self.venues["studio"]

    def get_mix_settings(self, instrument: str, fan_profile: Optional[FanProfile] = None) -> Dict[str, Any]:
        """
        Returns EQ and Compression settings for an instrument, adapted to the fan.
        """
        inst_key = instrument.lower()

        # 1. EQ Settings
        eq_ranges = self.eq_rules.get("instruments", {}).get(inst_key, {})

        # 2. Compression Settings
        # Default to 4:1 if not found
        ratio = self.comp_rules.get("ratios", {}).get(inst_key, 4)

        # Adapt to Fan
        if fan_profile:
            # If fan wants it "Harder" (velocity > 1.0), compress more?
            if fan_profile.velocity_multiplier > 1.1:
                ratio = min(20, int(ratio * 1.5))  # Increase ratio
            elif fan_profile.velocity_multiplier < 0.9:
                ratio = max(2, int(ratio * 0.7))  # Decrease ratio

        return {
            "eq": eq_ranges,
            "compression": {
                "ratio": f"{ratio}:1",
                "attack": "fast" if inst_key == "drums" else "medium"  # Simplified
            }
        }

    def set_stage(self, fan_profile: FanProfile) -> Dict[str, Any]:
        """
        Sets up the entire stage (Venue + Mix) for the band.
        """
        venue = self.get_venue_for_fan(fan_profile)

        return {
            "venue": venue,
            "mix_notes": f"Setting up {venue.name} for {fan_profile.name}.",
            "global_reverb": {
                "size": venue.reverb_size,
                "decay": venue.reverb_decay
            }
        }
