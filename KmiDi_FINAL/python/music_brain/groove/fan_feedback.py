"""
Fan Feedback & Learning Module.

Allows the band (Humanizers) to adapt to user feedback and playstyle.
Stores 'Fan Profiles' that modulate the base guide rules.
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


@dataclass
class FanProfile:
    """Represents the preferences of a specific audience or user playstyle."""
    name: str = "Default Fan"
    description: str = "Standard listener"

    # Multipliers/Offsets for Humanization
    velocity_multiplier: float = 1.0      # >1.0 = Harder, <1.0 = Softer
    timing_slop_multiplier: float = 1.0   # >1.0 = Looser, <1.0 = Tighter
    swing_modifier: float = 0.0           # Additive swing (0.0-1.0)

    # Preference Overrides
    preferred_feel: Optional[str] = None  # "ahead", "behind", "on_grid"

    def apply_velocity(self, min_v: int, max_v: int) -> Tuple[int, int]:
        """Scale velocity range by multiplier, clamping to 1-127."""
        center = (min_v + max_v) / 2.0
        spread = (max_v - min_v) / 2.0

        # Scale the center (intensity)
        new_center = center * self.velocity_multiplier

        # Scale spread by slop multiplier (looser = more dynamic range usually)
        new_spread = spread * self.timing_slop_multiplier

        new_min = max(1, min(127, int(new_center - new_spread)))
        new_max = max(1, min(127, int(new_center + new_spread)))

        # Ensure min <= max
        if new_min > new_max:
            new_min, new_max = new_max, new_min

        return (new_min, new_max)

    def apply_timing_offset(self, base_offset_ms: float) -> float:
        """Modulate timing offset based on preferred feel."""
        if self.preferred_feel == "ahead":
            # If base is behind (>0), force it negative or reduce it
            return -10.0 if base_offset_ms >= 0 else base_offset_ms * 1.5
        elif self.preferred_feel == "behind":
            # If base is ahead (<0), force it positive
            return 15.0 if base_offset_ms <= 0 else base_offset_ms * 1.5
        elif self.preferred_feel == "on_grid":
            return 0.0

        return base_offset_ms

    def apply_swing(self, base_swing: float) -> float:
        """Apply swing modifier."""
        return min(1.0, max(0.0, base_swing + self.swing_modifier))


class FanBase:
    """Manages the crowd and learns from them."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.profiles: Dict[str, FanProfile] = {
            "default": FanProfile(),
            "metalhead": FanProfile(
                name="Metalhead",
                description="Likes it fast, hard, and tight.",
                velocity_multiplier=1.2,
                timing_slop_multiplier=0.5,
                preferred_feel="ahead"
            ),
            "reggae_fan": FanProfile(
                name="Reggae Fan",
                description="Likes it laid back and swinging.",
                velocity_multiplier=0.8,
                timing_slop_multiplier=1.2,
                swing_modifier=0.15,
                preferred_feel="behind"
            ),
            "jazz_cat": FanProfile(
                name="Jazz Cat",
                description="Maximum dynamics and swing.",
                velocity_multiplier=0.9,
                timing_slop_multiplier=1.5,
                swing_modifier=0.2
            )
        }
        self.active_profile: FanProfile = self.profiles["default"]
        self.storage_path = storage_path

    def set_active_fan(self, name: str):
        """Switch the active fan profile."""
        if name.lower() in self.profiles:
            self.active_profile = self.profiles[name.lower()]
        else:
            print(f"Fan '{name}' not found. Using default.")
            self.active_profile = self.profiles["default"]

    def learn_from_feedback(self, feedback_type: str):
        """Adjust active profile based on direct feedback."""
        p = self.active_profile
        if feedback_type == "play_harder":
            p.velocity_multiplier *= 1.1
        elif feedback_type == "play_softer":
            p.velocity_multiplier *= 0.9
        elif feedback_type == "more_swing":
            p.swing_modifier += 0.05
        elif feedback_type == "tighter":
            p.timing_slop_multiplier *= 0.9
        elif feedback_type == "looser":
            p.timing_slop_multiplier *= 1.1

        # Clamp values
        p.swing_modifier = min(1.0, max(0.0, p.swing_modifier))
        p.velocity_multiplier = min(1.5, max(0.5, p.velocity_multiplier))

    def learn_from_playstyle(self, analysis_metrics: Dict[str, Any]):
        """
        Auto-tune profile based on analyzed performance metrics.
        Expected keys: 'avg_velocity', 'swing_detected', 'timing_consistency'
        """
        p = self.active_profile

        # 1. Velocity Matching
        if "avg_velocity" in analysis_metrics:
            # Normalize 0-127 to 0.5-1.5 multiplier range roughly
            avg = analysis_metrics["avg_velocity"]
            target_mult = avg / 100.0
            # Smooth transition (learning rate)
            p.velocity_multiplier = (
                p.velocity_multiplier * 0.8) + (target_mult * 0.2)

        # 2. Swing Matching
        if "swing_detected" in analysis_metrics:
            swing = analysis_metrics["swing_detected"]
            p.swing_modifier = (p.swing_modifier * 0.8) + (swing * 0.2)

        # 3. Timing/Slop
        if "timing_consistency" in analysis_metrics:
            # High consistency (1.0) -> Low slop (0.5)
            # Low consistency (0.0) -> High slop (1.5)
            consistency = analysis_metrics["timing_consistency"]
            target_slop = 1.5 - consistency
            p.timing_slop_multiplier = (
                p.timing_slop_multiplier * 0.8) + (target_slop * 0.2)
