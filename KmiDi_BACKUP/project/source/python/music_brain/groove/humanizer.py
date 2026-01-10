"""
Lightweight groove humanization utilities.

This module provides deterministic, dependency-light helpers so tests can run
without requiring full MIDI backends. The implementations favor predictable
math over randomness to keep unit tests stable while still simulating
human-like variation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from music_brain.groove.extractor import GrooveTemplate


@dataclass
class HumanizationProfile:
    """Configuration for timing/velocity variation."""

    timing_variation: float = 0.0
    velocity_variation: float = 0.0
    swing_factor: float = 0.0
    groove_intensity: float = 1.0


class TimingHumanizer:
    """Apply timing variation and swing to note timestamps."""

    def apply_timing_variation(
        self,
        times: Sequence[float],
        variation: float,
    ) -> List[float]:
        if not variation:
            return list(times)

        adjusted: List[float] = []
        for idx, t in enumerate(times):
            if t <= 0:
                adjusted.append(max(0.0, float(t)))
                continue
            # Deterministic sinusoidal offset per note index.
            delta = t * variation * 0.25 * math.sin(idx + 1)
            adjusted.append(max(0.0, float(t + delta)))
        return adjusted

    def apply_swing(
        self,
        times: Sequence[float],
        swing_factor: float,
    ) -> List[float]:
        if not swing_factor:
            return list(times)
        if len(times) < 2:
            return list(times)

        base_step = max(0.0, float(times[1] - times[0]))
        swung: List[float] = []
        for idx, t in enumerate(times):
            offset = (
                base_step * swing_factor * 0.5
                if idx % 2 == 1
                else -base_step * swing_factor * 0.25
            )
            swung.append(max(0.0, float(t + offset)))
        return swung


class VelocityHumanizer:
    """Apply velocity variation and shaping."""

    def apply_velocity_variation(
        self,
        velocities: Sequence[float],
        variation: float,
    ) -> List[float]:
        if not variation:
            return list(velocities)

        adjusted: List[float] = []
        for idx, v in enumerate(velocities):
            delta = v * variation * 0.3 * math.cos(idx + 1)
            new_v = max(0, min(127, int(round(v + delta))))
            adjusted.append(new_v)
        return adjusted

    def apply_velocity_curve(
        self,
        velocities: Sequence[float],
        curve_type: str = "linear",
    ) -> List[float]:
        if not velocities:
            return []

        if curve_type == "exponential":
            max_v = max(velocities) or 1
            return [
                max(0, min(127, int((v / max_v) ** 1.2 * 127)))
                for v in velocities
            ]

        # Linear fallback that preserves ordering.
        if len(velocities) == 1:
            return [max(0, min(127, int(velocities[0])))]

        min_v, max_v = min(velocities), max(velocities)
        span = max(max_v - min_v, 1)
        return [
            max(0, min(127, int(((v - min_v) / span) * 127)))
            for v in velocities
        ]


class GrooveHumanizer:
    """Apply profiles or templates to note collections."""

    def __init__(self) -> None:
        self.timing_humanizer = TimingHumanizer()
        self.velocity_humanizer = VelocityHumanizer()

    def apply_profile_to_notes(
        self,
        notes: Iterable,
        profile: HumanizationProfile,
    ) -> List:
        notes = list(notes)
        if not notes:
            return []

        intensity = max(0.0, profile.groove_intensity)
        times = [getattr(n, "time", 0.0) for n in notes]
        velocities = [getattr(n, "velocity", 0) for n in notes]

        times = self.timing_humanizer.apply_timing_variation(
            times, profile.timing_variation * intensity
        )
        times = self.timing_humanizer.apply_swing(
            times, profile.swing_factor * intensity
        )
        velocities = self.velocity_humanizer.apply_velocity_variation(
            velocities, profile.velocity_variation * intensity
        )

        humanized = []
        for idx, note in enumerate(notes):
            clone = copy.copy(note)
            clone.time = float(times[idx])
            clone.velocity = max(0, min(127, int(velocities[idx])))
            humanized.append(clone)
        return humanized

    def apply_template_to_notes(
        self,
        notes: Iterable,
        template: GrooveTemplate,
    ) -> List:
        notes = list(notes)
        if not notes:
            return []

        intensity = max(0.0, getattr(template, "intensity", 1.0))
        timing_deviations = getattr(template, "timing_deviations", []) or []
        velocity_curve = getattr(template, "velocity_curve", []) or []

        times = []
        for idx, note in enumerate(notes):
            base_time = float(getattr(note, "time", 0.0))
            if timing_deviations:
                deviation = timing_deviations[idx % len(timing_deviations)]
                base_time = max(0.0, base_time + deviation * intensity)
            times.append(base_time)

        if getattr(template, "swing_factor", 0.0):
            times = self.timing_humanizer.apply_swing(
                times, template.swing_factor * intensity
            )

        velocities = []
        for idx, note in enumerate(notes):
            vel = float(getattr(note, "velocity", 0))
            if velocity_curve:
                target = velocity_curve[idx % len(velocity_curve)]
                target = target * 127 if 0 < target <= 1 else target
                vel = vel * (1 - intensity) + target * intensity
            velocities.append(max(0, min(127, int(round(vel)))))

        humanized = []
        for idx, note in enumerate(notes):
            clone = copy.copy(note)
            clone.time = float(times[idx])
            clone.velocity = int(velocities[idx])
            humanized.append(clone)
        return humanized


def apply_humanization(
    track,
    profile: HumanizationProfile | None = None,
    template: GrooveTemplate | None = None,
):
    """
    Apply either a HumanizationProfile or GrooveTemplate to a track-like object.

    The track is shallow-copied; the returned object always owns a new notes
    list to avoid mutating callers' mocks or MIDI wrappers.
    """
    if profile is None and template is None:
        raise ValueError("Provide a HumanizationProfile or GrooveTemplate")

    humanizer = GrooveHumanizer()
    notes = getattr(track, "notes", [])

    if template is not None:
        new_notes = humanizer.apply_template_to_notes(notes, template)
    else:
        new_notes = humanizer.apply_profile_to_notes(notes, profile)  # type: ignore[arg-type]

    new_track = copy.copy(track)
    new_track.notes = new_notes
    return new_track


__all__ = [
    "GrooveTemplate",
    "HumanizationProfile",
    "TimingHumanizer",
    "VelocityHumanizer",
    "GrooveHumanizer",
    "apply_humanization",
]
