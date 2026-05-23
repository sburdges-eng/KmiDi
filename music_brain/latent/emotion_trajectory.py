"""Smooth (valence, arousal) trajectory planner across a song.

Closes **Emotion trajectory planning** and **Latent emotional
steering**. Waypoints define VA targets at specific bar positions;
``sample_bars(start, end)`` interpolates linearly between waypoints
and clamps to the [-1, 1] unit box. Optional blending with a per-user
bias (from ``music_brain.latent.feedback.UserModel.emotion_bias``)
combines an arc the composer specified with personalization the
listener earned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class EmotionWaypoint:
    bar: int
    valence: float
    arousal: float

    def __post_init__(self) -> None:
        if self.bar < 0:
            raise ValueError(f"bar must be >= 0, got {self.bar}")
        if not (-1.0 <= float(self.valence) <= 1.0):
            raise ValueError(f"valence must be in [-1, 1], got {self.valence}")
        if not (-1.0 <= float(self.arousal) <= 1.0):
            raise ValueError(f"arousal must be in [-1, 1], got {self.arousal}")


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


@dataclass(frozen=True)
class EmotionTrajectory:
    waypoints: List[EmotionWaypoint]

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("EmotionTrajectory requires at least one waypoint")
        last = -1
        for wp in self.waypoints:
            if wp.bar <= last:
                raise ValueError(
                    f"waypoints must be strictly monotonic by bar; "
                    f"got bar {wp.bar} after {last}")
            last = wp.bar

    def _va_at_bar(self, bar: int) -> Tuple[float, float]:
        wps = self.waypoints
        # Before first waypoint: hold the initial value.
        if bar <= wps[0].bar:
            return (float(wps[0].valence), float(wps[0].arousal))
        # After last waypoint: hold the terminal value.
        if bar >= wps[-1].bar:
            return (float(wps[-1].valence), float(wps[-1].arousal))
        # Linear interpolation between the bracketing waypoints.
        for i in range(len(wps) - 1):
            a, b = wps[i], wps[i + 1]
            if a.bar <= bar <= b.bar:
                if a.bar == b.bar:
                    return (float(a.valence), float(a.arousal))
                frac = (bar - a.bar) / (b.bar - a.bar)
                v = a.valence + (b.valence - a.valence) * frac
                ar = a.arousal + (b.arousal - a.arousal) * frac
                return (_clamp(v), _clamp(ar))
        # Unreachable given the monotonic check.
        return (float(wps[-1].valence), float(wps[-1].arousal))

    def sample_bars(self, *, start_bar: int, end_bar: int,
                    blend_with: Optional[Tuple[float, float]] = None,
                    blend: float = 0.0) -> Iterator[Tuple[int, float, float]]:
        if end_bar < start_bar:
            raise ValueError(
                f"end_bar ({end_bar}) must be >= start_bar ({start_bar})")
        if not (0.0 <= blend <= 1.0):
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        for bar in range(start_bar, end_bar):
            v, a = self._va_at_bar(bar)
            if blend_with is not None and blend > 0.0:
                v = (1.0 - blend) * v + blend * float(blend_with[0])
                a = (1.0 - blend) * a + blend * float(blend_with[1])
            yield bar, _clamp(v), _clamp(a)


__all__ = ["EmotionTrajectory", "EmotionWaypoint"]
