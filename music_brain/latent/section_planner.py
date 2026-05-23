"""Hierarchical structural planner: sections → bars → beats.

Closes a cluster of structural-awareness spec items:

  - Bar/beat structural awareness
  - Section-aware generation
  - Phrase-boundary prediction
  - Structural phrase planning
  - Hierarchical generation architecture
  - Intent-level planning

The planner is the top-down counterpart to the bottom-up latent
emission: it tells the decoder *where* in a song we are. A
``SectionPlan`` is a contiguous, non-overlapping sequence of
``Section``s, each with a role (intro/verse/chorus/…) and a length in
bars. ``BarPosition`` is the instantaneous (bar, beat) cursor that
realtime emission steps through.

Phrase boundaries fall on section starts by default; ``emit_internal``
splits each section at its midpoint, mirroring the most common 8-bar
verse / 4-bar phrase structure in pop arrangements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class SectionRole(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    BUILD = "build"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"


@dataclass(frozen=True)
class Section:
    role: SectionRole
    start_bar: int
    length_bars: int

    def __post_init__(self) -> None:
        if self.start_bar < 0:
            raise ValueError(f"start_bar must be >= 0, got {self.start_bar}")
        if self.length_bars <= 0:
            raise ValueError(f"length_bars must be > 0, got {self.length_bars}")

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.length_bars


@dataclass(frozen=True)
class BarPosition:
    bar: int
    beat: int
    beats_per_bar: int

    def __post_init__(self) -> None:
        if self.beats_per_bar <= 0:
            raise ValueError("beats_per_bar must be > 0")
        if not (0 <= self.beat < self.beats_per_bar):
            raise ValueError(
                f"beat must be in [0, {self.beats_per_bar}), got {self.beat}")
        if self.bar < 0:
            raise ValueError(f"bar must be >= 0, got {self.bar}")

    def step(self) -> "BarPosition":
        nb = self.beat + 1
        if nb >= self.beats_per_bar:
            return BarPosition(bar=self.bar + 1, beat=0,
                               beats_per_bar=self.beats_per_bar)
        return BarPosition(bar=self.bar, beat=nb,
                           beats_per_bar=self.beats_per_bar)


@dataclass(frozen=True)
class SectionPlan:
    sections: Tuple[Section, ...]
    beats_per_bar: int

    def __post_init__(self) -> None:
        if self.beats_per_bar <= 0:
            raise ValueError("beats_per_bar must be > 0")
        if not self.sections:
            return
        expected = self.sections[0].start_bar
        for s in self.sections:
            if s.start_bar > expected:
                raise ValueError(
                    f"gap between sections at bar {expected}")
            if s.start_bar < expected:
                raise ValueError(
                    f"overlap at bar {s.start_bar} (next expected {expected})")
            expected = s.end_bar

    @property
    def total_bars(self) -> int:
        if not self.sections:
            return 0
        return self.sections[-1].end_bar - self.sections[0].start_bar

    def section_at_bar(self, bar: int) -> Optional[Section]:
        for s in self.sections:
            if s.start_bar <= bar < s.end_bar:
                return s
        return None

    def phrase_boundary_bars(self, *, emit_internal: bool = False) -> Tuple[int, ...]:
        out: list[int] = []
        for s in self.sections:
            out.append(s.start_bar)
            if emit_internal and s.length_bars >= 4:
                mid = s.start_bar + s.length_bars // 2
                if mid != s.start_bar:
                    out.append(mid)
        return tuple(sorted(set(out)))

    def bars_until_next_boundary(self, bar: int, *,
                                 emit_internal: bool = False) -> int:
        bounds = self.phrase_boundary_bars(emit_internal=emit_internal)
        for b in bounds:
            if b >= bar:
                return b - bar
        return -1


class SectionPlanner:
    """Construct ``SectionPlan`` instances from intent-level inputs."""

    def __init__(self, *, beats_per_bar: int = 4) -> None:
        if beats_per_bar <= 0:
            raise ValueError("beats_per_bar must be > 0")
        self._beats_per_bar = int(beats_per_bar)

    def plan_default(self) -> SectionPlan:
        """A standard 16-bar verse-chorus arrangement.

        4 intro + 4 verse + 4 chorus + 4 outro = 16 bars.
        """
        return SectionPlan(
            sections=(
                Section(role=SectionRole.INTRO, start_bar=0, length_bars=4),
                Section(role=SectionRole.VERSE, start_bar=4, length_bars=4),
                Section(role=SectionRole.CHORUS, start_bar=8, length_bars=4),
                Section(role=SectionRole.OUTRO, start_bar=12, length_bars=4),
            ),
            beats_per_bar=self._beats_per_bar,
        )


__all__ = [
    "BarPosition",
    "Section",
    "SectionPlan",
    "SectionPlanner",
    "SectionRole",
]
