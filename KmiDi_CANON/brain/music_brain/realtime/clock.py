"""
Tempo-aware clock utilities for the realtime engine.

The clock converts between musical time (ticks / bars) and wall-clock time.
It does not manage scheduling threads; instead, the engine queries it to know
how many ticks should have elapsed since the transport started.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ClockSnapshot:
    """Immutable view of the clock state, useful for debug/metrics."""

    tempo_bpm: float
    ppq: int
    start_tick: int
    started_at: float  # monotonic seconds


class RealtimeClock:
    """High-resolution tempo clock."""

    def __init__(self, tempo_bpm: float = 82.0, ppq: int = 960) -> None:
        if tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be > 0")
        if ppq <= 0:
            raise ValueError("ppq must be > 0")

        self._tempo_bpm = float(tempo_bpm)
        self._ppq = int(ppq)
        self._start_tick = 0
        self._start_monotonic = time.monotonic()
        # Cache ticks_per_second to avoid recalculation
        self._ticks_per_second = (self._tempo_bpm / 60.0) * self._ppq
        self._seconds_per_tick = 1.0 / self._ticks_per_second

    # --------------------------------------------------------------------- #
    # Tempo / PPQ configuration
    # --------------------------------------------------------------------- #
    @property
    def tempo_bpm(self) -> float:
        return self._tempo_bpm

    def set_tempo(self, tempo_bpm: float) -> None:
        """Adjust tempo while preserving musical position."""

        if tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be > 0")
        current_tick = self.now_ticks()
        self._tempo_bpm = float(tempo_bpm)
        self._ticks_per_second = (self._tempo_bpm / 60.0) * self._ppq
        self._seconds_per_tick = 1.0 / self._ticks_per_second
        self.reset(start_tick=current_tick)

    @property
    def ppq(self) -> int:
        return self._ppq

    # --------------------------------------------------------------------- #
    # Time conversion helpers
    # --------------------------------------------------------------------- #
    def ticks_per_second(self) -> float:
        """Number of ticks that elapse each second for the current tempo."""

        return self._ticks_per_second

    def ticks_to_seconds(self, ticks: int) -> float:
        return ticks * self._seconds_per_tick

    def seconds_to_ticks(self, seconds: float) -> int:
        return int(seconds * self._ticks_per_second)

    # --------------------------------------------------------------------- #
    # State management
    # --------------------------------------------------------------------- #
    def reset(self, start_tick: int = 0) -> None:
        """Restart the clock and set a new reference tick."""

        self._start_tick = start_tick
        self._start_monotonic = time.monotonic()

    def snapshot(self) -> ClockSnapshot:
        return ClockSnapshot(
            tempo_bpm=self._tempo_bpm,
            ppq=self._ppq,
            start_tick=self._start_tick,
            started_at=self._start_monotonic,
        )

    # --------------------------------------------------------------------- #
    # Query interface
    # --------------------------------------------------------------------- #
    def now_ticks(self) -> int:
        """Current musical position in ticks since the latest reset."""

        elapsed_seconds = time.monotonic() - self._start_monotonic
        return self._start_tick + int(elapsed_seconds * self._ticks_per_second)

