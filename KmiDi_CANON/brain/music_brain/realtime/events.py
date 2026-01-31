"""
Event primitives used by the realtime engine.

These dataclasses intentionally mirror the structure defined in
`music_brain.structure.comprehensive_engine.NoteEvent` so that the
streaming layer can reuse existing harmony/groove code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

try:
    # Preferred import – keeps a single definition of NoteEvent across the app.
    from music_brain.structure.comprehensive_engine import NoteEvent
except ImportError:  # pragma: no cover - fallback for lightweight environments
    @dataclass
    class NoteEvent:  # type: ignore[override]
        """Minimal stub so realtime module can be imported without Logic deps."""

        pitch: int
        velocity: int
        start_tick: int
        duration_ticks: int


class ControlEventType(Enum):
    """Runtime control messages sent to the engine."""

    START = auto()
    STOP = auto()
    SEEK = auto()
    TEMPO = auto()
    RULE_BREAK = auto()
    SCENE = auto()
    LOOP = auto()


@dataclass(order=True)
class ScheduledEvent:
    """
    Wrapper around a NoteEvent with scheduling metadata.

    Ordering is determined by `start_tick` so the scheduler can use heapq
    without additional key functions.
    """

    start_tick: int
    note_event: NoteEvent = field(compare=False)
    channel: int = field(default=0, compare=False)
    event_id: str = field(default="", compare=False)
    metadata: Dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def end_tick(self) -> int:
        """Convenience accessor used by lookahead heuristics."""

        return self.start_tick + self.note_event.duration_ticks


@dataclass
class ControlEvent:
    """Control-plane messages that affect the transport or engine state."""

    event_type: ControlEventType
    value: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetricEvent:
    """Lightweight telemetry for observability."""

    name: str
    value: float
    metadata: Dict[str, float] = field(default_factory=dict)

