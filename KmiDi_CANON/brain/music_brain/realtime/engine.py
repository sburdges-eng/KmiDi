"""
High-level realtime engine coordinating clock, scheduler, and transports.

This is an intentionally small abstraction for Phase 3 prototyping. It does
not spawn background threads; callers are expected to invoke `process_tick`
regularly (e.g., via asyncio loop or JUCE timer) until a more sophisticated
transport layer is needed.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from music_brain.realtime.clock import RealtimeClock
from music_brain.realtime.events import ScheduledEvent
from music_brain.realtime.scheduler import EventScheduler
from music_brain.realtime.transport import BaseTransport

try:
    from music_brain.structure.comprehensive_engine import NoteEvent
except ImportError:  # pragma: no cover - fallback for lightweight environments
    from music_brain.realtime.events import NoteEvent  # type: ignore


class RealtimeEngine:
    """Coordinates realtime streaming of DAiW note events."""

    def __init__(self, tempo_bpm: float = 82.0, ppq: int = 960, lookahead_beats: float = 2.0) -> None:
        self.clock = RealtimeClock(tempo_bpm=tempo_bpm, ppq=ppq)
        self.scheduler = EventScheduler()
        self.lookahead_ticks = int(lookahead_beats * self.clock.ppq)
        self._transports: List[BaseTransport] = []
        self._running = False

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def add_transport(self, transport: BaseTransport) -> None:
        """Register a transport (MIDI, OSC, etc.) to receive events."""

        transport.attach_clock(self.clock)
        self._transports.append(transport)

    def remove_transport(self, transport: BaseTransport) -> None:
        if transport in self._transports:
            self._transports.remove(transport)
            transport.detach_clock()
            transport.close()

    def clear_transports(self) -> None:
        for transport in self._transports:
            transport.detach_clock()
            transport.close()
        self._transports.clear()

    # ------------------------------------------------------------------ #
    # Event ingestion
    # ------------------------------------------------------------------ #
    def load_note_events(
        self,
        note_events: Sequence[NoteEvent],
        channel: int = 0,
    ) -> None:
        """
        Convert raw NoteEvents into ScheduledEvents and queue them.

        Args:
            note_events: Sequence of DAiW NoteEvents (typically from harmony/groove).
            channel: MIDI channel (0-15) for emitted events.
        """
        if not note_events:
            return
        
        # Pre-allocate list for better performance
        scheduled = [None] * len(note_events)
        for idx, note in enumerate(note_events):
            scheduled[idx] = ScheduledEvent(
                start_tick=note.start_tick,
                note_event=note,
                channel=channel,
                event_id=str(idx),
            )
        self.scheduler.schedule_many(scheduled)

    def queue_events(self, events: Iterable[ScheduledEvent]) -> None:
        """Allow callers to push pre-built scheduled events."""

        self.scheduler.schedule_many(events)

    def reset_queue(self) -> None:
        self.scheduler.clear()

    # ------------------------------------------------------------------ #
    # Transport control
    # ------------------------------------------------------------------ #
    def start(self, start_tick: int = 0) -> None:
        self.clock.reset(start_tick=start_tick)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self.stop()
        self.clear_transports()

    # ------------------------------------------------------------------ #
    # Main processing hook
    # ------------------------------------------------------------------ #
    def process_tick(self) -> int:
        """
        Emit all events whose start_tick is within the lookahead window.

        Returns:
            Number of events emitted during this call.
        """
        # Fast path: early returns for common cases
        if not self._running:
            return 0
        
        transports = self._transports
        if not transports:
            return 0

        current_tick = self.clock.now_ticks()
        due = self.scheduler.pop_due_events(current_tick, self.lookahead_ticks)

        if not due:
            return 0

        # Emit to all transports (typically 1-2 transports)
        for transport in transports:
            transport.emit(due)

        return len(due)

