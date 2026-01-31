"""
Deterministic event scheduler for the realtime engine.

Uses a min-heap keyed by `ScheduledEvent.start_tick` so we can efficiently
fetch all events that should be emitted within a lookahead window.
"""

from __future__ import annotations

import heapq
from typing import Iterable, List, Optional

from music_brain.realtime.events import ScheduledEvent


class EventScheduler:
    """Priority queue wrapper for scheduled note events."""

    def __init__(self) -> None:
        self._queue: List[ScheduledEvent] = []

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #
    def schedule(self, event: ScheduledEvent) -> None:
        """Add a single event to the queue."""

        heapq.heappush(self._queue, event)

    def schedule_many(self, events: Iterable[ScheduledEvent]) -> None:
        """
        Bulk insert events (more efficient than repeated pushes).
        
        For large batches, we extend the list and heapify once, which is O(n)
        instead of O(n log n) for repeated heappush operations.
        """
        events_list = list(events) if not isinstance(events, list) else events
        if not events_list:
            return
        
        # For small batches, heappush is fine. For large batches, extend + heapify is faster.
        if len(events_list) < 100:
            for event in events_list:
                heapq.heappush(self._queue, event)
        else:
            self._queue.extend(events_list)
            heapq.heapify(self._queue)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def peek_next_tick(self) -> Optional[int]:
        """Return the tick of the next event without removing it."""

        if not self._queue:
            return None
        return self._queue[0].start_tick

    def pop_due_events(self, current_tick: int, lookahead_ticks: int) -> List[ScheduledEvent]:
        """
        Return all events scheduled to start within the lookahead window.

        Args:
            current_tick: Current transport position in ticks.
            lookahead_ticks: Events with start_tick <= current_tick + lookahead
                are considered due and will be removed from the queue.
        """

        due: List[ScheduledEvent] = []
        window = current_tick + lookahead_ticks

        while self._queue and self._queue[0].start_tick <= window:
            due.append(heapq.heappop(self._queue))

        return due

