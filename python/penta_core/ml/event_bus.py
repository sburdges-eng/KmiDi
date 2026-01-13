"""
Event Bus - Decoupled pub/sub communication between components.

Provides:
- Pub/sub event system
- Event filtering and routing
- Async event processing
- Event replay for debugging
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Pattern
import re

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data structure."""

    event_type: str
    data: Dict[str, Any]
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            import uuid

            self.event_id = f"evt_{uuid.uuid4().hex[:8]}"


class EventBus:
    """
    Event bus for decoupled component communication.

    Supports:
    - Pub/sub pattern
    - Event filtering
    - Async processing
    - Event replay
    """

    def __init__(
        self,
        max_history: int = 1000,
        enable_replay: bool = True,
    ):
        """
        Initialize event bus.

        Args:
            max_history: Maximum number of events to keep in history
            enable_replay: Enable event replay functionality
        """
        self.max_history = max_history
        self.enable_replay = enable_replay

        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._wildcard_subscribers: List[Tuple[Pattern, Callable[[Event], None]]] = []
        self._history: deque = deque(maxlen=max_history)

        # Statistics
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "subscribers": 0,
        }

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], None],
        pattern: bool = False,
    ) -> None:
        """
        Subscribe to events.

        Args:
            event_type: Event type or pattern (if pattern=True)
            callback: Callback function
            pattern: If True, treat event_type as regex pattern
        """
        if pattern:
            try:
                compiled_pattern = re.compile(event_type)
                self._wildcard_subscribers.append((compiled_pattern, callback))
                self._stats["subscribers"] = len(self._subscribers) + len(
                    self._wildcard_subscribers
                )
                logger.debug(f"Subscribed to pattern: {event_type}")
            except re.error as e:
                logger.error(f"Invalid pattern {event_type}: {e}")
        else:
            self._subscribers[event_type].append(callback)
            self._stats["subscribers"] = len(self._subscribers) + len(self._wildcard_subscribers)
            logger.debug(f"Subscribed to event type: {event_type}")

    def unsubscribe(
        self,
        event_type: str,
        callback: Optional[Callable[[Event], None]] = None,
    ) -> None:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type
            callback: Optional specific callback to remove (removes all if None)
        """
        if callback:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
        else:
            self._subscribers[event_type].clear()

        # Also remove from wildcard subscribers
        if callback:
            self._wildcard_subscribers = [
                (pattern, cb) for pattern, cb in self._wildcard_subscribers if cb != callback
            ]

        self._stats["subscribers"] = len(self._subscribers) + len(self._wildcard_subscribers)
        logger.debug(f"Unsubscribed from event type: {event_type}")

    def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "",
        async_delivery: bool = True,
    ) -> Event:
        """
        Publish an event.

        Args:
            event_type: Event type
            data: Event data
            source: Event source identifier
            async_delivery: Deliver asynchronously

        Returns:
            Published event
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
        )

        # Add to history
        if self.enable_replay:
            self._history.append(event)

        self._stats["events_published"] += 1

        # Deliver to subscribers
        if async_delivery:
            try:
                # Try to get running event loop
                loop = asyncio.get_running_loop()
                # Schedule async delivery in existing loop
                loop.create_task(self._deliver_async(event))
            except RuntimeError:
                # No event loop running, fall back to sync delivery
                self._deliver(event)
        else:
            self._deliver(event)

        return event

    async def publish_async(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "",
    ) -> Event:
        """Publish an event asynchronously."""
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
        )

        if self.enable_replay:
            self._history.append(event)

        self._stats["events_published"] += 1
        await self._deliver_async(event)

        return event

    def _deliver(self, event: Event):
        """Deliver event to subscribers synchronously."""
        # Direct subscribers
        for callback in self._subscribers.get(event.event_type, []):
            try:
                callback(event)
                self._stats["events_delivered"] += 1
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

        # Pattern subscribers
        for pattern, callback in self._wildcard_subscribers:
            if pattern.match(event.event_type):
                try:
                    callback(event)
                    self._stats["events_delivered"] += 1
                except Exception as e:
                    logger.error(f"Error in pattern callback: {e}")

    async def _deliver_async(self, event: Event):
        """Deliver event to subscribers asynchronously."""
        # Direct subscribers
        for callback in self._subscribers.get(event.event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                self._stats["events_delivered"] += 1
            except Exception as e:
                logger.error(f"Error in async event callback: {e}")

        # Pattern subscribers
        for pattern, callback in self._wildcard_subscribers:
            if pattern.match(event.event_type):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                    self._stats["events_delivered"] += 1
                except Exception as e:
                    logger.error(f"Error in async pattern callback: {e}")

    def replay_events(
        self,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        callback: Optional[Callable[[Event], None]] = None,
    ) -> List[Event]:
        """
        Replay events from history.

        Args:
            event_type: Optional event type filter
            since: Optional timestamp filter (replay events after this time)
            callback: Optional callback to call for each replayed event

        Returns:
            List of replayed events
        """
        if not self.enable_replay:
            logger.warning("Event replay is disabled")
            return []

        replayed = []
        for event in self._history:
            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if since and event.timestamp < since:
                continue

            replayed.append(event)

            # Call callback if provided
            if callback:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in replay callback: {e}")

        logger.info(f"Replayed {len(replayed)} events")
        return replayed

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "events_published": self._stats["events_published"],
            "events_delivered": self._stats["events_delivered"],
            "subscribers": self._stats["subscribers"],
            "history_size": len(self._history),
            "max_history": self.max_history,
            "event_types": list(self._subscribers.keys()),
        }

    def clear_history(self):
        """Clear event history."""
        self._history.clear()
        logger.info("Event history cleared")


# Singleton event bus
_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:
                _event_bus = EventBus()
    return _event_bus
