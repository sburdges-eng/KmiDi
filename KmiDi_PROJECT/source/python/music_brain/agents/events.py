#!/usr/bin/env python3
"""
Async Event-Driven Architecture for UnifiedHub.

Provides a lightweight event bus that supports:
- Async/await event handling
- Priority-based event processing
- Event filtering and transformation
- Request/response patterns for RPC-style calls

Usage:
    from music_brain.agents.events import EventBus, Event

    bus = EventBus()

    # Subscribe to events
    @bus.on("daw.play")
    async def on_play(event: Event):
        print(f"DAW playing: {event.data}")

    # Emit events
    await bus.emit("daw.play", {"tempo": 120})

    # Request/response pattern
    response = await bus.request("agent.ask", {"role": "composer", "query": "..."})
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

T = TypeVar("T")


# =============================================================================
# Event Types
# =============================================================================


class EventPriority(IntEnum):
    """Event processing priority."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15


@dataclass
class Event:
    """
    Event container with metadata.

    Attributes:
        type: Event type identifier (e.g., "daw.play", "voice.note_on")
        data: Event payload
        id: Unique event ID
        timestamp: When the event was created
        source: Optional source identifier
        priority: Processing priority
        propagate: Whether to continue propagation after handling
    """

    type: str
    data: Any = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    propagate: bool = True

    def stop_propagation(self) -> None:
        """Stop this event from propagating to other handlers."""
        self.propagate = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "priority": self.priority.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        """Deserialize from dictionary."""
        return cls(
            type=d["type"],
            data=d.get("data"),
            id=d.get("id", str(uuid.uuid4())[:8]),
            timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(),
            source=d.get("source"),
            priority=EventPriority(d.get("priority", EventPriority.NORMAL)),
        )


@dataclass
class EventResult:
    """Result from an event handler."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    handler_id: Optional[str] = None


# =============================================================================
# Event Handler
# =============================================================================


@dataclass
class EventHandler:
    """Registered event handler with metadata."""

    callback: Callable[[Event], Awaitable[Any]]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    priority: EventPriority = EventPriority.NORMAL
    once: bool = False
    filter_fn: Optional[Callable[[Event], bool]] = None

    def matches(self, event: Event) -> bool:
        """Check if this handler should process the event."""
        if self.filter_fn:
            return self.filter_fn(event)
        return True


# =============================================================================
# Event Bus
# =============================================================================


class EventBus:
    """
    Async event bus for decoupled communication.

    Features:
    - Wildcard subscriptions (e.g., "daw.*")
    - Priority-based handler ordering
    - One-time handlers
    - Request/response pattern for RPC
    - Event history for debugging
    """

    def __init__(self, history_size: int = 100):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._history: List[Event] = []
        self._history_size = history_size
        self._lock = asyncio.Lock()
        self._running = True
        self._stats = {
            "events_emitted": 0,
            "events_handled": 0,
            "errors": 0,
        }

    # =========================================================================
    # Subscription
    # =========================================================================

    def on(
        self,
        event_type: str,
        priority: EventPriority = EventPriority.NORMAL,
        once: bool = False,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> Callable:
        """
        Decorator to register an event handler.

        Args:
            event_type: Event type to listen for (supports wildcards like "daw.*")
            priority: Handler priority (higher = processed first)
            once: If True, handler is removed after first invocation
            filter_fn: Optional filter function

        Usage:
            @bus.on("daw.play")
            async def handle_play(event: Event):
                ...
        """

        def decorator(fn: Callable[[Event], Awaitable[Any]]) -> Callable:
            self.subscribe(event_type, fn, priority, once, filter_fn)
            return fn

        return decorator

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], Awaitable[Any]],
        priority: EventPriority = EventPriority.NORMAL,
        once: bool = False,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe to an event type.

        Returns:
            Handler ID for unsubscribing
        """
        handler = EventHandler(
            callback=callback,
            priority=priority,
            once=once,
            filter_fn=filter_fn,
        )
        self._handlers[event_type].append(handler)
        # Sort by priority (highest first)
        self._handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
        return handler.id

    def unsubscribe(self, event_type: str, handler_id: str) -> bool:
        """
        Unsubscribe a handler.

        Returns:
            True if handler was found and removed
        """
        handlers = self._handlers.get(event_type, [])
        for h in handlers:
            if h.id == handler_id:
                handlers.remove(h)
                return True
        return False

    def unsubscribe_all(self, event_type: Optional[str] = None) -> int:
        """
        Remove all handlers for an event type (or all if not specified).

        Returns:
            Number of handlers removed
        """
        if event_type:
            count = len(self._handlers.get(event_type, []))
            self._handlers[event_type] = []
            return count
        else:
            count = sum(len(h) for h in self._handlers.values())
            self._handlers.clear()
            return count

    # =========================================================================
    # Emission
    # =========================================================================

    async def emit(
        self,
        event_type: str,
        data: Any = None,
        source: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        wait: bool = True,
    ) -> List[EventResult]:
        """
        Emit an event.

        Args:
            event_type: Type of event
            data: Event payload
            source: Source identifier
            priority: Event priority
            wait: If True, wait for all handlers to complete

        Returns:
            List of results from handlers
        """
        event = Event(
            type=event_type,
            data=data,
            source=source,
            priority=priority,
        )

        return await self.emit_event(event, wait=wait)

    async def emit_event(self, event: Event, wait: bool = True) -> List[EventResult]:
        """
        Emit a pre-constructed Event.

        Args:
            event: Event to emit
            wait: If True, wait for handlers

        Returns:
            List of results from handlers
        """
        if not self._running:
            return []

        self._stats["events_emitted"] += 1

        # Add to history
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size :]

        # Find matching handlers
        handlers = self._get_matching_handlers(event.type)

        if not handlers:
            return []

        # Process handlers
        results: List[EventResult] = []
        handlers_to_remove: List[tuple] = []

        for event_type, handler in handlers:
            if not event.propagate:
                break

            if not handler.matches(event):
                continue

            try:
                result = await handler.callback(event)
                results.append(
                    EventResult(
                        success=True,
                        data=result,
                        handler_id=handler.id,
                    )
                )
                self._stats["events_handled"] += 1
            except Exception as e:
                results.append(
                    EventResult(
                        success=False,
                        error=str(e),
                        handler_id=handler.id,
                    )
                )
                self._stats["errors"] += 1

            # Mark once-handlers for removal
            if handler.once:
                handlers_to_remove.append((event_type, handler.id))

        # Remove once-handlers
        for et, hid in handlers_to_remove:
            self.unsubscribe(et, hid)

        return results

    def emit_sync(
        self,
        event_type: str,
        data: Any = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Emit an event from sync code (schedules on event loop).

        Note: Does not wait for handlers to complete.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event_type, data, source, wait=False))
        except RuntimeError:
            # No event loop - create one temporarily
            asyncio.run(self.emit(event_type, data, source, wait=False))

    def _get_matching_handlers(
        self, event_type: str
    ) -> List[tuple]:
        """Get all handlers matching an event type, including wildcards."""
        result: List[tuple] = []

        # Exact match
        for handler in self._handlers.get(event_type, []):
            result.append((event_type, handler))

        # Wildcard matches (e.g., "daw.*" matches "daw.play")
        parts = event_type.split(".")
        for i in range(len(parts)):
            wildcard = ".".join(parts[: i + 1]) + ".*"
            for handler in self._handlers.get(wildcard, []):
                result.append((wildcard, handler))

        # Global wildcard
        for handler in self._handlers.get("*", []):
            result.append(("*", handler))

        # Sort by priority
        result.sort(key=lambda x: x[1].priority, reverse=True)
        return result

    # =========================================================================
    # Request/Response Pattern
    # =========================================================================

    async def request(
        self,
        event_type: str,
        data: Any = None,
        timeout: float = 30.0,
    ) -> Any:
        """
        Request/response pattern - emit event and wait for response.

        The handler should call event.respond(data) or return a value.

        Args:
            event_type: Event type
            data: Request data
            timeout: Timeout in seconds

        Returns:
            Response data

        Raises:
            asyncio.TimeoutError: If timeout expires
            RuntimeError: If no response received
        """
        request_id = str(uuid.uuid4())[:8]
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        self._pending_requests[request_id] = future

        # Create request event with response callback
        event = Event(
            type=event_type,
            data={"request_id": request_id, "payload": data},
            source="request",
        )

        try:
            results = await asyncio.wait_for(
                self.emit_event(event),
                timeout=timeout,
            )

            # Check for direct return value
            for result in results:
                if result.success and result.data is not None:
                    return result.data

            # Check for pending response
            if not future.done():
                await asyncio.wait_for(future, timeout=timeout)

            return future.result()

        except asyncio.TimeoutError:
            raise
        finally:
            self._pending_requests.pop(request_id, None)

    def respond(self, request_id: str, data: Any) -> bool:
        """
        Respond to a request (called by handler).

        Args:
            request_id: ID from the request event
            data: Response data

        Returns:
            True if response was delivered
        """
        future = self._pending_requests.get(request_id)
        if future and not future.done():
            future.set_result(data)
            return True
        return False

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_history(self, event_type: Optional[str] = None) -> List[Event]:
        """Get event history, optionally filtered by type."""
        if event_type:
            return [e for e in self._history if e.type == event_type]
        return list(self._history)

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    @property
    def stats(self) -> Dict[str, int]:
        """Get bus statistics."""
        return dict(self._stats)

    def get_handler_count(self, event_type: Optional[str] = None) -> int:
        """Get number of registered handlers."""
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(h) for h in self._handlers.values())

    def shutdown(self) -> None:
        """Shutdown the event bus."""
        self._running = False
        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    def __repr__(self) -> str:
        return f"EventBus(handlers={self.get_handler_count()}, events={self._stats['events_emitted']})"


# =============================================================================
# Event Channels (Typed Events)
# =============================================================================


class EventChannel(Generic[T]):
    """
    Typed event channel for a specific data type.

    Usage:
        play_channel = EventChannel[PlayData](bus, "daw.play")

        @play_channel.on
        async def handle(data: PlayData):
            ...

        await play_channel.emit(PlayData(tempo=120))
    """

    def __init__(self, bus: EventBus, event_type: str):
        self._bus = bus
        self._event_type = event_type

    async def emit(self, data: T, **kwargs) -> List[EventResult]:
        """Emit typed data."""
        return await self._bus.emit(self._event_type, data, **kwargs)

    def on(
        self,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Callable:
        """Subscribe with typed callback."""

        def decorator(fn: Callable[[T], Awaitable[Any]]) -> Callable:
            async def wrapper(event: Event) -> Any:
                return await fn(event.data)

            self._bus.subscribe(self._event_type, wrapper, priority)
            return fn

        return decorator


# =============================================================================
# Async Queue for Background Processing
# =============================================================================


class EventQueue:
    """
    Async queue for background event processing.

    Useful for rate-limited operations or batching.
    """

    def __init__(self, bus: EventBus, max_size: int = 1000):
        self._bus = bus
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processor: Optional[Callable[[Event], Awaitable[None]]] = None

    async def start(
        self,
        processor: Optional[Callable[[Event], Awaitable[None]]] = None,
    ) -> None:
        """Start processing the queue."""
        self._running = True
        self._processor = processor
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop processing."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def put(self, event: Event) -> None:
        """Add an event to the queue."""
        await self._queue.put(event)

    def put_nowait(self, event: Event) -> bool:
        """Add event without waiting (returns False if full)."""
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                if self._processor:
                    await self._processor(event)
                else:
                    await self._bus.emit_event(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[EventQueue] Error processing event: {e}")

    @property
    def size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "Event",
    "EventResult",
    "EventHandler",
    "EventPriority",
    "EventBus",
    "EventChannel",
    "EventQueue",
]

