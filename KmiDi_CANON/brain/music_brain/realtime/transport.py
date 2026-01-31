"""
Transport abstractions for emitting scheduled events.

The realtime engine can host multiple transports at once (e.g., send MIDI to a
virtual port and simultaneously broadcast JSON over WebSocket). Each transport
only needs to implement `emit` and `close`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from threading import Lock, Timer, current_thread
from typing import Dict, Iterable, Optional, Set, Tuple, TYPE_CHECKING

from music_brain.realtime.events import ScheduledEvent

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from music_brain.realtime.clock import RealtimeClock


class BaseTransport(ABC):
    """Abstract base for realtime transports."""

    def __init__(self) -> None:
        self._clock: Optional["RealtimeClock"] = None

    @abstractmethod
    def emit(self, events: Iterable[ScheduledEvent]) -> None:
        """Send scheduled events to the underlying output."""

    def emit_metrics(self, *_args, **_kwargs) -> None:  # pragma: no cover - optional
        """Hook for subclasses that want telemetry (no-op by default)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (MIDI ports, sockets, etc.)."""

    # ------------------------------------------------------------------ #
    # Tempo context
    # ------------------------------------------------------------------ #
    def attach_clock(self, clock: "RealtimeClock") -> None:
        """Allow transports to convert ticks→seconds when needed."""

        self._clock = clock

    def detach_clock(self) -> None:
        """Detach the tempo reference (called during shutdown)."""

        self._clock = None


class MidiTransport(BaseTransport):
    """
    Simple MIDI transport powered by `mido.open_output`.

    Any import errors are deferred until instantiation so the realtime module
    can be imported on systems without mido (e.g., Cloud CI).
    """

    def __init__(self, port_name: Optional[str] = None) -> None:
        super().__init__()
        try:
            import mido
        except ImportError as exc:  # pragma: no cover - fallback path
            raise RuntimeError("MidiTransport requires the 'mido' package") from exc

        self._mido = mido
        self._port = mido.open_output(port_name) if port_name else mido.open_output()
        self._send_lock = Lock()
        self._notes_lock = Lock()
        self._timers_lock = Lock()
        self._active_notes: Dict[Tuple[int, int], int] = {}
        self._active_timers: Set[Timer] = set()

    def emit(self, events: Iterable[ScheduledEvent]) -> None:
        """
        Emit scheduled MIDI events to the output port.
        
        Note: Mido's real-time output ports ignore the `time` parameter in messages.
        We use threading.Timer to schedule note_off events after the note duration,
        rather than relying on Mido's time parameter which only works for file-based
        MIDI operations.
        """
        for scheduled in events:
            note = scheduled.note_event
            # Send note_on immediately
            self._send_note_on(note.pitch, note.velocity, scheduled.channel)
            
            # Calculate duration in seconds (converting from ticks if clock is attached)
            duration_seconds = self._duration_seconds(scheduled)
            
            if duration_seconds > 0:
                # Schedule note_off using Timer (Mido's time parameter doesn't work for real-time ports)
                self._schedule_note_off(note.pitch, scheduled.channel, duration_seconds)
            else:
                # Fall back to immediate note_off for zero/unknown durations
                self._send_note_off(note.pitch, scheduled.channel)

    def close(self) -> None:
        # Stop outstanding timers so no callbacks fire after close
        with self._timers_lock:
            for timer in list(self._active_timers):
                timer.cancel()
            self._active_timers.clear()

        # Flush any notes that are still held to avoid stuck notes
        with self._notes_lock:
            remaining = list(self._active_notes.items())
            self._active_notes.clear()

        for (pitch, channel), count in remaining:
            for _ in range(count):
                self._send_note_off(pitch, channel, update_registry=False)

        self._port.close()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _send_note_on(self, pitch: int, velocity: int, channel: int) -> None:
        message = self._mido.Message(
            "note_on",
            note=pitch,
            velocity=velocity,
            channel=channel,
            time=0,
        )
        self._safe_send(message)
        self._register_note(pitch, channel)

    def _send_note_off(self, pitch: int, channel: int, *, update_registry: bool = True) -> None:
        message = self._mido.Message(
            "note_off",
            note=pitch,
            velocity=0,
            channel=channel,
            time=0,
        )
        self._safe_send(message)
        if update_registry:
            self._release_note(pitch, channel)

    def _safe_send(self, message: object) -> None:
        with self._send_lock:
            self._port.send(message)

    def _register_note(self, pitch: int, channel: int) -> None:
        key = (pitch, channel)
        with self._notes_lock:
            self._active_notes[key] = self._active_notes.get(key, 0) + 1

    def _release_note(self, pitch: int, channel: int) -> None:
        key = (pitch, channel)
        with self._notes_lock:
            if key not in self._active_notes:
                return
            remaining = self._active_notes[key] - 1
            if remaining <= 0:
                self._active_notes.pop(key, None)
            else:
                self._active_notes[key] = remaining

    def _duration_seconds(self, scheduled: ScheduledEvent) -> float:
        """Convert note duration from ticks to seconds, with fallbacks."""
        note = scheduled.note_event
        # Direct attribute access is faster than getattr
        duration_ticks = getattr(note, "duration_ticks", None)
        if duration_ticks is None or duration_ticks <= 0:
            return 0.0

        # Fast path: use attached clock (most common case)
        clock = self._clock
        if clock is not None:
            return clock.ticks_to_seconds(duration_ticks)

        # Fallback to metadata if the transport wasn't attached to a clock (legacy usage).
        metadata = scheduled.metadata
        duration_seconds = metadata.get("duration_seconds")
        if duration_seconds is not None:
            return float(duration_seconds)

        seconds_per_tick = metadata.get("seconds_per_tick")
        if seconds_per_tick is not None:
            return float(seconds_per_tick) * duration_ticks

        return 0.0

    def _schedule_note_off(self, pitch: int, channel: int, duration_seconds: float) -> None:
        """
        Schedule a note_off event using threading.Timer.
        
        This is necessary because Mido's real-time output ports ignore the `time`
        parameter in MIDI messages. The time parameter only works for file-based
        MIDI operations, not real-time streaming.
        
        Args:
            pitch: MIDI note number (0-127)
            channel: MIDI channel (0-15)
            duration_seconds: Delay before sending note_off
        """
        timer = Timer(duration_seconds, self._timer_callback, args=(pitch, channel))
        timer.daemon = True
        with self._timers_lock:
            self._active_timers.add(timer)
        timer.start()

    def _timer_callback(self, pitch: int, channel: int) -> None:
        try:
            self._send_note_off(pitch, channel)
        finally:
            timer_thread = current_thread()
            if isinstance(timer_thread, Timer):
                with self._timers_lock:
                    self._active_timers.discard(timer_thread)


class OscTransport(BaseTransport):
    """
    OSC transport for sending events to JUCE plugins or other OSC receivers.
    
    Sends scheduled events as OSC messages to a remote receiver (typically a
    JUCE plugin running in a DAW). Events are serialized as JSON and sent via
    OSC. Supports both `/daiw/notes` (direct) and `/daiw/result` (brain_server format).
    
    Any import errors are deferred until instantiation so the realtime module
    can be imported on systems without python-osc (e.g., Cloud CI).
    
    Args:
        host: OSC receiver host (default: "127.0.0.1")
        port: OSC receiver port (default: 9001)
        osc_address: OSC address pattern (default: "/daiw/notes")
        auto_reconnect: Attempt to reconnect on send failures (default: True)
        max_reconnect_attempts: Maximum reconnection attempts (default: 3)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9001,
        osc_address: str = "/daiw/notes",
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
    ) -> None:
        super().__init__()
        try:
            from pythonosc import udp_client
        except ImportError as exc:  # pragma: no cover - fallback path
            raise RuntimeError("OscTransport requires the 'python-osc' package") from exc

        self._udp_client = udp_client
        self._host = host
        self._port = port
        self._osc_address = osc_address
        self._auto_reconnect = auto_reconnect
        self._max_reconnect_attempts = max_reconnect_attempts
        self._client: Optional[udp_client.UDPClient] = None
        self._send_lock = Lock()
        self._connected = False
        self._reconnect_attempts = 0
        self._last_error: Optional[str] = None

    def emit(self, events: Iterable[ScheduledEvent]) -> None:
        """
        Emit scheduled events as OSC messages.
        
        Events are serialized as JSON and sent to the configured OSC address.
        Each event includes pitch, velocity, channel, start time, and duration.
        
        Args:
            events: Iterable of ScheduledEvent objects to send
        """
        events_list = list(events)
        if not events_list:
            return

        # Ensure connection
        if not self._connected:
            if not self._connect():
                return

        # Serialize events to JSON
        notes_data = []
        for scheduled in events_list:
            note = scheduled.note_event
            note_dict = {
                "pitch": note.pitch,
                "velocity": note.velocity,
                "channel": scheduled.channel,
                "start_tick": scheduled.start_tick,
                "duration_ticks": getattr(note, "duration_ticks", 0),
                "event_id": scheduled.event_id,
                "metadata": scheduled.metadata,
            }
            notes_data.append(note_dict)

        # Send as OSC message with JSON payload
        try:
            from pythonosc import osc_message_builder
            builder = osc_message_builder.OscMessageBuilder(self._osc_address)
            builder.add_string(json.dumps(notes_data))
            message = builder.build()
            
            with self._send_lock:
                if self._client:
                    self._client.send(message)
                    self._reconnect_attempts = 0  # Reset on success
                    self._last_error = None
        except Exception as exc:  # pragma: no cover - network errors
            self._last_error = str(exc)
            self._connected = False
            
            # Attempt reconnection if enabled
            if self._auto_reconnect and self._reconnect_attempts < self._max_reconnect_attempts:
                self._reconnect_attempts += 1
                self._connect()

    def close(self) -> None:
        """Close the OSC connection."""
        with self._send_lock:
            self._client = None
            self._connected = False
            self._reconnect_attempts = 0
            self._last_error = None

    def _connect(self) -> bool:
        """
        Establish OSC connection to the receiver.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self._connected and self._client:
            return True

        try:
            self._client = self._udp_client.UDPClient(self._host, self._port)
            self._connected = True
            self._reconnect_attempts = 0
            self._last_error = None
            return True
        except Exception as exc:  # pragma: no cover - network errors
            self._connected = False
            self._last_error = str(exc)
            return False

    @property
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message, if any."""
        return self._last_error
