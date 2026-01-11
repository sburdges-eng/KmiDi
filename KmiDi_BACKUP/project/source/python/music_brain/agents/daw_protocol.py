#!/usr/bin/env python3
"""
DAW Protocol - Abstract Interface for Multi-DAW Support.

This module defines the abstract protocol that all DAW bridges must implement,
enabling the UnifiedHub to work with different DAWs (Ableton, Logic Pro,
Reaper, Bitwig, etc.) through a unified interface.

Architecture:
    DAWProtocol (Abstract)
        ├── AbletonBridge     (OSC + MIDI - existing)
        ├── LogicProBridge    (AppleScript + MIDI)
        ├── ReaperBridge      (OSC + ReaScript)
        └── BitwigBridge      (OSC + MIDI)

Usage:
    from music_brain.agents.daw_protocol import get_daw_bridge, DAWType

    # Get bridge for specific DAW
    bridge = get_daw_bridge(DAWType.LOGIC_PRO)
    bridge.connect()
    bridge.play()

    # Or auto-detect
    bridge = get_daw_bridge()  # Auto-detects running DAW
"""

from __future__ import annotations

import abc
import atexit
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)


# =============================================================================
# DAW Types
# =============================================================================


class DAWType(Enum):
    """Supported DAW types."""

    ABLETON = "ableton"
    LOGIC_PRO = "logic_pro"
    REAPER = "reaper"
    BITWIG = "bitwig"
    STUDIO_ONE = "studio_one"
    FL_STUDIO = "fl_studio"
    PRO_TOOLS = "pro_tools"
    CUBASE = "cubase"
    GENERIC_OSC = "generic_osc"
    GENERIC_MIDI = "generic_midi"


# =============================================================================
# Common Data Types (DAW-agnostic)
# =============================================================================


@dataclass
class TransportState:
    """Current DAW transport state (DAW-agnostic)."""

    playing: bool = False
    recording: bool = False
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    position_bars: float = 0.0
    position_beats: float = 0.0
    loop_enabled: bool = False
    loop_start: float = 0.0
    loop_end: float = 4.0


@dataclass
class TrackInfo:
    """Information about a DAW track (DAW-agnostic)."""

    index: int
    name: str
    track_type: str = "audio"  # audio, midi, aux, master
    armed: bool = False
    muted: bool = False
    soloed: bool = False
    volume: float = 0.0  # dB
    pan: float = 0.0  # -1 to 1
    color: Optional[str] = None


@dataclass
class ClipInfo:
    """Information about a clip/region (DAW-agnostic)."""

    track_index: int
    clip_index: int
    name: str
    start_time: float = 0.0
    length: float = 4.0
    color: Optional[str] = None
    playing: bool = False


@dataclass
class DAWCapabilities:
    """Describes what features a DAW supports."""

    # Transport
    has_transport: bool = True
    has_tempo_control: bool = True
    has_loop_control: bool = True

    # Tracks
    can_create_tracks: bool = True
    can_arm_tracks: bool = True
    can_mute_solo: bool = True

    # Clips/Regions
    has_clip_launcher: bool = False  # Ableton/Bitwig style
    has_arrangement: bool = True

    # MIDI
    has_midi_output: bool = True
    has_midi_input: bool = True

    # OSC
    has_osc: bool = True

    # Automation
    has_automation: bool = True

    # Markers
    has_markers: bool = True

    # Plugin control
    can_control_plugins: bool = False

    # Custom
    custom_features: Dict[str, bool] = field(default_factory=dict)


# =============================================================================
# DAW Protocol (Abstract Interface)
# =============================================================================


@runtime_checkable
class DAWProtocol(Protocol):
    """
    Abstract protocol defining the DAW bridge interface.

    All DAW-specific bridges must implement this protocol to be usable
    with the UnifiedHub.
    """

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def daw_type(self) -> DAWType:
        """Return the DAW type this bridge connects to."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to the DAW."""
        ...

    @property
    def transport(self) -> TransportState:
        """Get current transport state."""
        ...

    @property
    def tracks(self) -> Dict[int, TrackInfo]:
        """Get all tracks."""
        ...

    @property
    def capabilities(self) -> DAWCapabilities:
        """Get DAW capabilities."""
        ...

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def connect(self) -> bool:
        """Connect to the DAW. Returns True on success."""
        ...

    def disconnect(self) -> None:
        """Disconnect from the DAW."""
        ...

    # =========================================================================
    # Transport Control
    # =========================================================================

    def play(self) -> None:
        """Start playback."""
        ...

    def stop(self) -> None:
        """Stop playback."""
        ...

    def record(self) -> None:
        """Start recording."""
        ...

    def pause(self) -> None:
        """Pause playback (if supported, otherwise stop)."""
        ...

    def set_tempo(self, bpm: float) -> None:
        """Set tempo in BPM."""
        ...

    def set_position(self, bars: float, beats: float = 0.0) -> None:
        """Set playhead position."""
        ...

    def set_loop(self, enabled: bool, start: float = 0.0, end: float = 4.0) -> None:
        """Set loop region."""
        ...

    # =========================================================================
    # Track Control
    # =========================================================================

    def create_track(self, track_type: str = "midi", name: str = "") -> int:
        """Create a new track. Returns track index."""
        ...

    def delete_track(self, index: int) -> bool:
        """Delete a track. Returns True on success."""
        ...

    def arm_track(self, index: int, armed: bool = True) -> None:
        """Arm/disarm a track for recording."""
        ...

    def mute_track(self, index: int, muted: bool = True) -> None:
        """Mute/unmute a track."""
        ...

    def solo_track(self, index: int, soloed: bool = True) -> None:
        """Solo/unsolo a track."""
        ...

    def set_track_volume(self, index: int, volume_db: float) -> None:
        """Set track volume in dB."""
        ...

    def set_track_pan(self, index: int, pan: float) -> None:
        """Set track pan (-1 to 1)."""
        ...

    def get_track_info(self, index: int) -> Optional[TrackInfo]:
        """Get info for a specific track."""
        ...

    # =========================================================================
    # MIDI Control
    # =========================================================================

    def send_note_on(self, note: int, velocity: int = 100, channel: int = 0) -> None:
        """Send MIDI note on."""
        ...

    def send_note_off(self, note: int, channel: int = 0) -> None:
        """Send MIDI note off."""
        ...

    def send_note(
        self,
        note: int,
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        """Send a note with duration."""
        ...

    def send_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        """Send multiple notes as a chord."""
        ...

    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        """Send MIDI Control Change."""
        ...

    def send_pitch_bend(self, value: int, channel: int = 0) -> None:
        """Send pitch bend (-8192 to 8191)."""
        ...

    def all_notes_off(self, channel: Optional[int] = None) -> None:
        """Send all notes off."""
        ...

    # =========================================================================
    # Voice Control (for voice synthesis)
    # =========================================================================

    def set_vowel(self, vowel: str, channel: int = 0) -> None:
        """Set vowel for voice synthesis (A, E, I, O, U)."""
        ...

    def set_breathiness(self, amount: float, channel: int = 0) -> None:
        """Set breathiness (0 to 1)."""
        ...

    def set_vibrato(self, rate: float, depth: float, channel: int = 0) -> None:
        """Set vibrato parameters."""
        ...

    # =========================================================================
    # Clips/Regions (if supported)
    # =========================================================================

    def fire_clip(self, track: int, clip: int) -> None:
        """Fire/trigger a clip."""
        ...

    def stop_clip(self, track: int, clip: int) -> None:
        """Stop a clip."""
        ...

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on(self, event: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for an event."""
        ...

    def off(self, event: str, callback: Optional[Callable] = None) -> None:
        """Remove callback(s) for an event."""
        ...

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "DAWProtocol":
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...


# =============================================================================
# Abstract Base Class (with default implementations)
# =============================================================================


class BaseDAWBridge(abc.ABC):
    """
    Abstract base class for DAW bridges.

    Provides default implementations for common functionality.
    Subclasses must implement the abstract methods.
    """

    def __init__(self):
        self._connected = False
        self._transport = TransportState()
        self._tracks: Dict[int, TrackInfo] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._capabilities = DAWCapabilities()

        atexit.register(self._shutdown)

    # =========================================================================
    # Abstract Methods (must implement)
    # =========================================================================

    @property
    @abc.abstractmethod
    def daw_type(self) -> DAWType:
        """Return the DAW type."""
        ...

    @abc.abstractmethod
    def _do_connect(self) -> bool:
        """Internal connect implementation."""
        ...

    @abc.abstractmethod
    def _do_disconnect(self) -> None:
        """Internal disconnect implementation."""
        ...

    # =========================================================================
    # Properties (with defaults)
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def transport(self) -> TransportState:
        return self._transport

    @property
    def tracks(self) -> Dict[int, TrackInfo]:
        return self._tracks.copy()

    @property
    def capabilities(self) -> DAWCapabilities:
        return self._capabilities

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def connect(self) -> bool:
        if self._connected:
            return True
        self._connected = self._do_connect()
        return self._connected

    def disconnect(self) -> None:
        if self._connected:
            self._do_disconnect()
            self._connected = False

    def _shutdown(self) -> None:
        """Atexit handler."""
        self.disconnect()

    # =========================================================================
    # Transport (defaults that can be overridden)
    # =========================================================================

    def pause(self) -> None:
        """Default pause = stop."""
        self.stop()

    def set_loop(self, enabled: bool, start: float = 0.0, end: float = 4.0) -> None:
        """Default: no-op (not all DAWs support remote loop control)."""
        pass

    # =========================================================================
    # Track Control (defaults)
    # =========================================================================

    def delete_track(self, index: int) -> bool:
        """Default: not supported."""
        return False

    def get_track_info(self, index: int) -> Optional[TrackInfo]:
        return self._tracks.get(index)

    # =========================================================================
    # Clips (defaults for non-clip-launcher DAWs)
    # =========================================================================

    def fire_clip(self, track: int, clip: int) -> None:
        """Default: no-op."""
        pass

    def stop_clip(self, track: int, clip: int) -> None:
        """Default: no-op."""
        pass

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on(self, event: str, callback: Callable[[Any], None]) -> None:
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Optional[Callable] = None) -> None:
        if callback:
            if event in self._callbacks and callback in self._callbacks[event]:
                self._callbacks[event].remove(callback)
        else:
            self._callbacks.pop(event, None)

    def _trigger_callback(self, event: str, data: Any) -> None:
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                print(f"[{self.daw_type.value}] Callback error for {event}: {e}")

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "BaseDAWBridge":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __del__(self):
        self._shutdown()


# =============================================================================
# DAW Registry
# =============================================================================


class DAWRegistry:
    """
    Registry for DAW bridge implementations.

    Allows dynamic registration and lookup of DAW bridges.
    """

    _bridges: Dict[DAWType, Type[BaseDAWBridge]] = {}
    _default: Optional[DAWType] = None

    @classmethod
    def register(cls, daw_type: DAWType, bridge_class: Type[BaseDAWBridge]) -> None:
        """Register a DAW bridge implementation."""
        cls._bridges[daw_type] = bridge_class

    @classmethod
    def get(cls, daw_type: DAWType) -> Optional[Type[BaseDAWBridge]]:
        """Get a DAW bridge class by type."""
        return cls._bridges.get(daw_type)

    @classmethod
    def create(cls, daw_type: DAWType, **kwargs) -> Optional[BaseDAWBridge]:
        """Create a DAW bridge instance."""
        bridge_class = cls._bridges.get(daw_type)
        if bridge_class:
            return bridge_class(**kwargs)
        return None

    @classmethod
    def set_default(cls, daw_type: DAWType) -> None:
        """Set the default DAW type."""
        cls._default = daw_type

    @classmethod
    def get_default(cls) -> Optional[DAWType]:
        """Get the default DAW type."""
        return cls._default

    @classmethod
    def available(cls) -> List[DAWType]:
        """Get list of registered DAW types."""
        return list(cls._bridges.keys())

    @classmethod
    def auto_detect(cls) -> Optional[DAWType]:
        """
        Attempt to auto-detect which DAW is running.

        Returns the first DAW type that successfully connects.
        """
        import subprocess
        import platform

        system = platform.system()

        # Check for running DAW processes
        daw_processes = {
            DAWType.ABLETON: ["Ableton Live", "Live"],
            DAWType.LOGIC_PRO: ["Logic Pro", "Logic Pro X"],
            DAWType.REAPER: ["REAPER", "reaper"],
            DAWType.BITWIG: ["Bitwig Studio", "bitwig-studio"],
            DAWType.STUDIO_ONE: ["Studio One"],
            DAWType.FL_STUDIO: ["FL Studio", "FL64"],
            DAWType.PRO_TOOLS: ["Pro Tools"],
            DAWType.CUBASE: ["Cubase"],
        }

        if system == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["osascript", "-e", 'tell application "System Events" to get name of every process'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                running = result.stdout.lower()

                for daw_type, names in daw_processes.items():
                    for name in names:
                        if name.lower() in running:
                            if daw_type in cls._bridges:
                                return daw_type
            except Exception:
                pass

        elif system == "Windows":
            try:
                result = subprocess.run(
                    ["tasklist", "/FO", "CSV"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                running = result.stdout.lower()

                for daw_type, names in daw_processes.items():
                    for name in names:
                        if name.lower() in running:
                            if daw_type in cls._bridges:
                                return daw_type
            except Exception:
                pass

        # Fallback to default or first available
        if cls._default and cls._default in cls._bridges:
            return cls._default

        if cls._bridges:
            return next(iter(cls._bridges.keys()))

        return None


# =============================================================================
# Factory Function
# =============================================================================


def get_daw_bridge(
    daw_type: Optional[DAWType] = None,
    auto_detect: bool = True,
    **kwargs,
) -> Optional[BaseDAWBridge]:
    """
    Get a DAW bridge instance.

    Args:
        daw_type: Specific DAW type to use
        auto_detect: If True and daw_type is None, attempt to auto-detect
        **kwargs: Additional arguments passed to bridge constructor

    Returns:
        DAW bridge instance or None if not available
    """
    if daw_type is None:
        if auto_detect:
            daw_type = DAWRegistry.auto_detect()
        else:
            daw_type = DAWRegistry.get_default()

    if daw_type is None:
        return None

    return DAWRegistry.create(daw_type, **kwargs)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Types
    "DAWType",
    "TransportState",
    "TrackInfo",
    "ClipInfo",
    "DAWCapabilities",
    # Protocol
    "DAWProtocol",
    "BaseDAWBridge",
    # Registry
    "DAWRegistry",
    # Factory
    "get_daw_bridge",
]

