#!/usr/bin/env python3
"""
Async UnifiedHub - Event-Driven Architecture for DAiW Music Brain.

This is the async-first, reactive version of UnifiedHub. It provides:
- Non-blocking DAW/LLM communication via asyncio
- Reactive state with automatic UI propagation
- WebSocket API for real-time remote control
- Event-driven architecture for decoupled components

Usage:
    from music_brain.agents import AsyncUnifiedHub

    async def main():
        async with AsyncUnifiedHub() as hub:
            await hub.connect_daw()
            await hub.speak("Hello world")
            await hub.play()

            # Subscribe to state changes
            hub.state["voice"].subscribe(lambda o, n: print(f"Voice: {n}"))

            # Or use the WebSocket API from React:
            # ws = new WebSocket("ws://localhost:8765")
            # ws.send(JSON.stringify({type: "request", method: "play", id: 1}))

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ableton_bridge import (
    AbletonBridge,
    MIDIConfig,
    OSCConfig,
    TransportState,
    VoiceCC,
    VOWEL_FORMANTS,
)
from .crewai_music_agents import (
    AGENT_ROLES,
    LLMBackend,
    LocalLLM,
    LocalLLMConfig,
    MusicCrew,
    OnnxLLM,
    OnnxLLMConfig,
    ToolManager,
)
from .daw_protocol import (
    DAWType,
    DAWRegistry,
    BaseDAWBridge,
    get_daw_bridge,
)
from .events import Event, EventBus, EventPriority
from .reactive import ComputedValue, Observable, StateStore
from .unified_hub import (
    DAWState,
    HubConfig,
    LocalVoiceSynth,
    SessionConfig,
    VoiceState,
)

# Import DAW bridges to trigger auto-registration
from . import daw_bridges  # noqa: F401

# Optional WebSocket support
try:
    from .websocket_api import WebSocketServer, WEBSOCKETS_AVAILABLE
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServer = None  # type: ignore


# =============================================================================
# Async UnifiedHub
# =============================================================================


class AsyncUnifiedHub:
    """
    Async-first UnifiedHub with reactive state and event-driven architecture.

    This hub runs an asyncio event loop and provides:
    - Non-blocking operations for all DAW/LLM interactions
    - Observable state that auto-notifies UI on changes
    - Event bus for decoupled pub/sub communication
    - Optional WebSocket server for remote control

    State is accessed via the reactive store:
        hub.state["voice"].value  # Current VoiceState
        hub.state["daw"].update(tempo=120)  # Updates + notifies

    Events are emitted automatically on state changes:
        hub.events.on("voice.changed")(handler)
        hub.events.on("daw.play")(handler)
    """

    def __init__(
        self,
        config: Optional[HubConfig] = None,
        daw_type: Optional[DAWType] = None,
        enable_websocket: bool = True,
        websocket_port: int = 8765,
    ):
        self.config = config or HubConfig()
        self._daw_type = daw_type  # None = use config or auto-detect
        self._enable_websocket = enable_websocket and WEBSOCKETS_AVAILABLE
        self._websocket_port = websocket_port

        # =====================================================================
        # Reactive State Store
        # =====================================================================
        self.state = StateStore()

        # Register observable states
        self._session_state = Observable(SessionConfig(), name="session")
        self._voice_state = Observable(VoiceState(), name="voice")
        self._daw_state = Observable(DAWState(), name="daw")

        self.state.register("session", self._session_state)
        self.state.register("voice", self._voice_state)
        self.state.register("daw", self._daw_state)

        # Computed values
        self.is_active = ComputedValue(
            lambda: self._voice_state.value.active or self._daw_state.value.playing,
            [self._voice_state, self._daw_state],
            name="is_active",
        )

        # =====================================================================
        # Event Bus
        # =====================================================================
        self.events = EventBus()

        # Wire state changes to events
        self._session_state.subscribe_async(self._on_session_change)
        self._voice_state.subscribe_async(self._on_voice_change)
        self._daw_state.subscribe_async(self._on_daw_change)

        # =====================================================================
        # Components
        # =====================================================================
        self._daw: Optional[BaseDAWBridge] = None
        self._bridge: Optional[AbletonBridge] = None  # Legacy compatibility
        self._voice: Optional[LocalVoiceSynth] = None
        self._crew: Optional[MusicCrew] = None
        self._llm: Optional[LocalLLM] = None
        self._ws_server: Optional[WebSocketServer] = None

        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hub_")

        # State
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Ensure directories exist
        os.makedirs(self.config.session_dir, exist_ok=True)
        os.makedirs(self.config.config_dir, exist_ok=True)

        # Cleanup on exit
        atexit.register(self._sync_shutdown)

    # =========================================================================
    # State Change Handlers (emit events)
    # =========================================================================

    async def _on_session_change(self, old: SessionConfig, new: SessionConfig) -> None:
        """Emit event on session state change."""
        await self.events.emit("session.changed", asdict(new))
        if self._ws_server:
            await self._ws_server.broadcast("session.changed", asdict(new), "state")

    async def _on_voice_change(self, old: VoiceState, new: VoiceState) -> None:
        """Emit event on voice state change."""
        await self.events.emit("voice.changed", asdict(new))
        if self._ws_server:
            await self._ws_server.broadcast("voice.changed", asdict(new), "state")

    async def _on_daw_change(self, old: DAWState, new: DAWState) -> None:
        """Emit event on DAW state change."""
        await self.events.emit("daw.changed", asdict(new))
        if self._ws_server:
            await self._ws_server.broadcast("daw.changed", asdict(new), "state")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> "AsyncUnifiedHub":
        """Start the hub and all components."""
        self._running = True
        self._loop = asyncio.get_running_loop()

        # Initialize LLM (blocking operation, run in executor)
        backend = (
            LLMBackend.ONNX_HTTP
            if self.config.llm_backend.lower() in ["onnx", "onnx_http"]
            else LLMBackend.OLLAMA
        )

        await self._loop.run_in_executor(
            self._executor,
            self._init_llm,
            backend,
        )

        # Initialize bridge (blocking)
        await self._loop.run_in_executor(self._executor, self._init_bridge)

        # Initialize voice (needs MIDI bridge for formant control)
        midi_bridge = None
        if self._bridge and hasattr(self._bridge, 'midi'):
            midi_bridge = self._bridge.midi
        self._voice = LocalVoiceSynth(midi_bridge)

        # Initialize crew (blocking)
        await self._loop.run_in_executor(
            self._executor,
            self._init_crew,
            backend,
        )

        # Start WebSocket server
        if self._enable_websocket and WebSocketServer:
            self._ws_server = WebSocketServer(self)
            await self._ws_server.start(port=self._websocket_port)

        await self.events.emit("hub.started", {"timestamp": datetime.now().isoformat()})
        return self

    def _init_llm(self, backend: LLMBackend) -> None:
        """Initialize LLM (runs in executor)."""
        if backend == LLMBackend.ONNX_HTTP:
            self._llm = OnnxLLM(OnnxLLMConfig(base_url=self.config.llm_onnx_url))
        else:
            self._llm = LocalLLM(
                LocalLLMConfig(
                    model=self.config.llm_model,
                    base_url=self.config.llm_url,
                )
            )

    def _init_bridge(self) -> None:
        """Initialize DAW bridge (runs in executor)."""
        # Determine DAW type
        if self._daw_type:
            daw_type = self._daw_type
        elif hasattr(self.config, 'daw_type') and self.config.daw_type:
            # Map string to DAWType enum
            daw_type_map = {
                "ableton": DAWType.ABLETON,
                "logic_pro": DAWType.LOGIC_PRO,
                "reaper": DAWType.REAPER,
                "bitwig": DAWType.BITWIG,
            }
            daw_type = daw_type_map.get(self.config.daw_type.lower(), DAWType.ABLETON)
        else:
            daw_type = None  # Auto-detect

        # Create DAW bridge using the protocol abstraction
        self._daw = get_daw_bridge(daw_type, auto_detect=True)

        if self._daw is None:
            # Fallback to direct Ableton bridge for compatibility
            self._bridge = AbletonBridge(
                osc_config=OSCConfig(
                    host=self.config.osc_host,
                    send_port=self.config.osc_send_port,
                    receive_port=self.config.osc_receive_port,
                ),
                midi_config=MIDIConfig(
                    output_port=self.config.midi_port,
                    virtual=True,
                ),
            )
        else:
            # For Ableton, also keep legacy bridge reference for voice synth
            if self._daw.daw_type == DAWType.ABLETON:
                from .daw_bridges import AbletonDAWBridge
                if isinstance(self._daw, AbletonDAWBridge):
                    self._bridge = self._daw._osc_bridge  # type: ignore

    def _init_crew(self, backend: LLMBackend) -> None:
        """Initialize music crew (runs in executor)."""
        if backend == LLMBackend.ONNX_HTTP:
            self._crew = MusicCrew(
                llm_backend=backend,
                onnx_config=OnnxLLMConfig(base_url=self.config.llm_onnx_url),
            )
        else:
            self._crew = MusicCrew(
                LocalLLMConfig(
                    model=self.config.llm_model,
                    base_url=self.config.llm_url,
                )
            )
        if self._bridge:
            self._crew.setup(self._bridge)

    async def stop(self) -> None:
        """Stop the hub gracefully."""
        self._running = False

        await self.events.emit("hub.stopping", {})

        # Stop any active notes
        if self._voice_state.value.active:
            await self.note_off()

        # Stop WebSocket server
        if self._ws_server:
            await self._ws_server.stop()
            self._ws_server = None

        # Shutdown components in executor
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._shutdown_components,
        )

        self.events.shutdown()
        self._executor.shutdown(wait=False)

        await self.events.emit("hub.stopped", {})

    def _shutdown_components(self) -> None:
        """Shutdown components (runs in executor)."""
        if self._crew:
            self._crew.shutdown()
            self._crew = None

        if self._daw:
            self._daw.disconnect()
            self._daw = None

        if self._bridge:
            self._bridge.disconnect()
            self._bridge = None

        self._voice = None

    def _sync_shutdown(self) -> None:
        """Synchronous shutdown for atexit."""
        if self._running:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.stop())
                else:
                    loop.run_until_complete(self.stop())
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._running

    # =========================================================================
    # DAW Control (Async) - Multi-DAW Support
    # =========================================================================

    def _get_daw(self):
        """Get the active DAW bridge (protocol or legacy)."""
        return self._daw or self._bridge

    async def connect_daw(self) -> bool:
        """Connect to the configured DAW."""
        daw = self._get_daw()
        if not daw:
            return False

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            self._executor,
            daw.connect,
        )

        daw_name = daw.daw_type.value if hasattr(daw, 'daw_type') else "ableton"
        self._daw_state.update(connected=success)
        await self.events.emit("daw.connected", {"success": success, "daw": daw_name})
        return success

    async def disconnect_daw(self) -> None:
        """Disconnect from DAW."""
        daw = self._get_daw()
        if daw:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                daw.disconnect,
            )
            self._daw_state.update(connected=False)

    async def play(self) -> None:
        """Start DAW playback."""
        daw = self._get_daw()
        if daw:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                daw.play,
            )
            self._daw_state.update(playing=True)
            await self.events.emit("daw.play", {})

    async def stop_playback(self) -> None:
        """Stop DAW playback."""
        daw = self._get_daw()
        if daw:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                daw.stop,
            )
            self._daw_state.update(playing=False)
            await self.events.emit("daw.stop", {})

    async def record(self) -> None:
        """Start DAW recording."""
        daw = self._get_daw()
        if daw:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                daw.record,
            )
            self._daw_state.update(recording=True)
            await self.events.emit("daw.record", {})

    async def set_tempo(self, bpm: float) -> None:
        """Set DAW tempo."""
        daw = self._get_daw()
        if daw:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                daw.set_tempo,
                bpm,
            )
            self._daw_state.update(tempo=bpm)
            self._session_state.update(tempo=bpm)

    async def send_note(
        self,
        note: int,
        velocity: int = 100,
        duration_ms: int = 500,
    ) -> None:
        """Send a MIDI note to DAW."""
        daw = self._get_daw()
        if daw:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                daw.send_note,
                note,
                velocity,
                duration_ms,
            )

    async def send_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_ms: int = 500,
    ) -> None:
        """Send a chord to DAW."""
        daw = self._get_daw()
        if daw:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                daw.send_chord,
                notes,
                velocity,
                duration_ms,
            )

    # =========================================================================
    # Voice Control (Async)
    # =========================================================================

    async def speak(
        self,
        text: str,
        vowel: Optional[str] = None,
        rate: int = 175,
    ) -> None:
        """Speak text using local TTS."""
        if self._voice:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._voice.speak,
                text,
                vowel,
                rate,
            )
            await self.events.emit("voice.speak", {"text": text})

    async def note_on(
        self,
        pitch: int,
        velocity: int = 100,
        channel: Optional[int] = None,
    ) -> None:
        """Start a voice note."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if self._voice:
            self._voice.note_on(pitch, velocity, ch)
            self._voice_state.update(
                pitch=pitch,
                velocity=velocity,
                active=True,
            )
            await self.events.emit(
                "voice.note_on",
                {"pitch": pitch, "velocity": velocity},
            )

    async def note_off(
        self,
        pitch: Optional[int] = None,
        channel: Optional[int] = None,
    ) -> None:
        """Stop a voice note."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if self._voice:
            self._voice.note_off(pitch, ch)
            self._voice_state.update(active=False)
            await self.events.emit("voice.note_off", {"pitch": pitch})

    async def set_vowel(
        self,
        vowel: str,
        channel: Optional[int] = None,
    ) -> None:
        """Set voice vowel (A, E, I, O, U)."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if self._voice:
            self._voice.set_vowel(vowel, ch)
            self._voice_state.update(vowel=vowel.upper())

    async def set_breathiness(
        self,
        amount: float,
        channel: Optional[int] = None,
    ) -> None:
        """Set voice breathiness (0-1)."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if self._voice:
            self._voice.set_breathiness(amount, ch)
            self._voice_state.update(breathiness=amount)

    async def set_vibrato(
        self,
        rate: float,
        depth: float,
        channel: Optional[int] = None,
    ) -> None:
        """Set voice vibrato."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if self._voice:
            self._voice.set_vibrato(rate, depth, ch)
            self._voice_state.update(
                vibrato_rate=rate,
                vibrato_depth=depth,
            )

    # =========================================================================
    # AI Agents (Async)
    # =========================================================================

    async def ask_agent(self, role_id: str, task: str) -> str:
        """Ask a specific AI agent about a task."""
        if not self._crew:
            return "AI agents not initialized"

        response = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._crew.ask,
            role_id,
            task,
        )

        await self.events.emit(
            "agent.response",
            {"role": role_id, "task": task, "response": response},
        )
        return response

    async def produce(self, brief: str) -> Dict[str, str]:
        """Have the Producer coordinate a production task."""
        if not self._crew:
            return {"error": "AI agents not initialized"}

        result = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._crew.produce,
            brief,
        )
        return result

    async def suggest_progression(
        self,
        emotion: str,
        key: str = "C",
    ) -> str:
        """Suggest a chord progression for an emotion."""
        return await self.ask_agent(
            "composer",
            f"Suggest a chord progression in {key} for the emotion: {emotion}\n"
            f"Include modal interchange if appropriate and explain the emotional effect.",
        )

    @property
    def llm_available(self) -> bool:
        """Check if local LLM is available."""
        return self._llm is not None and self._llm.is_available

    # =========================================================================
    # Session Management (Async)
    # =========================================================================

    async def new_session(self, name: str = "untitled") -> None:
        """Create a new session."""
        self._session_state.set(SessionConfig(name=name))
        self._voice_state.set(VoiceState())
        await self.events.emit("session.new", {"name": name})

    async def save_session(self, name: Optional[str] = None) -> str:
        """Save current session to file."""
        session = self._session_state.value
        if name:
            self._session_state.update(name=name)
            session = self._session_state.value

        self._session_state.update(updated_at=datetime.now().isoformat())

        data = {
            "session": asdict(self._session_state.value),
            "voice_state": asdict(self._voice_state.value),
            "daw_state": asdict(self._daw_state.value),
        }

        filename = f"{session.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.config.session_dir, filename)

        async def write_file():
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: write_file,
        )

        # Actually write synchronously in executor
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        await self.events.emit("session.saved", {"path": filepath})
        return filepath

    async def load_session(self, filepath: str) -> bool:
        """Load a session from file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self._session_state.set(SessionConfig(**data.get("session", {})))
            self._voice_state.set(VoiceState(**data.get("voice_state", {})))

            # Apply voice state
            if self._voice:
                voice = self._voice_state.value
                await self.set_vowel(voice.vowel)
                await self.set_breathiness(voice.breathiness)
                await self.set_vibrato(voice.vibrato_rate, voice.vibrato_depth)

            # Apply DAW state
            if self._bridge and self._daw_state.value.connected:
                await self.set_tempo(self._session_state.value.tempo)

            await self.events.emit("session.loaded", {"path": filepath})
            return True

        except Exception as e:
            print(f"Error loading session: {e}")
            return False

    def list_sessions(self) -> List[str]:
        """List available session files."""
        session_dir = Path(self.config.session_dir)
        return [f.name for f in session_dir.glob("*.json")]

    # =========================================================================
    # Health & Status
    # =========================================================================

    def check_health(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "hub_running": self._running,
            "llm": {
                "available": self.llm_available,
                "backend": self.config.llm_backend,
            },
            "daw": {
                "connected": self._daw_state.value.connected,
                "playing": self._daw_state.value.playing,
            },
            "websocket": {
                "enabled": self._enable_websocket,
                "running": self._ws_server.is_running if self._ws_server else False,
                "clients": self._ws_server.client_count if self._ws_server else 0,
            },
            "state_versions": {
                "session": self._session_state.version,
                "voice": self._voice_state.version,
                "daw": self._daw_state.version,
            },
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        return self.state.to_dict()

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "AsyncUnifiedHub":
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


# =============================================================================
# Convenience Functions
# =============================================================================

_default_async_hub: Optional[AsyncUnifiedHub] = None


async def get_async_hub() -> AsyncUnifiedHub:
    """Get or create the default async hub."""
    global _default_async_hub
    if _default_async_hub is None:
        _default_async_hub = AsyncUnifiedHub()
        await _default_async_hub.start()
    return _default_async_hub


async def stop_async_hub() -> None:
    """Stop the default async hub."""
    global _default_async_hub
    if _default_async_hub:
        await _default_async_hub.stop()
        _default_async_hub = None


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "AsyncUnifiedHub",
    "get_async_hub",
    "stop_async_hub",
]

