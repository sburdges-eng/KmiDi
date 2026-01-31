"""
DAiW Unified AI Agent Hub

Central orchestration hub that combines all DAiW capabilities:
- Voice synthesis (formant, neural, wavetable)
- DAW control (Ableton, REAPER via OSC/MIDI)
- Audio processing (Pedalboard, DawDreamer, JACK)
- AI agents (CrewAI music production, MCP tools)

This hub provides:
1. Unified API across all subsystems
2. Session management and state persistence
3. Real-time audio routing
4. Agent orchestration
5. MCP server integration for Claude/AI assistants
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum
import json
import asyncio
import threading
import queue
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DAiWHub")


class HubState(Enum):
    """Hub operational state"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class AudioRoutingMode(Enum):
    """Audio routing configuration"""
    INTERNAL = "internal"           # All processing in Python/C++
    DAW_ROUTED = "daw_routed"       # Route through DAW
    HYBRID = "hybrid"               # Mixed routing
    STREAMING = "streaming"          # Stream to external (WebSocket)


@dataclass
class SessionConfig:
    """Configuration for a DAiW session"""
    name: str = "Untitled Session"
    sample_rate: int = 44100
    buffer_size: int = 512
    tempo: float = 120.0
    time_signature: tuple = (4, 4)
    routing_mode: AudioRoutingMode = AudioRoutingMode.INTERNAL

    # Voice settings
    default_voice: str = "default"
    voice_formant_mix: float = 0.8
    voice_breathiness: float = 0.2

    # DAW settings
    daw_host: str = "127.0.0.1"
    daw_osc_port: int = 11000
    daw_midi_port: Optional[str] = None

    # MCP settings
    mcp_enabled: bool = True
    mcp_port: int = 3000

    # Agent settings
    enable_agents: bool = True
    agent_model: str = "gpt-4"


@dataclass
class VoiceState:
    """Current voice synthesis state"""
    active: bool = False
    current_vowel: int = 0  # 0=A, 1=E, 2=I, 3=O, 4=U
    pitch_hz: float = 220.0
    formant_shift: float = 0.0
    breathiness: float = 0.2
    vibrato_rate: float = 5.5
    vibrato_depth: float = 0.3
    active_notes: List[int] = field(default_factory=list)
    current_voice: str = "default"


@dataclass
class DAWState:
    """Current DAW state"""
    connected: bool = False
    playing: bool = False
    recording: bool = False
    tempo: float = 120.0
    position_beats: float = 0.0
    track_count: int = 0
    armed_tracks: List[int] = field(default_factory=list)


class UnifiedHub:
    """
    Main hub for DAiW music production system.

    Provides centralized access to all subsystems:
    - Voice synthesis pipeline
    - DAW integration (Ableton/REAPER)
    - Audio processing chain
    - AI agent orchestration
    - MCP server for external AI access

    Example:
        hub = UnifiedHub()
        hub.start()

        # Voice synthesis
        hub.speak("Hello world")
        hub.set_vowel("A")
        hub.note_on(60, 100)

        # DAW control
        hub.connect_daw()
        hub.daw_play()
        hub.create_voice_track("Lead Vocal")

        # Agent tasks
        result = await hub.run_agent_task(
            "produce a vocal track saying 'hello'",
            style="ethereal"
        )

        hub.stop()
    """

    def __init__(self, config: Optional[SessionConfig] = None):
        """
        Initialize the unified hub.

        Args:
            config: Session configuration
        """
        self.config = config or SessionConfig()
        self._state = HubState.STOPPED

        # Subsystem references
        self._voice_pipeline = None
        self._ableton_bridge = None
        self._audio_processor = None
        self._mcp_server = None
        self._agent_crew = None

        # State tracking
        self._voice_state = VoiceState()
        self._daw_state = DAWState()

        # Event queues
        self._command_queue: queue.Queue = queue.Queue()
        self._response_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {}

        # Thread management
        self._main_thread: Optional[threading.Thread] = None
        self._running = False

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def start(self, blocking: bool = False) -> bool:
        """
        Start the hub and all subsystems.

        Args:
            blocking: If True, block until hub is stopped

        Returns:
            True if started successfully
        """
        if self._state == HubState.RUNNING:
            logger.warning("Hub already running")
            return True

        self._state = HubState.STARTING
        logger.info("Starting DAiW Unified Hub...")

        try:
            # Initialize subsystems
            self._init_voice_pipeline()
            self._init_audio_processor()

            if self.config.enable_agents:
                self._init_agents()

            self._running = True
            self._state = HubState.RUNNING

            logger.info("DAiW Hub started successfully")

            if blocking:
                self._run_main_loop()

            return True

        except Exception as e:
            logger.error(f"Failed to start hub: {e}")
            self._state = HubState.ERROR
            return False

    def stop(self):
        """Stop the hub and all subsystems."""
        self._shutdown()

    def _shutdown(self):
        """Internal shutdown - clean up all resources."""
        if self._state == HubState.STOPPED:
            return

        logger.info("Stopping DAiW Hub...")
        self._running = False

        # Stop all active notes
        for note in self._voice_state.active_notes.copy():
            try:
                self.note_off(note)
            except Exception:
                pass
        self._voice_state.active_notes.clear()
        self._voice_state.active = False

        # Disconnect from DAW
        if self._ableton_bridge:
            try:
                self._ableton_bridge.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting Ableton: {e}")
            self._ableton_bridge = None

        # Stop MCP server
        if self._mcp_server:
            try:
                # Would stop the server
                pass
            except Exception as e:
                logger.error(f"Error stopping MCP server: {e}")
            self._mcp_server = None

        # Clear audio processor
        if self._audio_processor:
            try:
                if hasattr(self._audio_processor, 'clear'):
                    self._audio_processor.clear()
            except Exception:
                pass
            self._audio_processor = None

        # Clear voice pipeline
        self._voice_pipeline = None

        # Clear agent crew
        self._agent_crew = None

        # Clear callbacks
        self._callbacks.clear()

        # Clear command queues
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break

        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for main thread to finish
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=2.0)
        self._main_thread = None

        self._state = HubState.STOPPED
        logger.info("DAiW Hub stopped")

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        self._shutdown()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures stop."""
        self.stop()
        return False

    def is_running(self) -> bool:
        """Check if hub is running."""
        return self._state == HubState.RUNNING and self._running

    def force_stop(self):
        """Force immediate stop without waiting."""
        self._running = False
        self._state = HubState.STOPPED
        logger.warning("DAiW Hub force stopped")

    def _run_main_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                # Process command queue
                while not self._command_queue.empty():
                    cmd = self._command_queue.get_nowait()
                    self._process_command(cmd)

                # Small sleep to prevent CPU spinning
                time.sleep(0.001)

            except KeyboardInterrupt:
                self.stop()
                break

    def _process_command(self, command: Dict[str, Any]):
        """Process a queued command."""
        cmd_type = command.get("type")
        params = command.get("params", {})

        handlers = {
            "voice_speak": lambda: self.speak(**params),
            "voice_note_on": lambda: self.note_on(**params),
            "voice_note_off": lambda: self.note_off(**params),
            "voice_set_vowel": lambda: self.set_vowel(**params),
            "daw_play": lambda: self.daw_play(),
            "daw_stop": lambda: self.daw_stop(),
        }

        handler = handlers.get(cmd_type)
        if handler:
            result = handler()
            self._response_queue.put({"command": cmd_type, "result": result})

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_voice_pipeline(self):
        """Initialize voice synthesis pipeline."""
        try:
            from .daiw_mcp_server import voice_pipeline
            self._voice_pipeline = voice_pipeline
            logger.info("Voice pipeline initialized")
        except ImportError as e:
            logger.warning(f"Voice pipeline not available: {e}")

    def _init_audio_processor(self):
        """Initialize audio processor."""
        try:
            from ..audio.framework_integrations import UnifiedAudioProcessor
            self._audio_processor = UnifiedAudioProcessor()
            logger.info("Audio processor initialized")
        except ImportError as e:
            logger.warning(f"Audio processor not available: {e}")

    def _init_agents(self):
        """Initialize AI agent crew."""
        try:
            from .crewai_music_agents import create_music_production_crew
            self._agent_crew = create_music_production_crew()
            logger.info("Agent crew initialized")
        except ImportError as e:
            logger.warning(f"Agent crew not available: {e}")

    # =========================================================================
    # Voice Synthesis
    # =========================================================================

    def speak(self, text: str, voice_name: Optional[str] = None) -> bool:
        """
        Synthesize text to speech.

        Args:
            text: Text to speak
            voice_name: Voice preset (uses default if None)

        Returns:
            True if successful
        """
        if not self._voice_pipeline:
            logger.error("Voice pipeline not available")
            return False

        voice = voice_name or self._voice_state.current_voice
        self._voice_pipeline.speak(text, voice)
        self._trigger_callbacks("voice_speak", {"text": text, "voice": voice})
        return True

    def note_on(self, note: int, velocity: int = 100) -> bool:
        """
        Trigger a note on event.

        Args:
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)

        Returns:
            True if successful
        """
        if note not in self._voice_state.active_notes:
            self._voice_state.active_notes.append(note)

        self._voice_state.active = True
        self._voice_state.pitch_hz = 440.0 * (2 ** ((note - 69) / 12))

        if self._voice_pipeline:
            self._voice_pipeline.note_on(note, velocity)

        self._trigger_callbacks("voice_note_on", {"note": note, "velocity": velocity})
        return True

    def note_off(self, note: int) -> bool:
        """
        Trigger a note off event.

        Args:
            note: MIDI note number

        Returns:
            True if successful
        """
        if note in self._voice_state.active_notes:
            self._voice_state.active_notes.remove(note)

        if not self._voice_state.active_notes:
            self._voice_state.active = False

        if self._voice_pipeline:
            self._voice_pipeline.note_off(note)

        self._trigger_callbacks("voice_note_off", {"note": note})
        return True

    def set_vowel(self, vowel: Union[str, int]) -> bool:
        """
        Set the current vowel sound.

        Args:
            vowel: Vowel (A/E/I/O/U or 0-4)

        Returns:
            True if successful
        """
        vowel_map = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

        if isinstance(vowel, str):
            vowel_idx = vowel_map.get(vowel.lower(), 0)
        else:
            vowel_idx = int(vowel) % 5

        self._voice_state.current_vowel = vowel_idx

        if self._voice_pipeline:
            self._voice_pipeline.set_vowel(vowel_idx)

        self._trigger_callbacks("voice_vowel", {"vowel": vowel_idx})
        return True

    def set_pitch(self, pitch_hz: float) -> bool:
        """Set voice pitch in Hz."""
        self._voice_state.pitch_hz = pitch_hz

        if self._voice_pipeline:
            self._voice_pipeline.set_pitch(pitch_hz)

        return True

    def set_formant_shift(self, shift: float) -> bool:
        """Set formant shift (-1.0 to 1.0)."""
        self._voice_state.formant_shift = max(-1.0, min(1.0, shift))

        if self._voice_pipeline:
            self._voice_pipeline.formant_shift(shift)

        return True

    def set_breathiness(self, breathiness: float) -> bool:
        """Set breathiness (0.0 to 1.0)."""
        self._voice_state.breathiness = max(0.0, min(1.0, breathiness))

        if self._voice_pipeline:
            self._voice_pipeline.set_breathiness(breathiness)

        return True

    def set_vibrato(self, rate: float, depth: float) -> bool:
        """Set vibrato parameters."""
        self._voice_state.vibrato_rate = rate
        self._voice_state.vibrato_depth = depth

        if self._voice_pipeline:
            self._voice_pipeline.set_vibrato(rate, depth)

        return True

    def train_voice(self, audio_file: str, voice_name: str) -> bool:
        """
        Train a new voice from audio file.

        Args:
            audio_file: Path to training audio
            voice_name: Name for the new voice

        Returns:
            True if successful
        """
        if not self._voice_pipeline:
            return False

        model = self._voice_pipeline.train_voice(audio_file, voice_name)
        return model is not None

    def list_voices(self) -> List[str]:
        """Get list of available voice presets."""
        if self._voice_pipeline:
            return self._voice_pipeline.list_voices()
        return ["default"]

    def get_voice_state(self) -> Dict[str, Any]:
        """Get current voice state."""
        return {
            "active": self._voice_state.active,
            "vowel": self._voice_state.current_vowel,
            "pitch_hz": self._voice_state.pitch_hz,
            "formant_shift": self._voice_state.formant_shift,
            "breathiness": self._voice_state.breathiness,
            "vibrato_rate": self._voice_state.vibrato_rate,
            "vibrato_depth": self._voice_state.vibrato_depth,
            "active_notes": self._voice_state.active_notes.copy(),
            "current_voice": self._voice_state.current_voice
        }

    # =========================================================================
    # DAW Control
    # =========================================================================

    def connect_daw(self, host: Optional[str] = None) -> bool:
        """
        Connect to DAW (Ableton Live).

        Args:
            host: DAW host address

        Returns:
            True if connected
        """
        try:
            from .ableton_bridge import DAiWAbletonIntegration

            self._ableton_bridge = DAiWAbletonIntegration(
                voice_pipeline=self._voice_pipeline
            )

            if host:
                self._ableton_bridge.osc_bridge.host = host

            success = self._ableton_bridge.connect()
            self._daw_state.connected = success

            if success:
                self._sync_daw_state()
                logger.info("Connected to DAW")
            else:
                logger.warning("Failed to connect to DAW")

            return success

        except ImportError as e:
            logger.error(f"Ableton bridge not available: {e}")
            return False

    def disconnect_daw(self):
        """Disconnect from DAW."""
        if self._ableton_bridge:
            self._ableton_bridge.disconnect()
            self._daw_state.connected = False

    def _sync_daw_state(self):
        """Sync state from DAW."""
        if not self._ableton_bridge:
            return

        info = self._ableton_bridge.get_session_info()
        self._daw_state.tempo = info.get("tempo", 120.0)
        self._daw_state.playing = info.get("playing", False)
        self._daw_state.recording = info.get("recording", False)
        self._daw_state.track_count = len(info.get("tracks", []))

    def daw_play(self) -> bool:
        """Start DAW playback."""
        if not self._ableton_bridge:
            return False

        self._ableton_bridge.osc_bridge.play()
        self._daw_state.playing = True
        return True

    def daw_stop(self) -> bool:
        """Stop DAW playback."""
        if not self._ableton_bridge:
            return False

        self._ableton_bridge.osc_bridge.stop()
        self._daw_state.playing = False
        return True

    def daw_record(self) -> bool:
        """Start DAW recording."""
        if not self._ableton_bridge:
            return False

        self._ableton_bridge.osc_bridge.record()
        self._daw_state.recording = True
        return True

    def set_tempo(self, bpm: float) -> bool:
        """Set session tempo."""
        self.config.tempo = bpm

        if self._ableton_bridge:
            self._ableton_bridge.osc_bridge.set_tempo(bpm)

        self._daw_state.tempo = bpm
        return True

    def create_voice_track(self, name: str = "DAiW Voice") -> int:
        """
        Create a new voice track in the DAW.

        Args:
            name: Track name

        Returns:
            Track index
        """
        if not self._ableton_bridge:
            return -1

        return self._ableton_bridge.create_voice_track(name)

    def render_voice_to_track(self, text: str, track_idx: int,
                              start_beat: float = 0) -> bool:
        """
        Render synthesized voice to a DAW track.

        Args:
            text: Text to synthesize
            track_idx: Target track
            start_beat: Start position in beats

        Returns:
            True if successful
        """
        if not self._ableton_bridge:
            return False

        return self._ableton_bridge.render_text_to_track(
            text, track_idx, start_beat,
            voice_name=self._voice_state.current_voice
        )

    def fire_clip(self, track: int, clip: int) -> bool:
        """Fire a clip in the DAW."""
        if not self._ableton_bridge:
            return False

        self._ableton_bridge.osc_bridge.fire_clip(track, clip)
        return True

    def get_daw_state(self) -> Dict[str, Any]:
        """Get current DAW state."""
        return {
            "connected": self._daw_state.connected,
            "playing": self._daw_state.playing,
            "recording": self._daw_state.recording,
            "tempo": self._daw_state.tempo,
            "position_beats": self._daw_state.position_beats,
            "track_count": self._daw_state.track_count,
            "armed_tracks": self._daw_state.armed_tracks.copy()
        }

    # =========================================================================
    # Audio Processing
    # =========================================================================

    def apply_effect(self, effect_type: str, **params) -> bool:
        """
        Apply an audio effect to the voice output.

        Args:
            effect_type: Effect type (reverb, delay, chorus, etc.)
            **params: Effect parameters

        Returns:
            True if successful
        """
        if not self._audio_processor:
            return False

        self._audio_processor.add_effect(effect_type, **params)
        return True

    def apply_effect_preset(self, preset: str) -> bool:
        """
        Apply a voice effect preset.

        Args:
            preset: Preset name (clean, warm, robotic, ethereal)

        Returns:
            True if successful
        """
        if not self._audio_processor:
            return False

        try:
            from ..audio.framework_integrations import EffectPreset

            presets = {
                "clean": EffectPreset.VOICE_CLEAN,
                "warm": EffectPreset.VOICE_WARM,
                "robotic": EffectPreset.VOICE_ROBOTIC,
                "ethereal": EffectPreset.VOICE_ETHEREAL,
            }

            preset_obj = presets.get(preset.lower())
            if preset_obj:
                self._audio_processor.apply_preset(preset_obj)
                return True

        except ImportError:
            pass

        return False

    def get_audio_processor_info(self) -> Dict[str, Any]:
        """Get audio processor information."""
        if not self._audio_processor:
            return {"available": False}

        return self._audio_processor.get_backend_info()

    # =========================================================================
    # Agent Orchestration
    # =========================================================================

    async def run_agent_task(self, task_description: str,
                            task_type: str = "voice",
                            **kwargs) -> Dict[str, Any]:
        """
        Run a task using the AI agent crew.

        Args:
            task_description: Description of the task
            task_type: Type of task (voice, song, vocal_direction)
            **kwargs: Additional task parameters

        Returns:
            Task result
        """
        if not self._agent_crew:
            return {"error": "Agent crew not available"}

        try:
            from .crewai_music_agents import run_production_task_async

            result = await run_production_task_async(
                task_type,
                **kwargs
            )

            return {"success": True, "result": result}

        except Exception as e:
            return {"error": str(e)}

    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents."""
        try:
            from .crewai_music_agents import get_available_roles, get_role_info, AgentRole

            roles = []
            for role_name in get_available_roles():
                role = AgentRole(role_name)
                roles.append(get_role_info(role))

            return roles

        except ImportError:
            return []

    # =========================================================================
    # MCP Server Integration
    # =========================================================================

    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get all MCP tool definitions for external AI access.

        Returns:
            List of MCP tool schemas
        """
        tools = []

        # Voice tools
        tools.extend([
            {
                "name": "hub_speak",
                "description": "Synthesize text to speech",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "voice_name": {"type": "string"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "hub_note_on",
                "description": "Trigger a musical note",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "integer", "minimum": 0, "maximum": 127},
                        "velocity": {"type": "integer", "minimum": 0, "maximum": 127}
                    },
                    "required": ["note"]
                }
            },
            {
                "name": "hub_note_off",
                "description": "Release a musical note",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "integer"}
                    },
                    "required": ["note"]
                }
            },
            {
                "name": "hub_set_vowel",
                "description": "Set voice vowel sound",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "vowel": {"type": "string", "enum": ["A", "E", "I", "O", "U"]}
                    },
                    "required": ["vowel"]
                }
            },
            {
                "name": "hub_voice_state",
                "description": "Get current voice synthesis state",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ])

        # DAW tools
        tools.extend([
            {
                "name": "hub_connect_daw",
                "description": "Connect to Ableton Live",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"}
                    }
                }
            },
            {
                "name": "hub_daw_play",
                "description": "Start DAW playback",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "hub_daw_stop",
                "description": "Stop DAW playback",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "hub_set_tempo",
                "description": "Set session tempo",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bpm": {"type": "number", "minimum": 20, "maximum": 999}
                    },
                    "required": ["bpm"]
                }
            },
            {
                "name": "hub_create_voice_track",
                "description": "Create a voice track in DAW",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            },
            {
                "name": "hub_daw_state",
                "description": "Get current DAW state",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ])

        # Agent tools
        tools.extend([
            {
                "name": "hub_run_agent_task",
                "description": "Run a music production task using AI agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string"},
                        "task_type": {
                            "type": "string",
                            "enum": ["voice", "song", "vocal_direction"]
                        }
                    },
                    "required": ["task_description"]
                }
            },
            {
                "name": "hub_list_agents",
                "description": "List available AI agents",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ])

        return tools

    async def handle_mcp_call(self, tool_name: str,
                              arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        handlers = {
            # Voice handlers
            "hub_speak": lambda: self.speak(**arguments),
            "hub_note_on": lambda: self.note_on(**arguments),
            "hub_note_off": lambda: self.note_off(**arguments),
            "hub_set_vowel": lambda: self.set_vowel(**arguments),
            "hub_voice_state": lambda: self.get_voice_state(),

            # DAW handlers
            "hub_connect_daw": lambda: self.connect_daw(**arguments),
            "hub_daw_play": lambda: self.daw_play(),
            "hub_daw_stop": lambda: self.daw_stop(),
            "hub_set_tempo": lambda: self.set_tempo(**arguments),
            "hub_create_voice_track": lambda: self.create_voice_track(**arguments),
            "hub_daw_state": lambda: self.get_daw_state(),

            # Agent handlers
            "hub_list_agents": lambda: self.get_available_agents(),
        }

        handler = handlers.get(tool_name)

        if handler:
            result = handler()
            return {"success": True, "result": result}

        if tool_name == "hub_run_agent_task":
            result = await self.run_agent_task(**arguments)
            return result

        return {"error": f"Unknown tool: {tool_name}"}

    # =========================================================================
    # Callback Management
    # =========================================================================

    def on(self, event: str, callback: Callable):
        """
        Register an event callback.

        Args:
            event: Event name
            callback: Callback function
        """
        self._callbacks.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable):
        """Remove an event callback."""
        if event in self._callbacks:
            self._callbacks[event] = [
                cb for cb in self._callbacks[event] if cb != callback
            ]

    def _trigger_callbacks(self, event: str, data: Dict[str, Any]):
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def save_session(self, filepath: str) -> bool:
        """
        Save current session state to file.

        Args:
            filepath: Path to save file

        Returns:
            True if successful
        """
        session_data = {
            "config": {
                "name": self.config.name,
                "sample_rate": self.config.sample_rate,
                "buffer_size": self.config.buffer_size,
                "tempo": self.config.tempo,
                "routing_mode": self.config.routing_mode.value,
                "default_voice": self.config.default_voice
            },
            "voice_state": self.get_voice_state(),
            "daw_state": self.get_daw_state()
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self, filepath: str) -> bool:
        """
        Load session state from file.

        Args:
            filepath: Path to session file

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)

            # Restore config
            config = session_data.get("config", {})
            self.config.name = config.get("name", self.config.name)
            self.config.sample_rate = config.get("sample_rate", self.config.sample_rate)
            self.config.tempo = config.get("tempo", self.config.tempo)
            self.config.default_voice = config.get("default_voice", self.config.default_voice)

            # Restore voice state
            voice = session_data.get("voice_state", {})
            self._voice_state.current_voice = voice.get("current_voice", "default")
            self._voice_state.breathiness = voice.get("breathiness", 0.2)
            self._voice_state.vibrato_rate = voice.get("vibrato_rate", 5.5)
            self._voice_state.vibrato_depth = voice.get("vibrato_depth", 0.3)

            logger.info(f"Session loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    # =========================================================================
    # Status & Info
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get overall hub status."""
        return {
            "state": self._state.value,
            "config": {
                "name": self.config.name,
                "sample_rate": self.config.sample_rate,
                "tempo": self.config.tempo,
                "routing_mode": self.config.routing_mode.value
            },
            "subsystems": {
                "voice_pipeline": self._voice_pipeline is not None,
                "ableton_bridge": self._ableton_bridge is not None,
                "audio_processor": self._audio_processor is not None,
                "agent_crew": self._agent_crew is not None
            },
            "voice": self.get_voice_state(),
            "daw": self.get_daw_state()
        }

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get hub capabilities."""
        return {
            "voice": [
                "text_to_speech",
                "vowel_synthesis",
                "formant_control",
                "voice_cloning",
                "neural_tts"
            ],
            "daw": [
                "ableton_control",
                "transport_control",
                "track_management",
                "clip_triggering",
                "midi_control"
            ],
            "audio": [
                "effects_processing",
                "mixing",
                "routing"
            ],
            "agents": [
                "voice_direction",
                "composition",
                "mixing",
                "lyrics",
                "arrangement"
            ]
        }


# ============================================================================
# Convenience Functions
# ============================================================================

_default_hub: Optional[UnifiedHub] = None


def get_hub() -> UnifiedHub:
    """Get or create the default hub instance."""
    global _default_hub
    if _default_hub is None:
        _default_hub = UnifiedHub()
    return _default_hub


def start_hub(config: Optional[SessionConfig] = None,
              blocking: bool = False) -> UnifiedHub:
    """Start a new hub with optional config."""
    global _default_hub
    _default_hub = UnifiedHub(config)
    _default_hub.start(blocking=blocking)
    return _default_hub


def stop_hub():
    """Stop the default hub."""
    global _default_hub
    if _default_hub:
        _default_hub.stop()
        _default_hub = None


def force_stop_hub():
    """Force stop the default hub immediately."""
    global _default_hub
    if _default_hub:
        _default_hub.force_stop()
        _default_hub = None


def shutdown_all():
    """
    Complete shutdown of all DAiW systems.

    This should be called when the application is exiting to ensure
    all resources are properly released.
    """
    global _default_hub

    # Stop the hub
    stop_hub()

    # Also shutdown the tool manager if it exists
    try:
        from .crewai_music_agents import shutdown_tools
        shutdown_tools()
    except ImportError:
        pass

    logger.info("All DAiW systems shut down")


# Register atexit handler for automatic cleanup
import atexit
atexit.register(shutdown_all)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DAiW Unified AI Agent Hub")
    parser.add_argument("--host", default="127.0.0.1", help="DAW host address")
    parser.add_argument("--tempo", type=float, default=120.0, help="Session tempo")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--connect-daw", action="store_true", help="Connect to DAW on start")

    args = parser.parse_args()

    config = SessionConfig(
        tempo=args.tempo,
        sample_rate=args.sample_rate,
        daw_host=args.host
    )

    hub = start_hub(config)

    if args.connect_daw:
        hub.connect_daw()

    print("DAiW Hub running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_hub()
        print("Hub stopped.")


if __name__ == "__main__":
    main()
