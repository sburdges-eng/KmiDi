"""
DAiW Ableton Live Integration Bridge

Provides bidirectional communication between DAiW voice synthesis
and Ableton Live via multiple protocols:
- AbletonOSC (OSC-based remote control)
- MIDI for real-time note/CC control
- Remote Script API (Max for Live device)

Based on patterns from:
- xiaolaa2/ableton-copilot-mcp
- ahujasid/ableton-mcp
- dancohen81/Abai

This bridge enables:
1. Real-time voice synthesis triggered by Ableton clips
2. DAiW formant control via Ableton automation
3. Voice track rendering directly into Ableton
4. AI agent control of Ableton through DAiW MCP
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum
import threading
import queue
import time
import json
import struct
import asyncio
from pathlib import Path

# OSC communication
try:
    from pythonosc import udp_client, dispatcher, osc_server
    from pythonosc.osc_message_builder import OscMessageBuilder
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

# MIDI communication
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

# Audio processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class AbletonConnectionState(Enum):
    """Connection state with Ableton Live"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class AbletonTrackInfo:
    """Information about an Ableton Live track"""
    index: int
    name: str
    armed: bool = False
    muted: bool = False
    soloed: bool = False
    volume: float = 0.0  # dB
    pan: float = 0.0  # -1.0 to 1.0
    color: int = 0
    has_midi_input: bool = False
    has_audio_output: bool = True


@dataclass
class AbletonClipInfo:
    """Information about an Ableton Live clip"""
    track_index: int
    clip_index: int
    name: str
    length: float  # bars
    playing: bool = False
    triggered: bool = False
    color: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    loop_start: float = 0.0
    loop_end: float = 0.0


@dataclass
class AbletonDeviceInfo:
    """Information about an Ableton Live device/plugin"""
    track_index: int
    device_index: int
    name: str
    class_name: str
    parameters: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True


class AbletonOSCBridge:
    """
    OSC-based communication with Ableton Live.

    Requires AbletonOSC or similar OSC remote script installed in Ableton.
    Download: https://github.com/ideoforms/AbletonOSC

    Example:
        bridge = AbletonOSCBridge()
        bridge.connect()

        # Get session info
        tracks = bridge.get_tracks()

        # Control transport
        bridge.play()
        bridge.set_tempo(120)

        # Create voice track
        track_idx = bridge.create_track("DAiW Voice")
        bridge.arm_track(track_idx)
    """

    # Default OSC ports for AbletonOSC
    DEFAULT_SEND_PORT = 11000
    DEFAULT_RECEIVE_PORT = 11001

    def __init__(self,
                 host: str = "127.0.0.1",
                 send_port: int = DEFAULT_SEND_PORT,
                 receive_port: int = DEFAULT_RECEIVE_PORT):
        """
        Initialize the Ableton OSC bridge.

        Args:
            host: Ableton Live host address
            send_port: Port to send OSC messages to Ableton
            receive_port: Port to receive OSC messages from Ableton
        """
        if not OSC_AVAILABLE:
            raise ImportError("python-osc not available. Install with: pip install python-osc")

        self.host = host
        self.send_port = send_port
        self.receive_port = receive_port

        self._client: Optional[udp_client.SimpleUDPClient] = None
        self._server: Optional[osc_server.ThreadingOSCUDPServer] = None
        self._dispatcher: Optional[dispatcher.Dispatcher] = None

        self._state = AbletonConnectionState.DISCONNECTED
        self._response_queue: queue.Queue = queue.Queue()
        self._callbacks: Dict[str, List[Callable]] = {}

        # Cached session data
        self._tracks: List[AbletonTrackInfo] = []
        self._tempo: float = 120.0
        self._time_signature: Tuple[int, int] = (4, 4)
        self._playing: bool = False
        self._recording: bool = False
        self._session_length: float = 0.0

        self._server_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """
        Establish connection with Ableton Live.

        Returns:
            True if connection successful
        """
        try:
            self._state = AbletonConnectionState.CONNECTING

            # Create OSC client for sending
            self._client = udp_client.SimpleUDPClient(self.host, self.send_port)

            # Create dispatcher for receiving
            self._dispatcher = dispatcher.Dispatcher()
            self._setup_handlers()

            # Create and start server
            self._server = osc_server.ThreadingOSCUDPServer(
                ("0.0.0.0", self.receive_port),
                self._dispatcher
            )

            self._server_thread = threading.Thread(target=self._server.serve_forever)
            self._server_thread.daemon = True
            self._server_thread.start()

            # Request initial state from Ableton
            self._send("/live/test")
            time.sleep(0.2)

            # Get session info
            self.refresh_session()

            self._state = AbletonConnectionState.CONNECTED
            return True

        except Exception as e:
            self._state = AbletonConnectionState.ERROR
            print(f"Failed to connect to Ableton: {e}")
            return False

    def disconnect(self):
        """Disconnect from Ableton Live."""
        self._shutdown()

    def _shutdown(self):
        """Internal shutdown - clean up all resources."""
        # Stop the OSC server
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None

        # Wait for server thread to finish
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        self._server_thread = None

        # Clear client
        self._client = None
        self._dispatcher = None

        # Clear state
        self._tracks.clear()
        self._callbacks.clear()

        # Clear response queue
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break

        self._state = AbletonConnectionState.DISCONNECTED

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        self._shutdown()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures disconnect."""
        self.disconnect()
        return False

    def is_connected(self) -> bool:
        """Check if connected to Ableton."""
        return self._state == AbletonConnectionState.CONNECTED

    def _setup_handlers(self):
        """Set up OSC message handlers."""
        # Response handlers
        self._dispatcher.map("/live/test", self._handle_test)
        self._dispatcher.map("/live/tempo", self._handle_tempo)
        self._dispatcher.map("/live/playing", self._handle_playing)
        self._dispatcher.map("/live/recording", self._handle_recording)

        # Track handlers
        self._dispatcher.map("/live/track/info", self._handle_track_info)
        self._dispatcher.map("/live/track/volume", self._handle_track_volume)
        self._dispatcher.map("/live/track/pan", self._handle_track_pan)
        self._dispatcher.map("/live/track/arm", self._handle_track_arm)
        self._dispatcher.map("/live/track/mute", self._handle_track_mute)
        self._dispatcher.map("/live/track/solo", self._handle_track_solo)

        # Clip handlers
        self._dispatcher.map("/live/clip/info", self._handle_clip_info)
        self._dispatcher.map("/live/clip/playing", self._handle_clip_playing)

        # Device handlers
        self._dispatcher.map("/live/device/info", self._handle_device_info)
        self._dispatcher.map("/live/device/parameter", self._handle_device_param)

        # Catch-all for debugging
        self._dispatcher.set_default_handler(self._handle_default)

    def _send(self, address: str, *args):
        """Send an OSC message to Ableton."""
        if self._client:
            self._client.send_message(address, args)

    def _handle_default(self, address: str, *args):
        """Default handler for unmatched messages."""
        self._response_queue.put((address, args))

    def _handle_test(self, address: str, *args):
        """Handle test/ping response."""
        self._state = AbletonConnectionState.CONNECTED

    def _handle_tempo(self, address: str, *args):
        """Handle tempo response."""
        if args:
            self._tempo = float(args[0])

    def _handle_playing(self, address: str, *args):
        """Handle playing state."""
        if args:
            self._playing = bool(args[0])

    def _handle_recording(self, address: str, *args):
        """Handle recording state."""
        if args:
            self._recording = bool(args[0])

    def _handle_track_info(self, address: str, *args):
        """Handle track info response."""
        if len(args) >= 3:
            track = AbletonTrackInfo(
                index=int(args[0]),
                name=str(args[1]),
                color=int(args[2]) if len(args) > 2 else 0
            )
            # Update or add track
            existing = next((t for t in self._tracks if t.index == track.index), None)
            if existing:
                self._tracks[self._tracks.index(existing)] = track
            else:
                self._tracks.append(track)

    def _handle_track_volume(self, address: str, *args):
        """Handle track volume response."""
        if len(args) >= 2:
            track_idx = int(args[0])
            volume = float(args[1])
            for track in self._tracks:
                if track.index == track_idx:
                    track.volume = volume
                    break

    def _handle_track_pan(self, address: str, *args):
        """Handle track pan response."""
        if len(args) >= 2:
            track_idx = int(args[0])
            pan = float(args[1])
            for track in self._tracks:
                if track.index == track_idx:
                    track.pan = pan
                    break

    def _handle_track_arm(self, address: str, *args):
        """Handle track arm response."""
        if len(args) >= 2:
            track_idx = int(args[0])
            armed = bool(args[1])
            for track in self._tracks:
                if track.index == track_idx:
                    track.armed = armed
                    break

    def _handle_track_mute(self, address: str, *args):
        """Handle track mute response."""
        if len(args) >= 2:
            track_idx = int(args[0])
            muted = bool(args[1])
            for track in self._tracks:
                if track.index == track_idx:
                    track.muted = muted
                    break

    def _handle_track_solo(self, address: str, *args):
        """Handle track solo response."""
        if len(args) >= 2:
            track_idx = int(args[0])
            soloed = bool(args[1])
            for track in self._tracks:
                if track.index == track_idx:
                    track.soloed = soloed
                    break

    def _handle_clip_info(self, address: str, *args):
        """Handle clip info response."""
        pass  # Implement as needed

    def _handle_clip_playing(self, address: str, *args):
        """Handle clip playing status."""
        pass  # Implement as needed

    def _handle_device_info(self, address: str, *args):
        """Handle device info response."""
        pass  # Implement as needed

    def _handle_device_param(self, address: str, *args):
        """Handle device parameter response."""
        pass  # Implement as needed

    # =========================================================================
    # Transport Control
    # =========================================================================

    def play(self):
        """Start playback."""
        self._send("/live/play")
        self._playing = True

    def stop(self):
        """Stop playback."""
        self._send("/live/stop")
        self._playing = False

    def continue_playback(self):
        """Continue playback from current position."""
        self._send("/live/continue")

    def record(self):
        """Start recording."""
        self._send("/live/record")
        self._recording = True

    def set_tempo(self, bpm: float):
        """Set session tempo."""
        self._send("/live/tempo", bpm)
        self._tempo = bpm

    def get_tempo(self) -> float:
        """Get session tempo."""
        self._send("/live/tempo/get")
        time.sleep(0.05)
        return self._tempo

    def set_time(self, beats: float):
        """Set playback position in beats."""
        self._send("/live/time", beats)

    def get_time(self) -> float:
        """Get current playback position in beats."""
        self._send("/live/time/get")
        # Wait for response
        time.sleep(0.05)
        return 0.0  # Would be updated by handler

    def set_loop(self, start: float, length: float):
        """Set loop region."""
        self._send("/live/loop/start", start)
        self._send("/live/loop/length", length)

    def enable_loop(self, enabled: bool = True):
        """Enable/disable looping."""
        self._send("/live/loop", int(enabled))

    def is_playing(self) -> bool:
        """Check if playing."""
        return self._playing

    def is_recording(self) -> bool:
        """Check if recording."""
        return self._recording

    # =========================================================================
    # Track Management
    # =========================================================================

    def refresh_session(self):
        """Refresh all session data from Ableton."""
        self._send("/live/tracks/get")
        self._send("/live/tempo/get")
        self._send("/live/playing/get")
        time.sleep(0.1)

    def get_tracks(self) -> List[AbletonTrackInfo]:
        """Get all tracks."""
        self._send("/live/tracks/get")
        time.sleep(0.1)
        return self._tracks.copy()

    def get_track(self, index: int) -> Optional[AbletonTrackInfo]:
        """Get track by index."""
        for track in self._tracks:
            if track.index == index:
                return track
        return None

    def create_track(self, name: str, track_type: str = "audio") -> int:
        """
        Create a new track.

        Args:
            name: Track name
            track_type: "audio" or "midi"

        Returns:
            New track index
        """
        if track_type == "midi":
            self._send("/live/track/create/midi")
        else:
            self._send("/live/track/create/audio")

        time.sleep(0.2)
        self.refresh_session()

        # Find the new track and rename it
        if self._tracks:
            new_track_idx = max(t.index for t in self._tracks)
            self.rename_track(new_track_idx, name)
            return new_track_idx

        return -1

    def delete_track(self, index: int):
        """Delete a track."""
        self._send("/live/track/delete", index)
        self._tracks = [t for t in self._tracks if t.index != index]

    def rename_track(self, index: int, name: str):
        """Rename a track."""
        self._send("/live/track/name", index, name)
        for track in self._tracks:
            if track.index == index:
                track.name = name
                break

    def arm_track(self, index: int, armed: bool = True):
        """Arm/disarm track for recording."""
        self._send("/live/track/arm", index, int(armed))

    def mute_track(self, index: int, muted: bool = True):
        """Mute/unmute track."""
        self._send("/live/track/mute", index, int(muted))

    def solo_track(self, index: int, soloed: bool = True):
        """Solo/unsolo track."""
        self._send("/live/track/solo", index, int(soloed))

    def set_track_volume(self, index: int, volume_db: float):
        """Set track volume in dB."""
        # Convert dB to 0-1 range (approximate)
        volume_linear = 10 ** (volume_db / 20) if volume_db > -70 else 0.0
        volume_linear = min(1.0, max(0.0, volume_linear))
        self._send("/live/track/volume", index, volume_linear)

    def set_track_pan(self, index: int, pan: float):
        """Set track pan (-1.0 to 1.0)."""
        self._send("/live/track/pan", index, pan)

    # =========================================================================
    # Clip Management
    # =========================================================================

    def fire_clip(self, track: int, clip: int):
        """Fire/trigger a clip."""
        self._send("/live/clip/fire", track, clip)

    def stop_clip(self, track: int, clip: int):
        """Stop a clip."""
        self._send("/live/clip/stop", track, clip)

    def create_clip(self, track: int, clip: int, length: float = 4.0) -> bool:
        """
        Create a new clip.

        Args:
            track: Track index
            clip: Clip slot index
            length: Clip length in bars

        Returns:
            True if successful
        """
        self._send("/live/clip/create", track, clip, length)
        return True

    def set_clip_notes(self, track: int, clip: int,
                       notes: List[Tuple[int, float, float, int]]):
        """
        Set MIDI notes in a clip.

        Args:
            track: Track index
            clip: Clip index
            notes: List of (pitch, start_time, duration, velocity)
        """
        # Clear existing notes
        self._send("/live/clip/notes/clear", track, clip)

        # Add new notes
        for pitch, start, duration, velocity in notes:
            self._send("/live/clip/notes/add", track, clip,
                      pitch, start, duration, velocity)

    def set_clip_name(self, track: int, clip: int, name: str):
        """Set clip name."""
        self._send("/live/clip/name", track, clip, name)

    # =========================================================================
    # Device Control
    # =========================================================================

    def get_devices(self, track: int) -> List[AbletonDeviceInfo]:
        """Get all devices on a track."""
        self._send("/live/track/devices/get", track)
        time.sleep(0.1)
        return []  # Would return cached device list

    def set_device_parameter(self, track: int, device: int,
                            param: int, value: float):
        """Set a device parameter."""
        self._send("/live/device/parameter", track, device, param, value)

    def load_device(self, track: int, device_uri: str):
        """
        Load a device onto a track.

        Args:
            track: Track index
            device_uri: Device URI (e.g., "Reverb", "vst3://...")
        """
        self._send("/live/device/load", track, device_uri)

    # =========================================================================
    # Audio Export
    # =========================================================================

    def export_audio(self, output_path: str,
                     start_bar: float = 0, end_bar: float = -1):
        """
        Export audio from Ableton.

        Note: This typically requires a Max for Live device or
        custom Remote Script to work properly.
        """
        self._send("/live/export/audio", output_path, start_bar, end_bar)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_tempo_change(self, callback: Callable[[float], None]):
        """Register callback for tempo changes."""
        self._callbacks.setdefault("tempo", []).append(callback)

    def on_transport_change(self, callback: Callable[[bool, bool], None]):
        """Register callback for transport changes (playing, recording)."""
        self._callbacks.setdefault("transport", []).append(callback)

    def on_clip_trigger(self, callback: Callable[[int, int], None]):
        """Register callback for clip triggers."""
        self._callbacks.setdefault("clip_trigger", []).append(callback)

    @property
    def state(self) -> AbletonConnectionState:
        """Get connection state."""
        return self._state


class AbletonMIDIBridge:
    """
    MIDI-based communication with Ableton Live.

    Uses virtual MIDI ports for note/CC control and clip triggering.
    More reliable than OSC for real-time performance.

    Example:
        bridge = AbletonMIDIBridge()
        bridge.connect("DAiW Virtual Output")

        # Send notes
        bridge.note_on(60, 100)
        bridge.note_off(60)

        # Control formants via CC
        bridge.control_change(1, 64)  # Vowel position
    """

    # CC mappings for DAiW voice control
    CC_VOWEL_POSITION = 1      # Modulation wheel -> vowel morph
    CC_FORMANT_SHIFT = 74      # Filter cutoff -> formant shift
    CC_BREATHINESS = 2         # Breath -> breathiness
    CC_VIBRATO_RATE = 76       # Sound controller 7 -> vibrato rate
    CC_VIBRATO_DEPTH = 77      # Sound controller 8 -> vibrato depth
    CC_PITCH_BEND_RANGE = 85   # Reserved for pitch bend range
    CC_VOICE_SELECT = 86       # Voice preset selection

    def __init__(self):
        if not MIDO_AVAILABLE:
            raise ImportError("mido not available. Install with: pip install mido python-rtmidi")

        self._input_port: Optional[mido.ports.BaseInput] = None
        self._output_port: Optional[mido.ports.BaseOutput] = None
        self._virtual_port_name = "DAiW Voice Bridge"

        self._note_callbacks: List[Callable] = []
        self._cc_callbacks: List[Callable] = []
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None

    def connect(self, output_port: Optional[str] = None,
                input_port: Optional[str] = None,
                create_virtual: bool = True) -> bool:
        """
        Connect to MIDI ports.

        Args:
            output_port: Name of output port (to Ableton)
            input_port: Name of input port (from Ableton)
            create_virtual: Create virtual MIDI port

        Returns:
            True if connected
        """
        try:
            if create_virtual:
                # Create virtual MIDI port
                self._output_port = mido.open_output(
                    self._virtual_port_name,
                    virtual=True
                )
                self._input_port = mido.open_input(
                    f"{self._virtual_port_name} In",
                    virtual=True,
                    callback=self._handle_midi_message
                )
            else:
                # Connect to existing ports
                if output_port:
                    self._output_port = mido.open_output(output_port)
                if input_port:
                    self._input_port = mido.open_input(
                        input_port,
                        callback=self._handle_midi_message
                    )

            self._running = True
            return True

        except Exception as e:
            print(f"Failed to connect MIDI: {e}")
            return False

    def disconnect(self):
        """Disconnect MIDI ports."""
        self._shutdown()

    def _shutdown(self):
        """Internal shutdown - clean up all MIDI resources."""
        self._running = False

        # Send all notes off before closing
        if self._output_port:
            try:
                # All notes off on all channels
                for channel in range(16):
                    msg = mido.Message('control_change', control=123, value=0, channel=channel)
                    self._output_port.send(msg)
            except Exception:
                pass

        # Close output port
        if self._output_port:
            try:
                self._output_port.close()
            except Exception:
                pass
            self._output_port = None

        # Close input port
        if self._input_port:
            try:
                self._input_port.close()
            except Exception:
                pass
            self._input_port = None

        # Clear callbacks
        self._note_callbacks.clear()
        self._cc_callbacks.clear()

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        self._shutdown()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures disconnect."""
        self.disconnect()
        return False

    def is_connected(self) -> bool:
        """Check if MIDI is connected."""
        return self._running and (self._output_port is not None or self._input_port is not None)

    def _handle_midi_message(self, message: mido.Message):
        """Handle incoming MIDI message."""
        if message.type == 'note_on':
            for callback in self._note_callbacks:
                callback(message.note, message.velocity, message.channel)
        elif message.type == 'note_off':
            for callback in self._note_callbacks:
                callback(message.note, 0, message.channel)
        elif message.type == 'control_change':
            for callback in self._cc_callbacks:
                callback(message.control, message.value, message.channel)

    def note_on(self, note: int, velocity: int = 100, channel: int = 0):
        """Send note on."""
        if self._output_port:
            msg = mido.Message('note_on', note=note, velocity=velocity, channel=channel)
            self._output_port.send(msg)

    def note_off(self, note: int, channel: int = 0):
        """Send note off."""
        if self._output_port:
            msg = mido.Message('note_off', note=note, velocity=0, channel=channel)
            self._output_port.send(msg)

    def control_change(self, control: int, value: int, channel: int = 0):
        """Send control change."""
        if self._output_port:
            msg = mido.Message('control_change', control=control, value=value, channel=channel)
            self._output_port.send(msg)

    def pitch_bend(self, value: int, channel: int = 0):
        """Send pitch bend (-8192 to 8191)."""
        if self._output_port:
            msg = mido.Message('pitchwheel', pitch=value, channel=channel)
            self._output_port.send(msg)

    def program_change(self, program: int, channel: int = 0):
        """Send program change (voice preset)."""
        if self._output_port:
            msg = mido.Message('program_change', program=program, channel=channel)
            self._output_port.send(msg)

    # =========================================================================
    # DAiW Voice Control via MIDI
    # =========================================================================

    def set_vowel_position(self, position: float):
        """Set vowel morph position (0.0 to 1.0)."""
        cc_value = int(position * 127)
        self.control_change(self.CC_VOWEL_POSITION, cc_value)

    def set_formant_shift(self, shift: float):
        """Set formant shift (-1.0 to 1.0)."""
        cc_value = int((shift + 1.0) * 63.5)
        self.control_change(self.CC_FORMANT_SHIFT, cc_value)

    def set_breathiness(self, breathiness: float):
        """Set breathiness (0.0 to 1.0)."""
        cc_value = int(breathiness * 127)
        self.control_change(self.CC_BREATHINESS, cc_value)

    def set_vibrato(self, rate: float, depth: float):
        """Set vibrato parameters."""
        rate_cc = int(rate * 127)  # 0-127 maps to vibrato rate
        depth_cc = int(depth * 127)
        self.control_change(self.CC_VIBRATO_RATE, rate_cc)
        self.control_change(self.CC_VIBRATO_DEPTH, depth_cc)

    def select_voice(self, voice_index: int):
        """Select voice preset."""
        self.control_change(self.CC_VOICE_SELECT, voice_index)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_note(self, callback: Callable[[int, int, int], None]):
        """Register note callback (note, velocity, channel)."""
        self._note_callbacks.append(callback)

    def on_cc(self, callback: Callable[[int, int, int], None]):
        """Register CC callback (control, value, channel)."""
        self._cc_callbacks.append(callback)

    @staticmethod
    def list_ports() -> Tuple[List[str], List[str]]:
        """List available MIDI ports."""
        inputs = mido.get_input_names()
        outputs = mido.get_output_names()
        return inputs, outputs


class DAiWAbletonIntegration:
    """
    High-level integration between DAiW voice synthesis and Ableton Live.

    Combines OSC and MIDI bridges with the DAiW voice synthesis pipeline
    to provide seamless DAW integration.

    Example:
        integration = DAiWAbletonIntegration()
        integration.connect()

        # Create a voice track in Ableton
        track_idx = integration.create_voice_track("Lead Vocal")

        # Render voice to Ableton
        integration.render_text_to_track(
            "Hello world",
            track_idx,
            start_beat=0,
            voice_name="default"
        )

        # Enable live MIDI control of voice
        integration.enable_midi_voice_control(track_idx)
    """

    def __init__(self, voice_pipeline=None):
        """
        Initialize DAiW-Ableton integration.

        Args:
            voice_pipeline: Optional VoiceSynthesisPipeline instance
        """
        self.osc_bridge = AbletonOSCBridge()
        self.midi_bridge = AbletonMIDIBridge()

        self._voice_pipeline = voice_pipeline
        self._voice_tracks: Dict[int, str] = {}  # track_idx -> voice_name
        self._midi_controlled_tracks: List[int] = []

        # Import voice pipeline if not provided
        if self._voice_pipeline is None:
            try:
                from .daiw_mcp_server import voice_pipeline
                self._voice_pipeline = voice_pipeline
            except ImportError:
                pass

    def connect(self) -> bool:
        """
        Connect to Ableton Live via both OSC and MIDI.

        Returns:
            True if connected
        """
        osc_connected = self.osc_bridge.connect()
        midi_connected = self.midi_bridge.connect()

        if midi_connected:
            # Set up MIDI callbacks for voice control
            self.midi_bridge.on_note(self._handle_midi_note)
            self.midi_bridge.on_cc(self._handle_midi_cc)

        return osc_connected or midi_connected

    def disconnect(self):
        """Disconnect from Ableton Live."""
        self._shutdown()

    def _shutdown(self):
        """Internal shutdown - clean up all resources."""
        # Clear MIDI controlled tracks
        self._midi_controlled_tracks.clear()
        self._voice_tracks.clear()

        # Disconnect bridges
        if self.osc_bridge:
            self.osc_bridge.disconnect()

        if self.midi_bridge:
            self.midi_bridge.disconnect()

        # Clear voice pipeline reference
        self._voice_pipeline = None

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        self._shutdown()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures disconnect."""
        self.disconnect()
        return False

    def is_connected(self) -> bool:
        """Check if connected to Ableton."""
        return self.osc_bridge.is_connected() or self.midi_bridge.is_connected()

    def _handle_midi_note(self, note: int, velocity: int, channel: int):
        """Handle incoming MIDI note for voice control."""
        if channel in self._midi_controlled_tracks:
            if velocity > 0:
                # Note on - trigger voice
                frequency = 440.0 * (2 ** ((note - 69) / 12))
                if self._voice_pipeline:
                    # This would connect to the voice pipeline
                    pass
            else:
                # Note off
                pass

    def _handle_midi_cc(self, control: int, value: int, channel: int):
        """Handle incoming MIDI CC for voice control."""
        if channel in self._midi_controlled_tracks:
            normalized = value / 127.0

            if control == AbletonMIDIBridge.CC_VOWEL_POSITION:
                # Map to vowel (0=A, 0.2=E, 0.4=I, 0.6=O, 0.8=U)
                vowel_idx = int(normalized * 5)
                if self._voice_pipeline:
                    # Set vowel
                    pass

            elif control == AbletonMIDIBridge.CC_FORMANT_SHIFT:
                shift = (normalized * 2.0) - 1.0  # -1 to 1
                if self._voice_pipeline:
                    # Set formant shift
                    pass

            elif control == AbletonMIDIBridge.CC_BREATHINESS:
                if self._voice_pipeline:
                    # Set breathiness
                    pass

    def create_voice_track(self, name: str = "DAiW Voice") -> int:
        """
        Create a new voice track in Ableton.

        Args:
            name: Track name

        Returns:
            Track index
        """
        track_idx = self.osc_bridge.create_track(name, track_type="audio")
        self._voice_tracks[track_idx] = "default"
        return track_idx

    def render_text_to_track(self, text: str, track_idx: int,
                             start_beat: float = 0,
                             voice_name: Optional[str] = None) -> bool:
        """
        Render synthesized voice text to an Ableton track.

        Args:
            text: Text to synthesize
            track_idx: Target track index
            start_beat: Start position in beats
            voice_name: Voice preset to use

        Returns:
            True if successful
        """
        if not self._voice_pipeline:
            print("Voice pipeline not available")
            return False

        try:
            # Synthesize the text
            audio = self._voice_pipeline.synthesize(text, voice_name)

            if audio is None:
                return False

            # Calculate clip length based on audio duration
            sample_rate = self._voice_pipeline.sample_rate
            duration_seconds = len(audio) / sample_rate
            tempo = self.osc_bridge.get_tempo()
            duration_beats = (duration_seconds / 60.0) * tempo

            # Export audio to temp file
            import tempfile
            import soundfile as sf

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio, sample_rate)

            # This would typically require a Max for Live device
            # or custom Remote Script to import audio
            # For now, we just prepare the data

            return True

        except Exception as e:
            print(f"Failed to render text to track: {e}")
            return False

    def enable_midi_voice_control(self, track_idx: int):
        """
        Enable MIDI control of voice parameters for a track.

        The MIDI channel corresponds to the track index (0-15).
        """
        channel = track_idx % 16
        self._midi_controlled_tracks.append(channel)

    def disable_midi_voice_control(self, track_idx: int):
        """Disable MIDI voice control for a track."""
        channel = track_idx % 16
        if channel in self._midi_controlled_tracks:
            self._midi_controlled_tracks.remove(channel)

    def sync_to_tempo(self) -> float:
        """Sync voice synthesis to Ableton tempo."""
        tempo = self.osc_bridge.get_tempo()
        # Update voice pipeline timing if needed
        return tempo

    def get_session_info(self) -> Dict[str, Any]:
        """Get current Ableton session info."""
        return {
            "tempo": self.osc_bridge._tempo,
            "playing": self.osc_bridge._playing,
            "recording": self.osc_bridge._recording,
            "tracks": [
                {
                    "index": t.index,
                    "name": t.name,
                    "armed": t.armed,
                    "muted": t.muted,
                    "volume": t.volume
                }
                for t in self.osc_bridge._tracks
            ],
            "voice_tracks": self._voice_tracks,
            "midi_controlled": self._midi_controlled_tracks
        }


# ============================================================================
# MCP Tool Extensions for Ableton
# ============================================================================

def create_ableton_mcp_tools() -> List[Dict[str, Any]]:
    """
    Create MCP tool definitions for Ableton Live control.

    These tools can be added to the DAiW MCP server to expose
    Ableton control to AI agents like Claude.
    """
    return [
        {
            "name": "ableton_connect",
            "description": "Connect to Ableton Live",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "Ableton host address",
                        "default": "127.0.0.1"
                    }
                }
            }
        },
        {
            "name": "ableton_transport",
            "description": "Control Ableton transport (play, stop, record)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "stop", "record", "continue"],
                        "description": "Transport action"
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "ableton_set_tempo",
            "description": "Set Ableton session tempo",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bpm": {
                        "type": "number",
                        "description": "Tempo in BPM",
                        "minimum": 20,
                        "maximum": 999
                    }
                },
                "required": ["bpm"]
            }
        },
        {
            "name": "ableton_create_track",
            "description": "Create a new track in Ableton",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Track name"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["audio", "midi"],
                        "default": "audio"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "ableton_arm_track",
            "description": "Arm a track for recording",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track_index": {
                        "type": "integer",
                        "description": "Track index"
                    },
                    "armed": {
                        "type": "boolean",
                        "default": True
                    }
                },
                "required": ["track_index"]
            }
        },
        {
            "name": "ableton_fire_clip",
            "description": "Fire/trigger a clip",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track": {
                        "type": "integer",
                        "description": "Track index"
                    },
                    "clip": {
                        "type": "integer",
                        "description": "Clip slot index"
                    }
                },
                "required": ["track", "clip"]
            }
        },
        {
            "name": "ableton_create_voice_clip",
            "description": "Create a clip with synthesized voice audio",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track": {
                        "type": "integer",
                        "description": "Track index"
                    },
                    "clip": {
                        "type": "integer",
                        "description": "Clip slot index"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to synthesize"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice preset name"
                    }
                },
                "required": ["track", "clip", "text"]
            }
        },
        {
            "name": "ableton_get_session",
            "description": "Get current Ableton session info",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "ableton_set_track_volume",
            "description": "Set track volume",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track_index": {
                        "type": "integer"
                    },
                    "volume_db": {
                        "type": "number",
                        "minimum": -70,
                        "maximum": 6
                    }
                },
                "required": ["track_index", "volume_db"]
            }
        },
        {
            "name": "ableton_midi_voice_control",
            "description": "Send MIDI control to DAiW voice synth",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "control": {
                        "type": "string",
                        "enum": ["vowel", "formant", "breathiness", "vibrato"],
                        "description": "Control type"
                    },
                    "value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Normalized value (0-1)"
                    }
                },
                "required": ["control", "value"]
            }
        }
    ]


# ============================================================================
# Utility Functions
# ============================================================================

def check_ableton_connection() -> Dict[str, Any]:
    """
    Check if Ableton Live is running and accessible.

    Returns:
        Connection status and available ports
    """
    status = {
        "osc_available": OSC_AVAILABLE,
        "midi_available": MIDO_AVAILABLE,
        "ableton_detected": False,
        "midi_ports": {"inputs": [], "outputs": []}
    }

    if MIDO_AVAILABLE:
        inputs, outputs = AbletonMIDIBridge.list_ports()
        status["midi_ports"]["inputs"] = list(inputs)
        status["midi_ports"]["outputs"] = list(outputs)

        # Check for Ableton-related ports
        ableton_keywords = ["ableton", "live", "iac"]
        for port in inputs + outputs:
            if any(kw in port.lower() for kw in ableton_keywords):
                status["ableton_detected"] = True
                break

    return status


def get_default_cc_mappings() -> Dict[str, int]:
    """Get default MIDI CC mappings for voice control."""
    return {
        "vowel_position": AbletonMIDIBridge.CC_VOWEL_POSITION,
        "formant_shift": AbletonMIDIBridge.CC_FORMANT_SHIFT,
        "breathiness": AbletonMIDIBridge.CC_BREATHINESS,
        "vibrato_rate": AbletonMIDIBridge.CC_VIBRATO_RATE,
        "vibrato_depth": AbletonMIDIBridge.CC_VIBRATO_DEPTH,
        "voice_select": AbletonMIDIBridge.CC_VOICE_SELECT
    }
