"""
DAiW Audio Framework Integrations

Unified interface for multiple audio processing backends:
- Pedalboard (Spotify) - Effects processing, VST/AU hosting
- DawDreamer - Full DAW rendering, FAUST, VST instruments
- JACK/PipeWire - Real-time audio routing (Linux/macOS)
- Web Audio API - Browser-based audio (via WebSocket bridge)

This module provides a consistent API across all backends for the
DAiW voice synthesis and audio processing pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum
from pathlib import Path
import numpy as np
import json
import threading
import time

# Backend availability detection
PEDALBOARD_AVAILABLE = False
DAWDREAMER_AVAILABLE = False
JACK_AVAILABLE = False
SOUNDFILE_AVAILABLE = False

try:
    import pedalboard
    from pedalboard import (
        Pedalboard, Chorus, Reverb, Delay, Distortion,
        Compressor, Gain, Limiter, HighpassFilter, LowpassFilter,
        PitchShift, Resample, Convolution, load_plugin
    )
    from pedalboard.io import AudioFile, AudioStream
    PEDALBOARD_AVAILABLE = True
except ImportError:
    pass

try:
    import dawdreamer as daw
    DAWDREAMER_AVAILABLE = True
except ImportError:
    pass

try:
    import jack
    JACK_AVAILABLE = True
except ImportError:
    pass

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    pass


class AudioBackend(Enum):
    """Available audio processing backends"""
    PEDALBOARD = "pedalboard"
    DAWDREAMER = "dawdreamer"
    JACK = "jack"
    PIPEWIRE = "pipewire"
    WEB_AUDIO = "webaudio"
    NATIVE = "native"  # Direct numpy processing


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 44100
    buffer_size: int = 512
    channels: int = 2
    bit_depth: int = 24
    backend: AudioBackend = AudioBackend.PEDALBOARD


@dataclass
class EffectPreset:
    """Preset for audio effects chain"""
    name: str
    effects: List[Dict[str, Any]] = field(default_factory=list)

    # Voice-specific presets
    VOICE_CLEAN = None
    VOICE_WARM = None
    VOICE_ROBOTIC = None
    VOICE_ETHEREAL = None


# Initialize voice presets
EffectPreset.VOICE_CLEAN = EffectPreset(
    name="voice_clean",
    effects=[
        {"type": "highpass", "cutoff_hz": 80},
        {"type": "compressor", "threshold_db": -20, "ratio": 3.0},
        {"type": "gain", "gain_db": 3.0}
    ]
)

EffectPreset.VOICE_WARM = EffectPreset(
    name="voice_warm",
    effects=[
        {"type": "highpass", "cutoff_hz": 60},
        {"type": "lowpass", "cutoff_hz": 12000},
        {"type": "compressor", "threshold_db": -18, "ratio": 4.0},
        {"type": "chorus", "rate_hz": 0.5, "depth": 0.1},
        {"type": "reverb", "room_size": 0.2, "wet_level": 0.15}
    ]
)

EffectPreset.VOICE_ROBOTIC = EffectPreset(
    name="voice_robotic",
    effects=[
        {"type": "distortion", "drive_db": 10},
        {"type": "highpass", "cutoff_hz": 200},
        {"type": "lowpass", "cutoff_hz": 4000},
        {"type": "delay", "delay_seconds": 0.02, "feedback": 0.3}
    ]
)

EffectPreset.VOICE_ETHEREAL = EffectPreset(
    name="voice_ethereal",
    effects=[
        {"type": "pitch_shift", "semitones": 12},
        {"type": "reverb", "room_size": 0.9, "wet_level": 0.6},
        {"type": "delay", "delay_seconds": 0.3, "feedback": 0.5},
        {"type": "chorus", "rate_hz": 0.3, "depth": 0.4}
    ]
)


class AudioProcessor(ABC):
    """Abstract base class for audio processors"""

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio buffer"""
        pass

    @abstractmethod
    def get_latency_samples(self) -> int:
        """Get processing latency in samples"""
        pass


class PedalboardProcessor(AudioProcessor):
    """
    Pedalboard-based audio processor (Spotify).

    Provides high-performance audio effects with VST3/AU plugin support.

    Example:
        processor = PedalboardProcessor()
        processor.add_effect("reverb", room_size=0.5)
        processor.add_effect("compressor", threshold_db=-20)

        output = processor.process(audio, 44100)
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        if not PEDALBOARD_AVAILABLE:
            raise ImportError("pedalboard not available. Install with: pip install pedalboard")

        self.config = config or AudioConfig()
        self.board = Pedalboard([])
        self._plugins: Dict[str, Any] = {}

    def add_effect(self, effect_type: str, **kwargs) -> 'PedalboardProcessor':
        """
        Add an effect to the chain.

        Args:
            effect_type: Type of effect (reverb, delay, chorus, etc.)
            **kwargs: Effect-specific parameters

        Returns:
            self for chaining
        """
        effect = self._create_effect(effect_type, **kwargs)
        if effect:
            self.board.append(effect)
        return self

    def _create_effect(self, effect_type: str, **kwargs):
        """Create a pedalboard effect from type string."""
        effect_map = {
            "reverb": lambda: Reverb(
                room_size=kwargs.get("room_size", 0.5),
                wet_level=kwargs.get("wet_level", 0.33),
                dry_level=kwargs.get("dry_level", 0.4)
            ),
            "delay": lambda: Delay(
                delay_seconds=kwargs.get("delay_seconds", 0.25),
                feedback=kwargs.get("feedback", 0.3),
                mix=kwargs.get("mix", 0.5)
            ),
            "chorus": lambda: Chorus(
                rate_hz=kwargs.get("rate_hz", 1.0),
                depth=kwargs.get("depth", 0.25),
                mix=kwargs.get("mix", 0.5)
            ),
            "distortion": lambda: Distortion(
                drive_db=kwargs.get("drive_db", 25)
            ),
            "compressor": lambda: Compressor(
                threshold_db=kwargs.get("threshold_db", -20),
                ratio=kwargs.get("ratio", 4.0),
                attack_ms=kwargs.get("attack_ms", 1.0),
                release_ms=kwargs.get("release_ms", 100)
            ),
            "gain": lambda: Gain(
                gain_db=kwargs.get("gain_db", 0.0)
            ),
            "limiter": lambda: Limiter(
                threshold_db=kwargs.get("threshold_db", -6),
                release_ms=kwargs.get("release_ms", 100)
            ),
            "highpass": lambda: HighpassFilter(
                cutoff_frequency_hz=kwargs.get("cutoff_hz", 80)
            ),
            "lowpass": lambda: LowpassFilter(
                cutoff_frequency_hz=kwargs.get("cutoff_hz", 8000)
            ),
            "pitch_shift": lambda: PitchShift(
                semitones=kwargs.get("semitones", 0)
            ),
        }

        creator = effect_map.get(effect_type.lower())
        if creator:
            return creator()
        return None

    def load_vst(self, plugin_path: str, plugin_name: Optional[str] = None) -> bool:
        """
        Load a VST3/AU plugin.

        Args:
            plugin_path: Path to plugin file
            plugin_name: Optional name for reference

        Returns:
            True if loaded successfully
        """
        try:
            plugin = load_plugin(plugin_path)
            name = plugin_name or Path(plugin_path).stem
            self._plugins[name] = plugin
            self.board.append(plugin)
            return True
        except Exception as e:
            print(f"Failed to load plugin {plugin_path}: {e}")
            return False

    def apply_preset(self, preset: EffectPreset):
        """Apply an effect preset."""
        self.clear()
        for effect_config in preset.effects:
            effect_type = effect_config.pop("type")
            self.add_effect(effect_type, **effect_config)

    def clear(self):
        """Clear all effects."""
        self.board = Pedalboard([])
        self._plugins.clear()

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through the effect chain."""
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure 2D (channels, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        return self.board(audio, sample_rate)

    def process_file(self, input_path: str, output_path: str):
        """Process an audio file through the effect chain."""
        with AudioFile(input_path) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate

        processed = self.process(audio, sample_rate)

        with AudioFile(output_path, 'w', sample_rate, processed.shape[0]) as f:
            f.write(processed)

    def get_latency_samples(self) -> int:
        """Get total latency of the effect chain."""
        # Pedalboard processes in real-time with minimal latency
        return self.config.buffer_size


class DawDreamerProcessor(AudioProcessor):
    """
    DawDreamer-based audio processor.

    Provides full DAW rendering with VST instruments, FAUST, and MIDI.

    Example:
        processor = DawDreamerProcessor()
        processor.load_synth("path/to/synth.vst3")
        processor.set_midi(midi_notes)

        audio = processor.render(duration=10.0)
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        if not DAWDREAMER_AVAILABLE:
            raise ImportError("dawdreamer not available. Install with: pip install dawdreamer")

        self.config = config or AudioConfig()
        self.engine = daw.RenderEngine(
            self.config.sample_rate,
            self.config.buffer_size
        )
        self._processors: Dict[str, Any] = {}
        self._graph_built = False

    def add_plugin(self, name: str, plugin_path: str) -> bool:
        """
        Add a VST3/AU plugin processor.

        Args:
            name: Unique name for the processor
            plugin_path: Path to plugin file

        Returns:
            True if added successfully
        """
        try:
            processor = self.engine.make_plugin_processor(name, plugin_path)
            self._processors[name] = processor
            return True
        except Exception as e:
            print(f"Failed to add plugin {name}: {e}")
            return False

    def add_faust(self, name: str, dsp_code: str) -> bool:
        """
        Add a FAUST DSP processor.

        Args:
            name: Unique name for the processor
            dsp_code: FAUST DSP code string

        Returns:
            True if added successfully
        """
        try:
            processor = self.engine.make_faust_processor(name)
            processor.set_dsp_string(dsp_code)
            self._processors[name] = processor
            return True
        except Exception as e:
            print(f"Failed to add FAUST processor {name}: {e}")
            return False

    def add_playback(self, name: str, audio: np.ndarray) -> bool:
        """
        Add a playback processor for pre-recorded audio.

        Args:
            name: Unique name for the processor
            audio: Audio data (samples, channels)

        Returns:
            True if added successfully
        """
        try:
            processor = self.engine.make_playback_processor(name, audio)
            self._processors[name] = processor
            return True
        except Exception as e:
            print(f"Failed to add playback processor {name}: {e}")
            return False

    def set_midi(self, processor_name: str, notes: List[tuple]):
        """
        Set MIDI notes for a processor.

        Args:
            processor_name: Name of target processor
            notes: List of (note, velocity, start_time, duration) tuples
        """
        if processor_name not in self._processors:
            return

        processor = self._processors[processor_name]

        # Clear existing MIDI
        processor.clear_midi()

        # Add notes
        for note, velocity, start, duration in notes:
            processor.add_midi_note(note, velocity, start, duration)

    def build_graph(self, connections: List[tuple]):
        """
        Build the processor graph.

        Args:
            connections: List of (source, dest) processor name tuples
        """
        graph = []
        for source, dest in connections:
            if source in self._processors and dest in self._processors:
                graph.append((self._processors[source], []))

        # Simplified: connect all to output
        self.engine.load_graph([(p, []) for p in self._processors.values()])
        self._graph_built = True

    def render(self, duration: float) -> np.ndarray:
        """
        Render audio for specified duration.

        Args:
            duration: Duration in seconds

        Returns:
            Rendered audio array
        """
        if not self._graph_built:
            self.build_graph([])

        self.engine.render(duration)
        return self.engine.get_audio()

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio (adds as playback and renders)."""
        self.add_playback("input", audio)
        self.build_graph([])
        duration = len(audio) / sample_rate
        return self.render(duration)

    def get_latency_samples(self) -> int:
        """Get rendering latency."""
        return self.config.buffer_size


class JackAudioClient:
    """
    JACK Audio Connection Kit client.

    Provides real-time audio routing between applications with
    ultra-low latency.

    Example:
        client = JackAudioClient("DAiW_Voice")
        client.register_process_callback(my_callback)
        client.connect_to("system:playback_1")
        client.activate()
    """

    def __init__(self, client_name: str = "DAiW"):
        if not JACK_AVAILABLE:
            raise ImportError("JACK not available. Install with: pip install JACK-Client")

        self.client_name = client_name
        self.client = None
        self._input_ports = []
        self._output_ports = []
        self._process_callback: Optional[Callable] = None
        self._running = False

    def start(self) -> bool:
        """Start the JACK client."""
        try:
            self.client = jack.Client(self.client_name)

            # Register ports
            self._input_ports = [
                self.client.inports.register("input_L"),
                self.client.inports.register("input_R")
            ]
            self._output_ports = [
                self.client.outports.register("output_L"),
                self.client.outports.register("output_R")
            ]

            # Set process callback
            @self.client.set_process_callback
            def process(frames):
                if self._process_callback:
                    # Get input audio
                    input_L = self._input_ports[0].get_array()
                    input_R = self._input_ports[1].get_array()
                    input_audio = np.vstack([input_L, input_R])

                    # Process
                    output_audio = self._process_callback(input_audio, frames)

                    # Write output
                    if output_audio is not None:
                        self._output_ports[0].get_array()[:] = output_audio[0]
                        self._output_ports[1].get_array()[:] = output_audio[1]

            self.client.activate()
            self._running = True
            return True

        except Exception as e:
            print(f"Failed to start JACK client: {e}")
            return False

    def stop(self):
        """Stop the JACK client."""
        if self.client:
            self.client.deactivate()
            self.client.close()
            self._running = False

    def set_process_callback(self, callback: Callable[[np.ndarray, int], np.ndarray]):
        """
        Set the audio processing callback.

        Args:
            callback: Function(input_audio, num_frames) -> output_audio
        """
        self._process_callback = callback

    def connect_to(self, port_pattern: str):
        """Connect output to a JACK port pattern."""
        if not self.client:
            return

        ports = self.client.get_ports(port_pattern, is_input=True)
        for i, port in enumerate(ports[:2]):
            self.client.connect(self._output_ports[i], port)

    def connect_from(self, port_pattern: str):
        """Connect input from a JACK port pattern."""
        if not self.client:
            return

        ports = self.client.get_ports(port_pattern, is_output=True)
        for i, port in enumerate(ports[:2]):
            self.client.connect(port, self._input_ports[i])

    @property
    def sample_rate(self) -> int:
        """Get JACK sample rate."""
        return self.client.samplerate if self.client else 44100

    @property
    def buffer_size(self) -> int:
        """Get JACK buffer size."""
        return self.client.blocksize if self.client else 512


class WebAudioBridge:
    """
    WebSocket bridge to Web Audio API in browser.

    Enables browser-based audio processing and visualization
    that syncs with the Python DAiW backend.

    Example:
        bridge = WebAudioBridge()
        bridge.start_server(port=8765)
        bridge.send_audio(audio_data)
        bridge.send_command("setVolume", 0.8)
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._server = None
        self._clients = set()
        self._running = False

    def start_server(self):
        """Start the WebSocket server."""
        try:
            import asyncio
            import websockets

            async def handler(websocket, path):
                self._clients.add(websocket)
                try:
                    async for message in websocket:
                        await self._handle_message(message, websocket)
                finally:
                    self._clients.discard(websocket)

            async def serve():
                async with websockets.serve(handler, self.host, self.port):
                    await asyncio.Future()  # Run forever

            self._running = True
            asyncio.get_event_loop().run_until_complete(serve())

        except ImportError:
            print("websockets not available. Install with: pip install websockets")

    async def _handle_message(self, message: str, websocket):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            command = data.get("command")

            if command == "ready":
                await websocket.send(json.dumps({"status": "connected"}))
            elif command == "audioData":
                # Receive audio data from browser
                audio_base64 = data.get("audio")
                # Process...

        except json.JSONDecodeError:
            pass

    async def send_audio(self, audio: np.ndarray, sample_rate: int):
        """Send audio data to all connected browsers."""
        import base64

        # Convert to base64
        audio_bytes = audio.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()

        message = json.dumps({
            "type": "audio",
            "data": audio_b64,
            "sampleRate": sample_rate,
            "channels": audio.shape[0] if audio.ndim > 1 else 1
        })

        for client in self._clients:
            await client.send(message)

    async def send_command(self, command: str, **params):
        """Send a command to all connected browsers."""
        message = json.dumps({
            "type": "command",
            "command": command,
            "params": params
        })

        for client in self._clients:
            await client.send(message)


class UnifiedAudioProcessor:
    """
    Unified audio processor that can use any backend.

    Provides a consistent API regardless of which audio
    framework is being used underneath.

    Example:
        # Automatically selects best available backend
        processor = UnifiedAudioProcessor()

        # Or specify backend
        processor = UnifiedAudioProcessor(backend=AudioBackend.DAWDREAMER)

        # Process audio
        processor.apply_preset(EffectPreset.VOICE_WARM)
        output = processor.process(audio, 44100)
    """

    def __init__(self, config: Optional[AudioConfig] = None,
                 backend: Optional[AudioBackend] = None):
        self.config = config or AudioConfig()

        # Auto-select backend if not specified
        if backend is None:
            backend = self._detect_best_backend()

        self.backend = backend
        self._processor = self._create_processor()
        self._effect_chain: List[Dict[str, Any]] = []

    def _detect_best_backend(self) -> AudioBackend:
        """Detect the best available backend."""
        if PEDALBOARD_AVAILABLE:
            return AudioBackend.PEDALBOARD
        elif DAWDREAMER_AVAILABLE:
            return AudioBackend.DAWDREAMER
        elif JACK_AVAILABLE:
            return AudioBackend.JACK
        else:
            return AudioBackend.NATIVE

    def _create_processor(self) -> AudioProcessor:
        """Create the backend processor."""
        if self.backend == AudioBackend.PEDALBOARD:
            return PedalboardProcessor(self.config)
        elif self.backend == AudioBackend.DAWDREAMER:
            return DawDreamerProcessor(self.config)
        else:
            return NativeProcessor(self.config)

    def add_effect(self, effect_type: str, **kwargs) -> 'UnifiedAudioProcessor':
        """Add an effect to the chain."""
        self._effect_chain.append({"type": effect_type, **kwargs})

        if hasattr(self._processor, 'add_effect'):
            self._processor.add_effect(effect_type, **kwargs)

        return self

    def apply_preset(self, preset: EffectPreset):
        """Apply an effect preset."""
        self._effect_chain = preset.effects.copy()

        if hasattr(self._processor, 'apply_preset'):
            self._processor.apply_preset(preset)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through the effect chain."""
        return self._processor.process(audio, sample_rate)

    def process_file(self, input_path: str, output_path: str):
        """Process an audio file."""
        if hasattr(self._processor, 'process_file'):
            self._processor.process_file(input_path, output_path)
        elif SOUNDFILE_AVAILABLE:
            audio, sr = sf.read(input_path)
            processed = self.process(audio.T, sr)
            sf.write(output_path, processed.T, sr)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "backend": self.backend.value,
            "available_backends": {
                "pedalboard": PEDALBOARD_AVAILABLE,
                "dawdreamer": DAWDREAMER_AVAILABLE,
                "jack": JACK_AVAILABLE,
            },
            "sample_rate": self.config.sample_rate,
            "buffer_size": self.config.buffer_size,
            "latency_samples": self._processor.get_latency_samples()
        }


class NativeProcessor(AudioProcessor):
    """
    Native numpy-based audio processor (fallback).

    Provides basic audio processing when no specialized
    backend is available.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._effects: List[Callable] = []

    def add_effect(self, effect_type: str, **kwargs) -> 'NativeProcessor':
        """Add a native effect."""
        effect = self._create_native_effect(effect_type, **kwargs)
        if effect:
            self._effects.append(effect)
        return self

    def _create_native_effect(self, effect_type: str, **kwargs) -> Optional[Callable]:
        """Create a native numpy effect."""
        if effect_type == "gain":
            gain_linear = 10 ** (kwargs.get("gain_db", 0) / 20)
            return lambda x: x * gain_linear

        elif effect_type == "highpass":
            # Simple first-order highpass
            cutoff = kwargs.get("cutoff_hz", 80)
            return lambda x: self._simple_highpass(x, cutoff)

        elif effect_type == "lowpass":
            cutoff = kwargs.get("cutoff_hz", 8000)
            return lambda x: self._simple_lowpass(x, cutoff)

        return None

    def _simple_highpass(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Simple first-order highpass filter."""
        from scipy import signal
        nyquist = self.config.sample_rate / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        b, a = signal.butter(1, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def _simple_lowpass(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Simple first-order lowpass filter."""
        from scipy import signal
        nyquist = self.config.sample_rate / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        b, a = signal.butter(1, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through native effects."""
        for effect in self._effects:
            audio = effect(audio)
        return audio

    def get_latency_samples(self) -> int:
        return 0


# Integration with DAiW voice synthesis
def create_voice_effects_chain(preset: str = "clean") -> UnifiedAudioProcessor:
    """
    Create an effects chain optimized for voice synthesis.

    Args:
        preset: One of "clean", "warm", "robotic", "ethereal"

    Returns:
        Configured UnifiedAudioProcessor
    """
    processor = UnifiedAudioProcessor()

    presets = {
        "clean": EffectPreset.VOICE_CLEAN,
        "warm": EffectPreset.VOICE_WARM,
        "robotic": EffectPreset.VOICE_ROBOTIC,
        "ethereal": EffectPreset.VOICE_ETHEREAL,
    }

    preset_obj = presets.get(preset, EffectPreset.VOICE_CLEAN)
    processor.apply_preset(preset_obj)

    return processor


def get_available_frameworks() -> Dict[str, bool]:
    """Get availability of all audio frameworks."""
    return {
        "pedalboard": PEDALBOARD_AVAILABLE,
        "dawdreamer": DAWDREAMER_AVAILABLE,
        "jack": JACK_AVAILABLE,
        "soundfile": SOUNDFILE_AVAILABLE,
    }
