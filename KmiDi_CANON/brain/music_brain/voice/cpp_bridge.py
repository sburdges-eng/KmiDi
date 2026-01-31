"""
Python → C++ Voice Synthesis Bridge

This module provides a unified pipeline for sending voice models and phonemes
from Python to the C++ JUCE VoiceProcessor for real-time synthesis.

Communication is via OSC (Open Sound Control) protocol.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict
import json
import time
import threading
from enum import Enum

try:
    from pythonosc import udp_client, dispatcher, osc_server
    from pythonosc.osc_message_builder import OscMessageBuilder
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

from music_brain.vocal.parrot import (
    VoiceModel, VoiceCharacteristics, FormantData, VowelType,
    ParrotVocalSynthesizer, ParrotConfig
)
from music_brain.vocal.phonemes import Phoneme, PhonemeType, text_to_phonemes


# OSC Address patterns for voice synthesis
class OSCAddresses:
    """OSC address patterns for C++ VoiceProcessor communication"""
    # Outgoing (Python → C++)
    LOAD_VOICE_MODEL = "/voice/model/load"      # Send voice model JSON
    SPEAK_TEXT = "/voice/speak"                  # Send text to synthesize
    QUEUE_PHONEME = "/voice/phoneme"             # Queue single phoneme
    SET_VOWEL = "/voice/vowel"                   # Set current vowel (0-5)
    SET_PITCH = "/voice/pitch"                   # Set pitch in Hz
    NOTE_ON = "/voice/note/on"                   # MIDI note on (note, velocity)
    NOTE_OFF = "/voice/note/off"                 # MIDI note off
    SET_FORMANT_SHIFT = "/voice/formant/shift"  # Formant shift multiplier
    SET_BREATHINESS = "/voice/breathiness"       # Breathiness 0.0-1.0
    SET_VIBRATO = "/voice/vibrato"               # Vibrato amount

    # Incoming (C++ → Python)
    STATUS = "/voice/status"                     # Status updates
    PHONEME_COMPLETE = "/voice/phoneme/complete" # Phoneme finished
    ERROR = "/voice/error"                       # Error messages


@dataclass
class CppBridgeConfig:
    """Configuration for C++ bridge connection"""
    cpp_host: str = "127.0.0.1"
    cpp_port: int = 9000           # Send to C++
    python_port: int = 9001        # Receive from C++
    auto_connect: bool = True
    timeout: float = 5.0           # Connection timeout


class VoiceCppBridge:
    """
    Bridge between Python voice synthesis and C++ JUCE VoiceProcessor.

    This class handles:
    1. Sending learned voice models to C++ for real-time rendering
    2. Converting text to phonemes and streaming to C++
    3. Real-time control of vowels, pitch, formants
    4. Receiving status updates from C++

    Example usage:
        from music_brain.voice.cpp_bridge import VoiceCppBridge
        from music_brain.vocal.parrot import ParrotVocalSynthesizer

        # Train a voice model
        parrot = ParrotVocalSynthesizer()
        model = parrot.train_parrot("voice_sample.wav", "my_voice")

        # Connect to C++ and send model
        bridge = VoiceCppBridge()
        bridge.connect()
        bridge.load_voice_model(model)

        # Speak text
        bridge.speak_text("Hello world")

        # Real-time vowel control
        bridge.set_vowel(VowelType.A)
        bridge.set_pitch(200.0)
        bridge.note_on(60, 0.8)
    """

    def __init__(self, config: Optional[CppBridgeConfig] = None):
        """Initialize the C++ bridge."""
        if not OSC_AVAILABLE:
            raise ImportError(
                "python-osc required for C++ bridge. "
                "Install with: pip install python-osc"
            )

        self.config = config or CppBridgeConfig()
        self.client: Optional[udp_client.SimpleUDPClient] = None
        self.server: Optional[osc_server.ThreadingOSCUDPServer] = None
        self.server_thread: Optional[threading.Thread] = None

        self.connected = False
        self.current_model: Optional[VoiceModel] = None

        # Callbacks
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_phoneme_complete: Optional[Callable[[int], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None

        if self.config.auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Connect to C++ VoiceProcessor via OSC.

        Returns:
            True if connection successful
        """
        try:
            # Create OSC client (send to C++)
            self.client = udp_client.SimpleUDPClient(
                self.config.cpp_host,
                self.config.cpp_port
            )

            # Create OSC server (receive from C++)
            disp = dispatcher.Dispatcher()
            disp.map(OSCAddresses.STATUS, self._handle_status)
            disp.map(OSCAddresses.PHONEME_COMPLETE, self._handle_phoneme_complete)
            disp.map(OSCAddresses.ERROR, self._handle_error)

            self.server = osc_server.ThreadingOSCUDPServer(
                (self.config.cpp_host, self.config.python_port),
                disp
            )

            # Start server in background thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()

            self.connected = True
            print(f"Connected to C++ VoiceProcessor at {self.config.cpp_host}:{self.config.cpp_port}")
            return True

        except Exception as e:
            print(f"Failed to connect to C++ VoiceProcessor: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from C++ VoiceProcessor."""
        if self.server:
            self.server.shutdown()
            self.server = None

        if self.server_thread:
            self.server_thread.join(timeout=1.0)
            self.server_thread = None

        self.client = None
        self.connected = False
        print("Disconnected from C++ VoiceProcessor")

    def load_voice_model(self, model: VoiceModel) -> bool:
        """
        Send a trained voice model to C++ for synthesis.

        Args:
            model: Trained VoiceModel from ParrotVocalSynthesizer

        Returns:
            True if model sent successfully
        """
        if not self.connected or not self.client:
            print("Not connected to C++ VoiceProcessor")
            return False

        try:
            # Convert model to JSON
            model_json = self._voice_model_to_json(model)

            # Send via OSC
            self.client.send_message(OSCAddresses.LOAD_VOICE_MODEL, model_json)

            self.current_model = model
            print(f"Loaded voice model '{model.name}' to C++")
            return True

        except Exception as e:
            print(f"Failed to load voice model: {e}")
            return False

    def speak_text(self, text: str, pitch: Optional[float] = None):
        """
        Send text to C++ for synthesis.

        Args:
            text: Text to synthesize
            pitch: Optional pitch override in Hz
        """
        if not self.connected or not self.client:
            print("Not connected to C++ VoiceProcessor")
            return

        # Convert text to phonemes
        phonemes = text_to_phonemes(text)

        # Send each phoneme
        for i, phoneme in enumerate(phonemes):
            self._send_phoneme(phoneme, i, pitch)

        print(f"Queued {len(phonemes)} phonemes for synthesis")

    def set_vowel(self, vowel: VowelType):
        """Set current vowel for real-time synthesis."""
        if not self.connected or not self.client:
            return

        vowel_index = {
            VowelType.A: 0,
            VowelType.E: 1,
            VowelType.I: 2,
            VowelType.O: 3,
            VowelType.U: 4,
            VowelType.SCHWA: 5,
        }.get(vowel, 0)

        self.client.send_message(OSCAddresses.SET_VOWEL, vowel_index)

    def set_pitch(self, pitch_hz: float):
        """Set pitch in Hz for real-time synthesis."""
        if not self.connected or not self.client:
            return

        self.client.send_message(OSCAddresses.SET_PITCH, pitch_hz)

    def note_on(self, midi_note: int, velocity: float = 0.8):
        """
        Trigger note on (MIDI-style).

        Args:
            midi_note: MIDI note number (60 = middle C)
            velocity: Velocity 0.0-1.0
        """
        if not self.connected or not self.client:
            return

        self.client.send_message(OSCAddresses.NOTE_ON, [midi_note, velocity])

    def note_off(self):
        """Trigger note off."""
        if not self.connected or not self.client:
            return

        self.client.send_message(OSCAddresses.NOTE_OFF, 0)

    def set_formant_shift(self, shift: float):
        """
        Set formant shift multiplier.

        Args:
            shift: Formant frequency multiplier (1.0 = no shift)
                   < 1.0 = lower formants (darker, larger vocal tract)
                   > 1.0 = higher formants (brighter, smaller vocal tract)
        """
        if not self.connected or not self.client:
            return

        self.client.send_message(OSCAddresses.SET_FORMANT_SHIFT, shift)

    def set_breathiness(self, amount: float):
        """
        Set breathiness amount.

        Args:
            amount: Breathiness 0.0-1.0
        """
        if not self.connected or not self.client:
            return

        self.client.send_message(OSCAddresses.SET_BREATHINESS, amount)

    def set_vibrato(self, amount: float):
        """
        Set vibrato intensity.

        Args:
            amount: Vibrato amount 0.0-1.0
        """
        if not self.connected or not self.client:
            return

        self.client.send_message(OSCAddresses.SET_VIBRATO, amount)

    # Callbacks
    def on_status(self, callback: Callable[[str], None]):
        """Set callback for status updates from C++."""
        self._on_status = callback

    def on_phoneme_complete(self, callback: Callable[[int], None]):
        """Set callback for phoneme completion."""
        self._on_phoneme_complete = callback

    def on_error(self, callback: Callable[[str], None]):
        """Set callback for errors from C++."""
        self._on_error = callback

    # Private methods
    def _send_phoneme(self, phoneme: Phoneme, index: int, pitch_override: Optional[float] = None):
        """Send a single phoneme to C++."""
        if not self.client:
            return

        # Map phoneme to vowel type
        vowel_map = {
            'ɑ': 0, 'a': 0,  # A
            'ɛ': 1, 'e': 1,  # E
            'ɪ': 2, 'i': 2,  # I
            'ɔ': 3, 'o': 3,  # O
            'ʊ': 4, 'u': 4,  # U
            'ə': 5,          # SCHWA
        }

        vowel_index = vowel_map.get(phoneme.symbol.lower(), 5)
        is_consonant = 1 if phoneme.phoneme_type == PhonemeType.CONSONANT else 0
        pitch = pitch_override if pitch_override else (phoneme.pitch or 200.0)

        # Send: [index, vowel_index, duration, pitch, stress, is_consonant]
        self.client.send_message(
            OSCAddresses.QUEUE_PHONEME,
            [index, vowel_index, phoneme.duration, pitch, phoneme.stress, is_consonant]
        )

    def _voice_model_to_json(self, model: VoiceModel) -> str:
        """Convert VoiceModel to JSON for C++."""
        char = model.characteristics

        # Build JSON structure matching C++ parser expectations
        data = {
            "name": model.name,
            "average_pitch": char.average_pitch,
            "pitch_range": list(char.pitch_range),
            "vibrato_rate": char.vibrato_rate,
            "vibrato_depth": char.vibrato_depth,
            "spectral_centroid_mean": char.spectral_centroid_mean,
            "spectral_rolloff_mean": char.spectral_rolloff_mean,
            "spectral_bandwidth_mean": char.spectral_bandwidth_mean,
            "jitter": char.jitter,
            "shimmer": char.shimmer,
            "breathiness": char.breathiness,
            "nasality": char.nasality,
            "attack_time": char.attack_time,
            "release_time": char.release_time,
            "vowel_formants": {}
        }

        # Add vowel formants
        vowel_names = {
            VowelType.A: "a",
            VowelType.E: "e",
            VowelType.I: "i",
            VowelType.O: "o",
            VowelType.U: "u",
            VowelType.SCHWA: "ə",
        }

        for vowel_type, formants_list in char.vowel_formants.items():
            if formants_list:
                vowel_name = vowel_names.get(vowel_type, "a")
                data["vowel_formants"][vowel_name] = [
                    {
                        "f1": f.f1,
                        "f2": f.f2,
                        "f3": f.f3,
                        "confidence": f.confidence
                    }
                    for f in formants_list
                ]

        return json.dumps(data)

    def _handle_status(self, address: str, *args):
        """Handle status message from C++."""
        if self._on_status and args:
            self._on_status(str(args[0]))

    def _handle_phoneme_complete(self, address: str, *args):
        """Handle phoneme complete message from C++."""
        if self._on_phoneme_complete and args:
            self._on_phoneme_complete(int(args[0]))

    def _handle_error(self, address: str, *args):
        """Handle error message from C++."""
        if self._on_error and args:
            self._on_error(str(args[0]))


class VoiceSynthesisPipeline:
    """
    Complete voice synthesis pipeline integrating Python analysis
    with C++ real-time rendering.

    This is the main entry point for the unified Python→C++ voice system.

    Example usage:
        from music_brain.voice.cpp_bridge import VoiceSynthesisPipeline

        # Create pipeline
        pipeline = VoiceSynthesisPipeline()

        # Train on voice samples
        pipeline.train_voice("samples/voice1.wav", "my_voice")
        pipeline.train_voice("samples/voice2.wav", "my_voice")  # More training

        # Connect to C++ and synthesize
        pipeline.connect_cpp()
        pipeline.speak("Hello, this is my synthesized voice!")

        # Real-time control
        pipeline.play_vowel_sequence(["a", "e", "i", "o", "u"], tempo=120)
    """

    def __init__(self, parrot_config: Optional[ParrotConfig] = None,
                 bridge_config: Optional[CppBridgeConfig] = None):
        """
        Initialize the pipeline.

        Args:
            parrot_config: Configuration for voice analysis
            bridge_config: Configuration for C++ connection
        """
        self.parrot = ParrotVocalSynthesizer(parrot_config or ParrotConfig())
        self.bridge = VoiceCppBridge(bridge_config or CppBridgeConfig(auto_connect=False))

        self.current_voice: Optional[str] = None

    def train_voice(self, audio_file: str, voice_name: str) -> VoiceModel:
        """
        Train voice model from audio file.

        Args:
            audio_file: Path to audio file
            voice_name: Name for the voice

        Returns:
            Trained VoiceModel
        """
        print(f"Training voice '{voice_name}' from {audio_file}...")
        model = self.parrot.train_parrot(audio_file, voice_name)
        self.current_voice = voice_name

        print(f"  Exposure time: {model.characteristics.exposure_time:.1f}s")
        print(f"  Confidence: {model.characteristics.confidence:.2%}")
        print(f"  Average pitch: {model.characteristics.average_pitch:.1f} Hz")

        return model

    def train_voice_batch(self, audio_files: List[str], voice_name: str) -> VoiceModel:
        """
        Train voice model from multiple audio files.

        Args:
            audio_files: List of audio file paths
            voice_name: Name for the voice

        Returns:
            Trained VoiceModel
        """
        return self.parrot.train_parrot_batch(audio_files, voice_name)

    def connect_cpp(self) -> bool:
        """
        Connect to C++ VoiceProcessor.

        Returns:
            True if connected successfully
        """
        success = self.bridge.connect()

        if success and self.current_voice:
            # Send current voice model to C++
            model = self.parrot.voice_models.get(self.current_voice)
            if model:
                self.bridge.load_voice_model(model)

        return success

    def disconnect_cpp(self):
        """Disconnect from C++ VoiceProcessor."""
        self.bridge.disconnect()

    def select_voice(self, voice_name: str) -> bool:
        """
        Select a trained voice for synthesis.

        Args:
            voice_name: Name of trained voice

        Returns:
            True if voice found and selected
        """
        if voice_name not in self.parrot.voice_models:
            print(f"Voice '{voice_name}' not found")
            return False

        self.current_voice = voice_name
        model = self.parrot.voice_models[voice_name]

        if self.bridge.connected:
            self.bridge.load_voice_model(model)

        return True

    def speak(self, text: str, pitch: Optional[float] = None):
        """
        Synthesize speech from text.

        Args:
            text: Text to speak
            pitch: Optional pitch override in Hz
        """
        if not self.bridge.connected:
            print("Not connected to C++. Call connect_cpp() first.")
            return

        self.bridge.speak_text(text, pitch)

    def play_vowel_sequence(self, vowels: List[str], tempo: float = 120.0,
                           note_duration: float = 0.5):
        """
        Play a sequence of vowels in rhythm.

        Args:
            vowels: List of vowel characters ('a', 'e', 'i', 'o', 'u')
            tempo: Tempo in BPM
            note_duration: Duration of each note in beats
        """
        if not self.bridge.connected:
            print("Not connected to C++. Call connect_cpp() first.")
            return

        vowel_map = {
            'a': VowelType.A,
            'e': VowelType.E,
            'i': VowelType.I,
            'o': VowelType.O,
            'u': VowelType.U,
        }

        beat_duration = 60.0 / tempo
        note_time = beat_duration * note_duration

        # Start first note
        self.bridge.note_on(60, 0.8)

        for i, vowel_char in enumerate(vowels):
            vowel = vowel_map.get(vowel_char.lower(), VowelType.SCHWA)
            self.bridge.set_vowel(vowel)
            time.sleep(note_time)

        self.bridge.note_off()

    def real_time_control(self):
        """
        Enter real-time control mode.
        Returns a controller object for live manipulation.
        """
        return RealTimeVoiceController(self.bridge)

    def list_voices(self) -> List[str]:
        """List all trained voice models."""
        return self.parrot.list_voices()

    def get_voice_info(self, voice_name: str) -> Dict[str, Any]:
        """Get information about a voice model."""
        return self.parrot.get_voice_info(voice_name)

    def blend_voices(self, voice1: str, voice2: str, ratio: float = 0.5,
                    output_name: Optional[str] = None) -> VoiceModel:
        """
        Blend two voice models together.

        Args:
            voice1: First voice name
            voice2: Second voice name
            ratio: Blend ratio (0.0 = voice1, 1.0 = voice2)
            output_name: Name for blended voice

        Returns:
            Blended VoiceModel
        """
        return self.parrot.blend_voices(voice1, voice2, ratio, output_name)


class RealTimeVoiceController:
    """
    Controller for real-time voice manipulation.

    Example:
        controller = pipeline.real_time_control()

        # Manual vowel control
        controller.set_vowel('a')
        controller.set_pitch(220)
        controller.trigger()
        time.sleep(0.5)
        controller.set_vowel('e')
        time.sleep(0.5)
        controller.release()

        # XY pad control (formant + pitch)
        controller.xy_control(x=0.5, y=0.5)  # x=formant, y=pitch
    """

    def __init__(self, bridge: VoiceCppBridge):
        self.bridge = bridge
        self.current_pitch = 200.0
        self.is_playing = False

    def set_vowel(self, vowel: str):
        """Set current vowel ('a', 'e', 'i', 'o', 'u')."""
        vowel_map = {
            'a': VowelType.A,
            'e': VowelType.E,
            'i': VowelType.I,
            'o': VowelType.O,
            'u': VowelType.U,
        }
        vowel_type = vowel_map.get(vowel.lower(), VowelType.SCHWA)
        self.bridge.set_vowel(vowel_type)

    def set_pitch(self, pitch_hz: float):
        """Set pitch in Hz."""
        self.current_pitch = pitch_hz
        self.bridge.set_pitch(pitch_hz)

    def set_midi_note(self, note: int):
        """Set pitch from MIDI note number."""
        pitch = 440.0 * (2.0 ** ((note - 69) / 12.0))
        self.set_pitch(pitch)

    def trigger(self, velocity: float = 0.8):
        """Trigger note on."""
        self.bridge.note_on(60, velocity)
        self.is_playing = True

    def release(self):
        """Release note."""
        self.bridge.note_off()
        self.is_playing = False

    def set_formant_shift(self, shift: float):
        """Set formant shift (0.5 to 2.0 typical range)."""
        self.bridge.set_formant_shift(shift)

    def set_breathiness(self, amount: float):
        """Set breathiness (0.0 to 1.0)."""
        self.bridge.set_breathiness(amount)

    def set_vibrato(self, amount: float):
        """Set vibrato intensity (0.0 to 1.0)."""
        self.bridge.set_vibrato(amount)

    def xy_control(self, x: float, y: float,
                   pitch_range: tuple = (100.0, 400.0),
                   formant_range: tuple = (0.7, 1.4)):
        """
        XY pad control for pitch and formants.

        Args:
            x: X position 0.0-1.0 (controls formant shift)
            y: Y position 0.0-1.0 (controls pitch)
            pitch_range: (min_hz, max_hz) for pitch mapping
            formant_range: (min, max) for formant shift mapping
        """
        # Map X to formant shift
        formant_shift = formant_range[0] + x * (formant_range[1] - formant_range[0])
        self.set_formant_shift(formant_shift)

        # Map Y to pitch
        pitch = pitch_range[0] + y * (pitch_range[1] - pitch_range[0])
        self.set_pitch(pitch)


# Convenience functions
def create_voice_pipeline() -> VoiceSynthesisPipeline:
    """Create a new voice synthesis pipeline with default settings."""
    return VoiceSynthesisPipeline()


def quick_speak(text: str, voice_file: Optional[str] = None):
    """
    Quick function to synthesize speech.

    Args:
        text: Text to speak
        voice_file: Optional audio file to clone voice from
    """
    pipeline = VoiceSynthesisPipeline()

    if voice_file:
        pipeline.train_voice(voice_file, "quick_voice")

    if pipeline.connect_cpp():
        pipeline.speak(text)
        time.sleep(len(text) * 0.1)  # Rough estimate of duration
        pipeline.disconnect_cpp()
    else:
        print("Failed to connect to C++ synthesizer")
