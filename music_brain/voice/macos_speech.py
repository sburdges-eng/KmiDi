"""
macOS Native Speech Synthesis Integration

Provides integration with macOS AVSpeechSynthesizer and NSSpeechSynthesizer
for high-quality native text-to-speech as an alternative backend.

This module can be used standalone or as a fallback when the C++ JUCE
synthesizer is not available.
"""

import subprocess
import tempfile
import os
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json

try:
    # Try to import PyObjC for direct macOS API access
    import objc
    from Foundation import NSObject
    from AppKit import NSSpeechSynthesizer
    PYOBJC_AVAILABLE = True
except ImportError:
    PYOBJC_AVAILABLE = False

try:
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class MacOSVoice(Enum):
    """Common macOS system voices"""
    # Premium voices (macOS 10.14+)
    SAMANTHA = "com.apple.speech.synthesis.voice.samantha"
    ALEX = "com.apple.speech.synthesis.voice.Alex"
    KAREN = "com.apple.speech.synthesis.voice.karen"
    DANIEL = "com.apple.speech.synthesis.voice.daniel"
    MOIRA = "com.apple.speech.synthesis.voice.moira"
    TESSA = "com.apple.speech.synthesis.voice.tessa"

    # Enhanced voices (macOS 12+)
    SIRI_FEMALE = "com.apple.voice.premium.en-US.Zoe"
    SIRI_MALE = "com.apple.voice.premium.en-US.Nate"

    # Default
    DEFAULT = ""


@dataclass
class SpeechConfig:
    """Configuration for macOS speech synthesis"""
    voice: str = ""                    # Voice identifier (empty = system default)
    rate: float = 175.0                # Words per minute (90-720)
    pitch: float = 1.0                 # Pitch multiplier (0.5-2.0)
    volume: float = 1.0                # Volume (0.0-1.0)
    output_file: Optional[str] = None  # If set, save audio to file


class MacOSSpeechSynthesizer:
    """
    macOS native speech synthesizer wrapper.

    Provides access to macOS text-to-speech capabilities with support for:
    - Multiple system voices
    - Rate, pitch, and volume control
    - Audio file output (AIFF format)
    - Voice enumeration

    Example usage:
        synth = MacOSSpeechSynthesizer()

        # List available voices
        voices = synth.list_voices()
        print(voices)

        # Speak text
        synth.speak("Hello, world!")

        # Save to file
        synth.speak_to_file("Hello, world!", "output.aiff")

        # Use specific voice
        synth.set_voice(MacOSVoice.SAMANTHA)
        synth.speak("This is Samantha speaking.")
    """

    def __init__(self, config: Optional[SpeechConfig] = None):
        """Initialize the macOS speech synthesizer."""
        self.config = config or SpeechConfig()
        self._synthesizer = None
        self._speaking_callback: Optional[Callable[[str], None]] = None

        if PYOBJC_AVAILABLE:
            self._init_native()
        else:
            print("PyObjC not available, using 'say' command fallback")

    def _init_native(self):
        """Initialize native NSSpeechSynthesizer."""
        try:
            if self.config.voice:
                self._synthesizer = NSSpeechSynthesizer.alloc().initWithVoice_(
                    self.config.voice
                )
            else:
                self._synthesizer = NSSpeechSynthesizer.alloc().init()

            if self._synthesizer:
                self._synthesizer.setRate_(self.config.rate)
                self._synthesizer.setVolume_(self.config.volume)
        except Exception as e:
            print(f"Failed to initialize NSSpeechSynthesizer: {e}")
            self._synthesizer = None

    def speak(self, text: str, blocking: bool = True):
        """
        Speak text using macOS TTS.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if PYOBJC_AVAILABLE and self._synthesizer:
            self._speak_native(text, blocking)
        else:
            self._speak_say_command(text, blocking)

    def _speak_native(self, text: str, blocking: bool):
        """Speak using native NSSpeechSynthesizer."""
        self._synthesizer.startSpeakingString_(text)

        if blocking:
            # Wait for speech to complete
            import time
            while self._synthesizer.isSpeaking():
                time.sleep(0.1)

    def _speak_say_command(self, text: str, blocking: bool):
        """Speak using 'say' command (fallback)."""
        cmd = ["say"]

        if self.config.voice:
            # Extract voice name from identifier
            voice_name = self.config.voice.split(".")[-1]
            cmd.extend(["-v", voice_name])

        if self.config.rate != 175.0:
            cmd.extend(["-r", str(int(self.config.rate))])

        cmd.append(text)

        if blocking:
            subprocess.run(cmd, check=True)
        else:
            subprocess.Popen(cmd)

    def speak_to_file(self, text: str, output_path: str) -> bool:
        """
        Synthesize text to audio file.

        Args:
            text: Text to synthesize
            output_path: Output file path (AIFF format)

        Returns:
            True if successful
        """
        if PYOBJC_AVAILABLE and self._synthesizer:
            return self._speak_to_file_native(text, output_path)
        else:
            return self._speak_to_file_say(text, output_path)

    def _speak_to_file_native(self, text: str, output_path: str) -> bool:
        """Save speech to file using native API."""
        try:
            from Foundation import NSURL
            url = NSURL.fileURLWithPath_(output_path)
            self._synthesizer.startSpeakingString_toURL_(text, url)

            # Wait for completion
            import time
            while self._synthesizer.isSpeaking():
                time.sleep(0.1)

            return os.path.exists(output_path)
        except Exception as e:
            print(f"Failed to save speech to file: {e}")
            return False

    def _speak_to_file_say(self, text: str, output_path: str) -> bool:
        """Save speech to file using 'say' command."""
        cmd = ["say", "-o", output_path]

        if self.config.voice:
            voice_name = self.config.voice.split(".")[-1]
            cmd.extend(["-v", voice_name])

        if self.config.rate != 175.0:
            cmd.extend(["-r", str(int(self.config.rate))])

        cmd.append(text)

        try:
            subprocess.run(cmd, check=True)
            return os.path.exists(output_path)
        except subprocess.CalledProcessError as e:
            print(f"'say' command failed: {e}")
            return False

    def set_voice(self, voice: MacOSVoice):
        """Set the speech voice."""
        self.config.voice = voice.value
        if self._synthesizer:
            self._synthesizer.setVoice_(voice.value)

    def set_voice_by_name(self, voice_name: str):
        """Set voice by name (e.g., 'Samantha', 'Alex')."""
        # Find voice identifier from name
        voices = self.list_voices()
        for voice in voices:
            if voice.get('name', '').lower() == voice_name.lower():
                self.config.voice = voice.get('identifier', '')
                if self._synthesizer:
                    self._synthesizer.setVoice_(self.config.voice)
                return True
        return False

    def set_rate(self, rate: float):
        """Set speech rate (words per minute, 90-720)."""
        self.config.rate = max(90.0, min(720.0, rate))
        if self._synthesizer:
            self._synthesizer.setRate_(self.config.rate)

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0)."""
        self.config.volume = max(0.0, min(1.0, volume))
        if self._synthesizer:
            self._synthesizer.setVolume_(self.config.volume)

    def stop(self):
        """Stop current speech."""
        if self._synthesizer:
            self._synthesizer.stopSpeaking()

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        if self._synthesizer:
            return self._synthesizer.isSpeaking()
        return False

    def list_voices(self) -> List[Dict[str, Any]]:
        """
        List all available system voices.

        Returns:
            List of voice dictionaries with 'name', 'identifier', 'language'
        """
        if PYOBJC_AVAILABLE:
            return self._list_voices_native()
        else:
            return self._list_voices_say()

    def _list_voices_native(self) -> List[Dict[str, Any]]:
        """List voices using native API."""
        try:
            voices = NSSpeechSynthesizer.availableVoices()
            result = []

            for voice_id in voices:
                attrs = NSSpeechSynthesizer.attributesForVoice_(voice_id)
                result.append({
                    'identifier': str(voice_id),
                    'name': str(attrs.get('VoiceName', '')),
                    'language': str(attrs.get('VoiceLocaleIdentifier', '')),
                    'gender': str(attrs.get('VoiceGender', '')),
                    'age': int(attrs.get('VoiceAge', 0)),
                })

            return result
        except Exception as e:
            print(f"Failed to list voices: {e}")
            return []

    def _list_voices_say(self) -> List[Dict[str, Any]]:
        """List voices using 'say -v ?' command."""
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                check=True
            )

            voices = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    # Parse: "Voice Name    language_code    # Description"
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        lang = parts[1] if len(parts) > 1 else ""
                        voices.append({
                            'name': name,
                            'identifier': f"com.apple.speech.synthesis.voice.{name.lower()}",
                            'language': lang,
                        })

            return voices
        except subprocess.CalledProcessError:
            return []


class MacOSVoiceCloner:
    """
    Voice cloning utility using macOS speech synthesis.

    This class can generate training data by having macOS speak text
    and recording the output for use with the Parrot voice learning system.
    """

    def __init__(self):
        self.synth = MacOSSpeechSynthesizer()

    def generate_training_samples(
        self,
        voice: MacOSVoice,
        output_dir: str,
        texts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate training audio samples from a macOS voice.

        Args:
            voice: macOS voice to use
            output_dir: Directory to save audio files
            texts: Optional custom texts (uses defaults if None)

        Returns:
            List of generated audio file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        if texts is None:
            texts = self._get_default_training_texts()

        self.synth.set_voice(voice)
        output_files = []

        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"sample_{i:03d}.aiff")
            if self.synth.speak_to_file(text, output_path):
                output_files.append(output_path)
                print(f"Generated: {output_path}")

        return output_files

    def _get_default_training_texts(self) -> List[str]:
        """Get default texts for voice training (phonetically diverse)."""
        return [
            # Vowel-heavy sentences
            "The rain in Spain falls mainly on the plain.",
            "How now brown cow, out on the town.",
            "Peter Piper picked a peck of pickled peppers.",
            "She sells seashells by the seashore.",

            # All vowels
            "Father bought hot coffee in the shop.",
            "The eagle sees the trees and leaves.",
            "I like to ride my bike at night.",
            "Go home and throw the stone alone.",
            "The moon illuminates the blue lagoon.",

            # Consonant variety
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "Sphinx of black quartz, judge my vow.",

            # Emotional range
            "What wonderful weather we're having today!",
            "Oh no, I can't believe this happened.",
            "Hmm, let me think about that for a moment.",
            "Yes! That's exactly what I was looking for!",

            # Numbers and technical
            "The temperature is seventy-two degrees Fahrenheit.",
            "Please call the number eight hundred five five five one two three four.",
            "The meeting is scheduled for three thirty PM.",
        ]

    def convert_to_wav(self, aiff_path: str, wav_path: Optional[str] = None) -> Optional[str]:
        """
        Convert AIFF to WAV format for Parrot training.

        Args:
            aiff_path: Input AIFF file path
            wav_path: Output WAV path (auto-generated if None)

        Returns:
            Output WAV path if successful
        """
        if not AUDIO_AVAILABLE:
            print("soundfile not available for audio conversion")
            return None

        if wav_path is None:
            wav_path = aiff_path.replace('.aiff', '.wav')

        try:
            # Read AIFF
            data, sr = sf.read(aiff_path)

            # Write WAV
            sf.write(wav_path, data, sr)
            return wav_path
        except Exception as e:
            print(f"Failed to convert audio: {e}")
            return None


# Integration with DAiW voice system
def create_macos_voice_backend(voice: MacOSVoice = MacOSVoice.DEFAULT) -> MacOSSpeechSynthesizer:
    """
    Create a macOS speech backend for the DAiW voice system.

    Args:
        voice: macOS voice to use

    Returns:
        Configured MacOSSpeechSynthesizer
    """
    config = SpeechConfig(voice=voice.value)
    synth = MacOSSpeechSynthesizer(config)
    return synth


def hybrid_speak(text: str, cpp_bridge=None, macos_fallback: bool = True):
    """
    Speak text using C++ synthesizer with macOS fallback.

    Args:
        text: Text to speak
        cpp_bridge: Optional VoiceCppBridge instance
        macos_fallback: Use macOS TTS if C++ not available
    """
    if cpp_bridge and cpp_bridge.connected:
        cpp_bridge.speak_text(text)
    elif macos_fallback:
        synth = MacOSSpeechSynthesizer()
        synth.speak(text)
    else:
        print("No speech backend available")


# Convenience functions
def say(text: str, voice: Optional[str] = None, rate: float = 175.0):
    """
    Quick function to speak text.

    Args:
        text: Text to speak
        voice: Optional voice name
        rate: Speech rate (words per minute)
    """
    config = SpeechConfig(rate=rate)
    synth = MacOSSpeechSynthesizer(config)

    if voice:
        synth.set_voice_by_name(voice)

    synth.speak(text)


def list_macos_voices() -> List[str]:
    """List available macOS voice names."""
    synth = MacOSSpeechSynthesizer()
    voices = synth.list_voices()
    return [v.get('name', '') for v in voices]


def generate_voice_samples(voice_name: str, output_dir: str) -> List[str]:
    """
    Generate training samples from a macOS voice.

    Args:
        voice_name: Name of macOS voice
        output_dir: Output directory for samples

    Returns:
        List of generated WAV file paths
    """
    cloner = MacOSVoiceCloner()

    # Find voice
    voices = cloner.synth.list_voices()
    voice = MacOSVoice.DEFAULT
    for v in voices:
        if v.get('name', '').lower() == voice_name.lower():
            voice = MacOSVoice(v.get('identifier', ''))
            break

    # Generate AIFF samples
    aiff_files = cloner.generate_training_samples(voice, output_dir)

    # Convert to WAV
    wav_files = []
    for aiff in aiff_files:
        wav = cloner.convert_to_wav(aiff)
        if wav:
            wav_files.append(wav)

    return wav_files
