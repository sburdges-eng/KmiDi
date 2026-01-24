"""
Neural Voice Synthesis Integration

Unified interface for neural TTS and voice cloning:
- Coqui TTS (XTTS, VITS, YourTTS)
- Bark (Suno AI)
- OpenVoice (MyShell AI)
- Piper (fast local TTS)

This module integrates with the existing DAiW voice system,
providing neural synthesis as an alternative to formant synthesis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from enum import Enum
import numpy as np
import tempfile
import os

# Backend availability
COQUI_AVAILABLE = False
BARK_AVAILABLE = False
OPENVOICE_AVAILABLE = False
PIPER_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    pass

try:
    from bark import generate_audio, preload_models, SAMPLE_RATE as BARK_SAMPLE_RATE
    BARK_AVAILABLE = True
except ImportError:
    pass

try:
    # OpenVoice import (requires manual installation)
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    OPENVOICE_AVAILABLE = True
except ImportError:
    pass

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class NeuralVoiceBackend(Enum):
    """Available neural voice backends"""
    COQUI = "coqui"
    BARK = "bark"
    OPENVOICE = "openvoice"
    PIPER = "piper"
    AUTO = "auto"


@dataclass
class NeuralVoiceConfig:
    """Configuration for neural voice synthesis"""
    backend: NeuralVoiceBackend = NeuralVoiceBackend.AUTO
    model_name: str = ""                    # Model identifier
    language: str = "en"                    # Language code
    speaker_wav: Optional[str] = None       # Reference audio for cloning
    device: str = "auto"                    # cuda/cpu/mps/auto
    use_gpu: bool = True
    sample_rate: int = 24000

    # Bark specific
    bark_speaker_preset: str = "v2/en_speaker_6"

    # Coqui specific
    coqui_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Voice style
    emotion: str = "neutral"                # happy, sad, angry, neutral
    speed: float = 1.0                      # Speech speed multiplier
    pitch_shift: float = 0.0               # Semitones


class NeuralVoiceSynthesizer(ABC):
    """Abstract base class for neural voice synthesizers"""

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text"""
        pass

    @abstractmethod
    def clone_voice(self, reference_audio: str) -> bool:
        """Clone a voice from reference audio"""
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get output sample rate"""
        pass


class CoquiVoiceSynthesizer(NeuralVoiceSynthesizer):
    """
    Coqui TTS synthesizer with XTTS, VITS, and voice cloning.

    Example:
        synth = CoquiVoiceSynthesizer()
        synth.clone_voice("reference.wav")
        audio = synth.synthesize("Hello world")
    """

    def __init__(self, config: Optional[NeuralVoiceConfig] = None):
        if not COQUI_AVAILABLE:
            raise ImportError(
                "Coqui TTS not available. Install with: pip install TTS"
            )

        self.config = config or NeuralVoiceConfig()

        # Detect device
        if self.config.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        # Load model
        self.tts = TTS(self.config.coqui_model).to(self.device)
        self.speaker_wav = self.config.speaker_wav
        self.sample_rate = 24000  # XTTS default

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Audio as numpy array
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            if self.speaker_wav:
                # Voice cloning mode
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=self.speaker_wav,
                    language=self.config.language,
                    file_path=temp_path
                )
            else:
                # Default voice
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_path
                )

            # Load audio
            audio, sr = sf.read(temp_path)
            self.sample_rate = sr
            return audio.astype(np.float32)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def clone_voice(self, reference_audio: str) -> bool:
        """
        Set reference audio for voice cloning.

        Args:
            reference_audio: Path to reference audio file

        Returns:
            True if successful
        """
        if not os.path.exists(reference_audio):
            print(f"Reference audio not found: {reference_audio}")
            return False

        self.speaker_wav = reference_audio
        return True

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def list_models(self) -> List[str]:
        """List available Coqui TTS models"""
        return TTS().list_models()

    def list_languages(self) -> List[str]:
        """List supported languages for current model"""
        if hasattr(self.tts, 'languages'):
            return self.tts.languages
        return ["en"]


class BarkVoiceSynthesizer(NeuralVoiceSynthesizer):
    """
    Bark audio generation model by Suno AI.

    Supports speech, music, and sound effects generation.

    Example:
        synth = BarkVoiceSynthesizer()
        audio = synth.synthesize("Hello world [laughs]")
    """

    def __init__(self, config: Optional[NeuralVoiceConfig] = None):
        if not BARK_AVAILABLE:
            raise ImportError(
                "Bark not available. Install with: pip install git+https://github.com/suno-ai/bark.git"
            )

        self.config = config or NeuralVoiceConfig()

        # Preload models
        print("Loading Bark models...")
        preload_models()

        self.sample_rate = BARK_SAMPLE_RATE
        self.speaker_preset = self.config.bark_speaker_preset

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize audio from text.

        Supports special tokens:
        - [laughs], [sighs], [music] for non-speech
        - [MAN], [WOMAN] for voice hints

        Args:
            text: Text with optional tokens

        Returns:
            Audio as numpy array
        """
        audio_array = generate_audio(
            text,
            history_prompt=self.speaker_preset
        )

        return audio_array.astype(np.float32)

    def clone_voice(self, reference_audio: str) -> bool:
        """
        Bark doesn't support true voice cloning.
        Use speaker presets instead.
        """
        print("Bark uses speaker presets, not voice cloning.")
        print("Available presets: v2/en_speaker_0 through v2/en_speaker_9")
        return False

    def set_speaker_preset(self, preset: str):
        """
        Set speaker preset.

        Args:
            preset: e.g. "v2/en_speaker_6", "v2/de_speaker_3"
        """
        self.speaker_preset = preset

    def get_sample_rate(self) -> int:
        return self.sample_rate

    @staticmethod
    def list_presets() -> List[str]:
        """List available speaker presets"""
        languages = ["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"]
        presets = []
        for lang in languages:
            for i in range(10):
                presets.append(f"v2/{lang}_speaker_{i}")
        return presets


class OpenVoiceSynthesizer(NeuralVoiceSynthesizer):
    """
    OpenVoice instant voice cloning by MyShell AI.

    Features:
    - Instant voice cloning from short reference
    - Cross-lingual cloning
    - Style/emotion control

    Example:
        synth = OpenVoiceSynthesizer()
        synth.clone_voice("reference.wav")
        audio = synth.synthesize("Hello in cloned voice")
    """

    def __init__(self, config: Optional[NeuralVoiceConfig] = None):
        if not OPENVOICE_AVAILABLE:
            raise ImportError(
                "OpenVoice not available. Install from: "
                "https://github.com/myshell-ai/OpenVoice"
            )

        self.config = config or NeuralVoiceConfig()

        # Device setup
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        # Load models (paths need to be configured)
        self.tone_color_converter = None
        self.source_se = None
        self.target_se = None
        self.sample_rate = 24000

        self._initialized = False

    def _initialize(self, checkpoint_path: str):
        """Initialize with checkpoint path"""
        self.tone_color_converter = ToneColorConverter(
            f"{checkpoint_path}/converter"
        )
        self.tone_color_converter.to(self.device)
        self._initialized = True

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize with cloned voice.

        Requires clone_voice() to be called first.
        """
        if not self._initialized or self.target_se is None:
            raise RuntimeError("Call clone_voice() first")

        # OpenVoice workflow:
        # 1. Generate base speech with TTS
        # 2. Convert tone color to target

        # This is a simplified implementation
        # Full implementation requires the base TTS model
        raise NotImplementedError(
            "Full OpenVoice synthesis requires additional setup. "
            "See https://github.com/myshell-ai/OpenVoice for details."
        )

    def clone_voice(self, reference_audio: str) -> bool:
        """
        Extract voice characteristics from reference audio.

        Args:
            reference_audio: Path to reference audio (3-10 seconds ideal)

        Returns:
            True if successful
        """
        if not self._initialized:
            print("OpenVoice not fully initialized. Provide checkpoint path.")
            return False

        try:
            # Extract speaker embedding
            self.target_se = se_extractor.get_se(
                reference_audio,
                self.tone_color_converter,
                target_dir="processed",
                vad=True
            )
            return True
        except Exception as e:
            print(f"Voice cloning failed: {e}")
            return False

    def get_sample_rate(self) -> int:
        return self.sample_rate


class UnifiedNeuralVoice:
    """
    Unified interface for all neural voice backends.

    Automatically selects the best available backend and provides
    a consistent API for voice synthesis and cloning.

    Example:
        voice = UnifiedNeuralVoice()

        # Clone a voice
        voice.clone_voice("my_voice.wav")

        # Synthesize
        audio = voice.synthesize("Hello, this is my cloned voice!")

        # Save
        voice.save_audio(audio, "output.wav")
    """

    def __init__(self, config: Optional[NeuralVoiceConfig] = None):
        self.config = config or NeuralVoiceConfig()
        self._backend: Optional[NeuralVoiceSynthesizer] = None

        # Auto-select backend
        if self.config.backend == NeuralVoiceBackend.AUTO:
            self._backend = self._select_best_backend()
        else:
            self._backend = self._create_backend(self.config.backend)

    def _select_best_backend(self) -> NeuralVoiceSynthesizer:
        """Select best available backend"""
        # Priority: Coqui > OpenVoice > Bark
        if COQUI_AVAILABLE:
            print("Using Coqui TTS backend")
            return CoquiVoiceSynthesizer(self.config)
        elif OPENVOICE_AVAILABLE:
            print("Using OpenVoice backend")
            return OpenVoiceSynthesizer(self.config)
        elif BARK_AVAILABLE:
            print("Using Bark backend")
            return BarkVoiceSynthesizer(self.config)
        else:
            raise ImportError(
                "No neural voice backend available. Install one of:\n"
                "  pip install TTS          # Coqui\n"
                "  pip install bark         # Bark\n"
            )

    def _create_backend(self, backend: NeuralVoiceBackend) -> NeuralVoiceSynthesizer:
        """Create specific backend"""
        if backend == NeuralVoiceBackend.COQUI:
            return CoquiVoiceSynthesizer(self.config)
        elif backend == NeuralVoiceBackend.BARK:
            return BarkVoiceSynthesizer(self.config)
        elif backend == NeuralVoiceBackend.OPENVOICE:
            return OpenVoiceSynthesizer(self.config)
        else:
            return self._select_best_backend()

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Audio as numpy array
        """
        return self._backend.synthesize(text)

    def clone_voice(self, reference_audio: str) -> bool:
        """
        Clone voice from reference audio.

        Args:
            reference_audio: Path to reference audio

        Returns:
            True if successful
        """
        return self._backend.clone_voice(reference_audio)

    def get_sample_rate(self) -> int:
        """Get output sample rate"""
        return self._backend.get_sample_rate()

    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio to file.

        Args:
            audio: Audio array
            output_path: Output file path
        """
        if SOUNDFILE_AVAILABLE:
            sf.write(output_path, audio, self.get_sample_rate())
        else:
            raise ImportError("soundfile required for saving audio")

    def synthesize_to_file(self, text: str, output_path: str):
        """
        Synthesize and save to file.

        Args:
            text: Text to synthesize
            output_path: Output file path
        """
        audio = self.synthesize(text)
        self.save_audio(audio, output_path)


# Integration with DAiW voice system
class DAiWNeuralVoiceIntegration:
    """
    Integration layer between neural TTS and DAiW formant synthesis.

    Combines:
    - Neural TTS for natural speech generation
    - Parrot formant synthesis for real-time control
    - Voice effects processing

    Example:
        integration = DAiWNeuralVoiceIntegration()

        # Learn voice characteristics from neural output
        integration.learn_from_neural("Hello world", "my_voice")

        # Use formant synthesis with learned characteristics
        audio = integration.synthesize_realtime("Hello again")
    """

    def __init__(self):
        self.neural_voice = None
        self.parrot = None

        # Try to import Parrot
        try:
            from music_brain.vocal.parrot import ParrotVocalSynthesizer, ParrotConfig
            self.parrot = ParrotVocalSynthesizer(ParrotConfig())
        except ImportError:
            pass

    def initialize_neural(self, backend: NeuralVoiceBackend = NeuralVoiceBackend.AUTO):
        """Initialize neural voice backend"""
        config = NeuralVoiceConfig(backend=backend)
        self.neural_voice = UnifiedNeuralVoice(config)

    def clone_voice_neural(self, reference_audio: str) -> bool:
        """Clone voice using neural backend"""
        if not self.neural_voice:
            self.initialize_neural()
        return self.neural_voice.clone_voice(reference_audio)

    def synthesize_neural(self, text: str) -> np.ndarray:
        """Synthesize using neural backend"""
        if not self.neural_voice:
            self.initialize_neural()
        return self.neural_voice.synthesize(text)

    def learn_from_neural(self, text: str, voice_name: str):
        """
        Generate neural speech and learn characteristics for formant synthesis.

        Args:
            text: Text to generate
            voice_name: Name for learned voice model
        """
        if not self.neural_voice:
            self.initialize_neural()

        if not self.parrot:
            print("Parrot not available for voice learning")
            return

        # Generate neural audio
        audio = self.neural_voice.synthesize(text)
        sr = self.neural_voice.get_sample_rate()

        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio, sr)

            # Learn with Parrot
            self.parrot.train_parrot(temp_path, voice_name)
            print(f"Learned voice characteristics as '{voice_name}'")

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def get_parrot_synthesizer(self):
        """Get the Parrot synthesizer for real-time control"""
        return self.parrot


# Convenience functions
def quick_neural_speak(text: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Quick function to synthesize speech with neural TTS.

    Args:
        text: Text to speak
        output_path: Optional path to save audio

    Returns:
        Audio array
    """
    voice = UnifiedNeuralVoice()
    audio = voice.synthesize(text)

    if output_path:
        voice.save_audio(audio, output_path)

    return audio


def quick_voice_clone(reference_audio: str, text: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Quick function to clone a voice and synthesize.

    Args:
        reference_audio: Path to reference audio
        text: Text to synthesize
        output_path: Optional path to save audio

    Returns:
        Audio array
    """
    voice = UnifiedNeuralVoice()
    voice.clone_voice(reference_audio)
    audio = voice.synthesize(text)

    if output_path:
        voice.save_audio(audio, output_path)

    return audio


def get_available_backends() -> Dict[str, bool]:
    """Get availability of neural voice backends"""
    return {
        "coqui": COQUI_AVAILABLE,
        "bark": BARK_AVAILABLE,
        "openvoice": OPENVOICE_AVAILABLE,
        "piper": PIPER_AVAILABLE,
        "torch": TORCH_AVAILABLE,
    }
