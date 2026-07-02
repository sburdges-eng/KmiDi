"""
Voice Processing - Auto-tune, modulation, and voice synthesis.

This module provides voice processing capabilities including:
- AutoTuneProcessor: Pitch correction for vocals
- VoiceModulator: Voice character modification
- VoiceSynthesizer: Text-to-speech and guide vocal generation
- NeuralBackend: DiffSinger/ONNX neural voice synthesis when available
"""

from dataclasses import dataclass

# Try to import other components (may fail if dependencies missing)
__all__ = []

try:
    from music_brain.voice.neural_backend import NeuralBackend  # noqa: F401
    from music_brain.voice.neural_backend import VoiceSynthesisConfig  # noqa: F401
    from music_brain.voice.neural_backend import create_neural_backend  # noqa: F401
    from music_brain.voice.neural_backend import check_neural_availability  # noqa: F401

    __all__.extend(
        [
            "NeuralBackend",
            "VoiceSynthesisConfig",
            "create_neural_backend",
            "check_neural_availability",
        ]
    )
except ImportError:
    pass

try:
    from music_brain.voice.auto_tune import (
        AutoTuneProcessor,
        AutoTuneSettings,
        get_auto_tune_preset,
    )

    __all__.extend(
        [
            "AutoTuneProcessor",
            "AutoTuneSettings",
            "get_auto_tune_preset",
        ]
    )
except ImportError:
    # Provide lightweight stubs so imports succeed even when optional deps
    # (e.g., librosa) are missing.
    class AutoTuneProcessor:
        def __init__(self, *args, **kwargs):
            self.available = False

        def process(self, audio, *args, **kwargs):  # pragma: no cover - stub
            # No-op passthrough when dependencies are missing.
            return audio

    @dataclass
    class AutoTuneSettings:
        key: str = "C"
        scale: str = "major"
        retune_speed: float = 0.5
        humanize: float = 0.0

    def get_auto_tune_preset(name: str) -> AutoTuneSettings:  # pragma: no cover - stub
        return AutoTuneSettings()

    __all__.extend(
        [
            "AutoTuneProcessor",
            "AutoTuneSettings",
            "get_auto_tune_preset",
        ]
    )

try:
    from music_brain.voice.modulator import VoiceModulator  # noqa: F401
    from music_brain.voice.modulator import ModulationSettings  # noqa: F401
    from music_brain.voice.modulator import get_modulation_preset  # noqa: F401

    __all__.extend(
        [
            "VoiceModulator",
            "ModulationSettings",
            "get_modulation_preset",
        ]
    )
except ImportError:
    pass

try:
    from music_brain.voice.synthesizer import VoiceSynthesizer  # noqa: F401
    from music_brain.voice.synthesizer import SynthConfig  # noqa: F401
    from music_brain.voice.synthesizer import get_voice_profile  # noqa: F401

    __all__.extend(
        [
            "VoiceSynthesizer",
            "SynthConfig",
            "get_voice_profile",
        ]
    )
except ImportError:
    pass

try:
    from music_brain.voice.voice_classifier import VoiceClassifier

    __all__.extend(["VoiceClassifier"])
except ImportError:

    class VoiceClassifier:  # type: ignore[no-redef]
        """Stub when voice_classifier dependencies are missing."""

        def __init__(self, *args, **kwargs):
            self.available = False

    __all__.extend(["VoiceClassifier"])

# Main-repo presets (used by misc_code, api; restore surface after merge)
try:
    from music_brain.voice.presets import MODULATION_PRESETS  # noqa: F401
    from music_brain.voice.presets import AUTO_TUNE_PRESETS  # noqa: F401
    from music_brain.voice.presets import VOICE_PROFILES  # noqa: F401

    __all__.extend(
        [
            "MODULATION_PRESETS",
            "AUTO_TUNE_PRESETS",
            "VOICE_PROFILES",
        ]
    )
except ImportError:
    pass

# Optional: macOS native TTS (main-repo only)
try:
    from music_brain.voice.macos_speech import MacOSVoice, MacOSSpeechSynthesizer  # noqa: F401

    __all__.extend(["MacOSVoice", "MacOSSpeechSynthesizer"])
except ImportError:
    pass

# Optional: neural TTS backends (Coqui, Bark, OpenVoice, Piper)
try:
    from music_brain.voice.neural_voice import UnifiedNeuralVoice  # noqa: F401
    from music_brain.voice.neural_voice import NeuralVoiceConfig  # noqa: F401
    from music_brain.voice.neural_voice import NeuralVoiceBackend  # noqa: F401

    __all__.extend(
        [
            "UnifiedNeuralVoice",
            "NeuralVoiceConfig",
            "NeuralVoiceBackend",
        ]
    )
except ImportError:
    pass

# Singing voice suite (ported from archive/feature/integration-finalize):
# phoneme/pitch processing, formant + neural synthesis, voice input/learning.
try:
    from music_brain.voice.phoneme_processor import PhonemeProcessor  # noqa: F401
    from music_brain.voice.phoneme_processor import PhonemeSequence  # noqa: F401
    from music_brain.voice.phoneme_processor import Phoneme  # noqa: F401

    __all__.extend(["PhonemeProcessor", "PhonemeSequence", "Phoneme"])
except ImportError:
    pass

try:
    from music_brain.voice.pitch_controller import PitchController  # noqa: F401
    from music_brain.voice.pitch_controller import PitchCurve  # noqa: F401
    from music_brain.voice.pitch_controller import ExpressionParams  # noqa: F401

    __all__.extend(["PitchController", "PitchCurve", "ExpressionParams"])
except ImportError:
    pass

try:
    from music_brain.voice.singing_synthesizer import SingingSynthesizer  # noqa: F401
    from music_brain.voice.singing_synthesizer import FormantConfig  # noqa: F401

    __all__.extend(["SingingSynthesizer", "FormantConfig"])
except ImportError:
    pass

try:
    from music_brain.voice.singing_voice import SingingVoice  # noqa: F401
    from music_brain.voice.singing_voice import create_singing_voice  # noqa: F401

    __all__.extend(["SingingVoice", "create_singing_voice"])
except ImportError:
    pass

try:
    from music_brain.voice.singing_voice_dev import SingingVoiceDev  # noqa: F401
    from music_brain.voice.singing_voice_dev import create_singing_voice_dev  # noqa: F401

    __all__.extend(["SingingVoiceDev", "create_singing_voice_dev"])
except ImportError:
    pass

try:
    from music_brain.voice.voice_input import VoiceRecorder  # noqa: F401
    from music_brain.voice.voice_input import VoiceMimic  # noqa: F401

    __all__.extend(["VoiceRecorder", "VoiceMimic"])
except ImportError:
    pass

try:
    from music_brain.voice.voice_learning import VoiceLearningManager  # noqa: F401
    from music_brain.voice.voice_learning import VoiceLearner  # noqa: F401
    from music_brain.voice.voice_learning import VoiceSampleStore  # noqa: F401
    from music_brain.voice.voice_learning import VoiceSample  # noqa: F401
    from music_brain.voice.voice_learning import LearnedVoiceProfile  # noqa: F401

    __all__.extend(
        [
            "VoiceLearningManager",
            "VoiceLearner",
            "VoiceSampleStore",
            "VoiceSample",
            "LearnedVoiceProfile",
        ]
    )
except ImportError:
    pass

try:
    from music_brain.voice.instrument_synth import InstrumentSynthesizer  # noqa: F401
    from music_brain.voice.instrument_synth import InstrumentConfig  # noqa: F401
    from music_brain.voice.instrument_synth import get_instrument_preset  # noqa: F401

    __all__.extend(["InstrumentSynthesizer", "InstrumentConfig", "get_instrument_preset"])
except ImportError:
    pass
