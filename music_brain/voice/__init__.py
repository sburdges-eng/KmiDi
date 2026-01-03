"""
Voice Processing - Auto-tune, modulation, and voice synthesis.

This module provides voice processing capabilities including:
- AutoTuneProcessor: Pitch correction for vocals
- VoiceModulator: Voice character modification
- VoiceSynthesizer: Text-to-speech and guide vocal generation
- NeuralBackend: DiffSinger/ONNX neural voice synthesis (7-8/10 quality)
"""

from dataclasses import dataclass

# Import neural backend (should always work)
from music_brain.voice.neural_backend import (
    NeuralBackend,
    VoiceSynthesisConfig,
    create_neural_backend,
    check_neural_availability,
)

# Try to import other components (may fail if dependencies missing)
__all__ = [
    # Neural Backend (always available)
    "NeuralBackend",
    "VoiceSynthesisConfig",
    "create_neural_backend",
    "check_neural_availability",
]

try:
    from music_brain.voice.auto_tune import (
        AutoTuneProcessor,
        AutoTuneSettings,
        get_auto_tune_preset,
    )
    __all__.extend([
        "AutoTuneProcessor",
        "AutoTuneSettings",
        "get_auto_tune_preset",
    ])
except ImportError:
    # Provide lightweight stubs so imports succeed even when optional deps (e.g., librosa) are missing.
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

    __all__.extend([
        "AutoTuneProcessor",
        "AutoTuneSettings",
        "get_auto_tune_preset",
    ])

try:
    from music_brain.voice.modulator import (
        VoiceModulator,
        ModulationSettings,
        get_modulation_preset,
    )
    __all__.extend([
        "VoiceModulator",
        "ModulationSettings",
        "get_modulation_preset",
    ])
except ImportError:
    pass

try:
    from music_brain.voice.synthesizer import (
        VoiceSynthesizer,
        SynthConfig,
        get_voice_profile,
    )
    __all__.extend([
        "VoiceSynthesizer",
        "SynthConfig",
        "get_voice_profile",
    ])
except ImportError:
    pass
