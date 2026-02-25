"""
Voice package exports.
"""

from .auto_tune import AutoTuneProcessor, AutoTuneSettings, get_auto_tune_preset
from .modulator import VoiceModulator, ModulationSettings, get_modulation_preset
from .synthesizer import VoiceSynthesizer, SynthConfig, get_voice_profile
from .voice_classifier import VoiceClassifier

__all__ = [
    "AutoTuneProcessor",
    "AutoTuneSettings",
    "get_auto_tune_preset",
    "VoiceModulator",
    "ModulationSettings",
    "get_modulation_preset",
    "VoiceSynthesizer",
    "SynthConfig",
    "get_voice_profile",
    "VoiceClassifier",
]
