"""
Voice processing utilities: auto-tune, modulation, synthesis.
"""

from music_brain.voice.auto_tune import AutoTuneProcessor, AutoTuneSettings, get_auto_tune_preset
from music_brain.voice.modulator import VoiceModulator, ModulationSettings, get_modulation_preset
from music_brain.voice.synthesizer import VoiceSynthesizer, SynthConfig, get_voice_profile

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
]

