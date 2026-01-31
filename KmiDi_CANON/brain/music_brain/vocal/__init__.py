"""
Vocal Synthesizer Module - Parrot Function

The Parrot function learns voice characteristics from uploaded audio and can
mimic voices after prolonged exposure. It learns:
- Vowel formants and transitions
- Accent characteristics
- Pitch contours and vibrato
- Timbre and spectral characteristics
- Speaking/singing style
"""

from music_brain.vocal.parrot import (
    ParrotVocalSynthesizer,
    VoiceModel,
    VoiceCharacteristics,
    ParrotConfig,
    analyze_voice,
    synthesize_vocal,
    train_parrot,
    load_voice_model,
    save_voice_model,
)
from music_brain.vocal.phonemes import (
    text_to_phonemes,
    Phoneme,
    PhonemeType,
)
from music_brain.vocal.synthesis import (
    formant_synthesize,
)

__all__ = [
    "ParrotVocalSynthesizer",
    "VoiceModel",
    "VoiceCharacteristics",
    "ParrotConfig",
    "analyze_voice",
    "synthesize_vocal",
    "train_parrot",
    "load_voice_model",
    "save_voice_model",
    "text_to_phonemes",
    "Phoneme",
    "PhonemeType",
    "formant_synthesize",
]

