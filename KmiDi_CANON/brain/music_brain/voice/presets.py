"""
Preset definitions for voice processing modules.
"""

from __future__ import annotations

AUTO_TUNE_PRESETS = {
    "transparent": {
        "strength": 0.5,
        "retune_speed": 0.8,
        "vibrato_preserve": 0.7,
        "formant_shift": 0.0,
    },
    "emotional_rescue": {
        "strength": 0.65,
        "retune_speed": 0.6,
        "vibrato_preserve": 0.9,
        "formant_shift": -0.2,
    },
    "hyperpop_lock": {
        "strength": 1.0,
        "retune_speed": 0.2,
        "vibrato_preserve": 0.2,
        "formant_shift": 0.4,
    },
}

MODULATION_PRESETS = {
    "intimate_whisper": {
        "formant_shift": 2.0,
        "noise_amount": 0.02,
        "low_pass_hz": 8000,
        "saturation": 0.1,
    },
    "distant_radio": {
        "formant_shift": -1.0,
        "noise_amount": 0.05,
        "band_limit": (400, 3500),
        "saturation": 0.2,
    },
    "robotic_sermon": {
        "formant_shift": -3.0,
        "noise_amount": 0.0,
        "bit_depth": 10,
        "saturation": 0.35,
    },
}

VOICE_PROFILES = {
    "guide_vulnerable": {
        "timbre": "breathy",
        "vibrato": 0.3,
        "dynamics": 0.6,
    },
    "guide_confident": {
        "timbre": "clean",
        "vibrato": 0.1,
        "dynamics": 0.9,
    },
    "ai_choir": {
        "timbre": "stacked",
        "vibrato": 0.4,
        "dynamics": 0.7,
    },
}

__all__ = ["AUTO_TUNE_PRESETS", "MODULATION_PRESETS", "VOICE_PROFILES"]

