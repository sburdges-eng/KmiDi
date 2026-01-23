"""
Voice processing presets for modulation, auto-tune, and synthesis.

This module provides preset configurations for:
- Voice modulation (formant shifting, filtering, etc.)
- Auto-tune settings
- Voice profiles for synthesis
"""

# Modulation presets
MODULATION_PRESETS = {
    "telephone": {
        "formant_shift": 0.0,
        "noise_amount": 0.1,
        "low_pass_hz": 3000.0,
        "band_limit": (300, 3400),
        "saturation": 0.2,
        "bit_depth": 8,
    },
    "radio": {
        "formant_shift": 0.0,
        "noise_amount": 0.05,
        "low_pass_hz": 5000.0,
        "band_limit": (200, 5000),
        "saturation": 0.3,
        "bit_depth": 12,
    },
    "robot": {
        "formant_shift": 0.0,
        "noise_amount": 0.0,
        "low_pass_hz": 4000.0,
        "band_limit": (500, 4000),
        "saturation": 0.5,
        "bit_depth": 6,
    },
    "vintage": {
        "formant_shift": 0.0,
        "noise_amount": 0.15,
        "low_pass_hz": 8000.0,
        "band_limit": None,
        "saturation": 0.4,
        "bit_depth": 10,
    },
}

# Auto-tune presets
AUTO_TUNE_PRESETS = {
    "subtle": {
        "strength": 0.3,
        "retune_speed": 0.2,
        "vibrato_preserve": 0.9,
        "formant_shift": 0.0,
    },
    "moderate": {
        "strength": 0.7,
        "retune_speed": 0.5,
        "vibrato_preserve": 0.8,
        "formant_shift": 0.0,
    },
    "aggressive": {
        "strength": 1.0,
        "retune_speed": 1.0,
        "vibrato_preserve": 0.5,
        "formant_shift": 0.0,
    },
    "cher": {
        "strength": 1.0,
        "retune_speed": 0.1,
        "vibrato_preserve": 0.3,
        "formant_shift": 0.0,
    },
}

# Voice profiles for synthesis
VOICE_PROFILES = {
    "breathy": {
        "timbre": "breathy",
        "vibrato": 0.2,
        "dynamics": 0.7,
    },
    "bright": {
        "timbre": "bright",
        "vibrato": 0.3,
        "dynamics": 0.8,
    },
    "warm": {
        "timbre": "warm",
        "vibrato": 0.15,
        "dynamics": 0.6,
    },
    "robotic": {
        "timbre": "robotic",
        "vibrato": 0.0,
        "dynamics": 1.0,
    },
}
