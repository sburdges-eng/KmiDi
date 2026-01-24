"""
DSP Module - Digital Signal Processing utilities for iDAW.

Provides:
- Envelope follower and pattern automation (Trace DSP)
- Sample playback engine and pitch shifting (Parrot DSP)
- Common DSP building blocks
"""

from penta_core.dsp.parrot_dsp import (
    GrainCloud,
    PitchShifter,
    PlaybackMode,
    SamplePlayback,
    create_grain_cloud,
    create_pitch_shifter,
    shift_pitch,
    time_stretch,
)
from penta_core.dsp.trace_dsp import (
    AutomationCurve,
    EnvelopeFollower,
    EnvelopeMode,
    PatternAutomation,
    apply_pattern_automation,
    create_envelope_follower,
    follow_envelope,
    generate_lfo_pattern,
)

__all__ = [
    # Trace DSP
    "EnvelopeFollower",
    "EnvelopeMode",
    "PatternAutomation",
    "AutomationCurve",
    "create_envelope_follower",
    "follow_envelope",
    "apply_pattern_automation",
    "generate_lfo_pattern",
    # Parrot DSP
    "SamplePlayback",
    "PitchShifter",
    "GrainCloud",
    "PlaybackMode",
    "create_pitch_shifter",
    "shift_pitch",
    "time_stretch",
    "create_grain_cloud",
]
