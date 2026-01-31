"""
Groove processors for generating rhythmic patterns with intentional rule-breaking.

This module contains functions that generate groove patterns that break
traditional rhythmic rules for emotional and creative effect.
"""

import math
from .base import GeneratedGroove


def generate_groove_constant_displacement(tempo: int) -> GeneratedGroove:
    """
    RHYTHM_ConstantDisplacement
    Shift pattern one 16th note late.
    """
    # 16 slots per bar at 16th note resolution
    # Positive = late, negative = early
    base_offset_ms = (60000 / tempo) / 4  # Duration of one 16th

    # Shift everything late by ~half a 16th
    displacement = base_offset_ms * 0.5

    timing = [displacement] * 16  # Constant late feel

    # Velocity: emphasize 2 and 4 (backbeat)
    velocity = [90, 60, 80, 60, 100, 60, 80, 60, 90, 60, 80, 60, 100, 60, 80, 60]

    return GeneratedGroove(
        pattern_name="Displaced Pocket",
        tempo_bpm=tempo,
        swing_factor=0.0,  # Straight but late
        timing_offsets_16th=timing,
        velocity_curve=velocity,
        rule_broken="RHYTHM_ConstantDisplacement",
        rule_effect="Perpetually behind the beat - unsettling, anxious",
    )


def generate_groove_tempo_fluctuation(tempo: int) -> GeneratedGroove:
    """
    RHYTHM_TempoFluctuation
    Gradual ±5 BPM drift over phrase.
    """
    # Create tempo drift curve over 16 beats (4 bars)
    # Starts at tempo, drifts up to tempo+5, back down

    timing = []
    for i in range(16):
        # Sinusoidal drift
        drift = 5 * math.sin(i * math.pi / 8)  # ±5 BPM
        # Convert BPM drift to ms offset
        base_16th_ms = (60000 / tempo) / 4
        drifted_16th_ms = (60000 / (tempo + drift)) / 4
        offset = drifted_16th_ms - base_16th_ms
        timing.append(offset)

    velocity = [95, 70, 85, 70, 100, 70, 85, 70, 95, 70, 85, 70, 100, 70, 85, 70]

    return GeneratedGroove(
        pattern_name="Breathing Tempo",
        tempo_bpm=tempo,
        swing_factor=0.15,
        timing_offsets_16th=timing,
        velocity_curve=velocity,
        rule_broken="RHYTHM_TempoFluctuation",
        rule_effect="Organic breathing, tension and release through tempo",
    )


def generate_groove_metric_modulation(tempo: int) -> GeneratedGroove:
    """
    RHYTHM_MetricModulation
    Switch implied time signature within loop.
    """
    # First 12 16ths in 4/4, last 4 feel like 3/4
    # Create accent pattern that implies 3/4 at end

    timing = [0] * 16

    # Velocity emphasizes different groupings
    # Bars 1-3: normal 4/4
    # Bar 4: implies 3/4 (accents every 3 instead of 4)
    velocity = [
        100, 60, 80, 60,  # Bar 1: 4/4
        100, 60, 80, 60,  # Bar 2: 4/4
        100, 60, 80, 60,  # Bar 3: 4/4
        100, 70, 80, 100, # Bar 4: shifted accents imply 3/4
    ]

    return GeneratedGroove(
        pattern_name="Metric Shift",
        tempo_bpm=tempo,
        swing_factor=0.0,
        timing_offsets_16th=timing,
        velocity_curve=velocity,
        rule_broken="RHYTHM_MetricModulation",
        rule_effect="Momentary disorientation as time signature shifts",
    )


def generate_groove_dropped_beats(tempo: int) -> GeneratedGroove:
    """
    RHYTHM_DroppedBeats
    Remove expected beats for emphasis through absence.
    """
    # Create gaps - velocity 0 = silence
    velocity = [
        100, 70, 85, 70,  # Bar 1: normal
        100, 70, 85, 0,   # Bar 2: drop the "and" of 4
        100, 0, 85, 70,   # Bar 3: drop the 2
        100, 70, 0, 70,   # Bar 4: drop the 3
    ]

    timing = [0] * 16

    return GeneratedGroove(
        pattern_name="Breathe Space",
        tempo_bpm=tempo,
        swing_factor=0.1,
        timing_offsets_16th=timing,
        velocity_curve=velocity,
        rule_broken="RHYTHM_DroppedBeats",
        rule_effect="Impact through absence - the silence hits harder",
    )


def generate_groove_polyrhythmic_layers(tempo: int) -> GeneratedGroove:
    """
    RHYTHM_PolyrhythmicLayers
    Layer conflicting rhythmic patterns (3 against 4, 5 against 4, etc.).
    """
    # Create a polyrhythmic pattern - 3 against 4
    # In 16 16th notes (4 beats), we need accents at 3:4 ratio
    # 4 beats = positions 0, 4, 8, 12 (every 4th 16th)
    # 3 against 4 = positions 0, 5.33, 10.66 (every 5.33 16ths)

    timing = []
    velocity = []

    for i in range(16):
        # Primary layer (4/4 feel)
        is_four_accent = (i % 4 == 0)
        # Secondary layer (3 over 4 - approximated)
        three_positions = [0, 5, 11]  # Approximation of 3 over 4
        is_three_accent = i in three_positions

        if is_four_accent and is_three_accent:
            velocity.append(110)  # Both layers align - maximum accent
        elif is_four_accent:
            velocity.append(95)   # 4/4 accent
        elif is_three_accent:
            velocity.append(85)   # 3/4 accent
        else:
            velocity.append(60)   # Weak

        # Slight timing push on the "3" layer to emphasize conflict
        if is_three_accent and not is_four_accent:
            timing.append(-5)  # Slightly early, creates tension
        else:
            timing.append(0)

    return GeneratedGroove(
        pattern_name="Polyrhythmic Tension",
        tempo_bpm=tempo,
        swing_factor=0.0,  # Keep straight to hear the polyrhythm
        timing_offsets_16th=timing,
        velocity_curve=velocity,
        rule_broken="RHYTHM_PolyrhythmicLayers",
        rule_effect="Competing rhythmic grids create complexity and internal conflict",
    )
