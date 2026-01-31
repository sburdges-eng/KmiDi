"""
Temporal processors for generating time-based structure guidelines with intentional rule-breaking.

This module contains functions that generate temporal patterns that break
traditional pacing and timing rules for emotional and creative effect.
"""

from .base import GeneratedTemporal


def generate_temporal_extended_intro() -> GeneratedTemporal:
    """
    TEMPORAL_ExtendedIntro
    Unusually long intro for anticipation and world-building.
    """
    return GeneratedTemporal(
        pacing="Slow establishment then normal",
        section_timing=[
            {"section": "Intro", "duration": "2-4 minutes", "notes": "The setup IS the story"},
            {"section": "Main content", "duration": "Normal", "notes": "Feels like relief when it arrives"},
        ],
        pause_strategy="Extended anticipation before 'the beginning'",
        transition_style="Gradual accumulation - each element earned",
        time_feel="Patient, deliberately slow - testing the listener",
        special_moments=[
            {"moment": "First vocal/main element entry", "timing": "60+ seconds in", "effect": "Arrival feels significant"},
            {"moment": "Texture builds", "timing": "Throughout intro", "effect": "World construction"},
            {"moment": "Full arrangement", "timing": "2+ minutes", "effect": "Finally here"},
        ],
        rule_broken="TEMPORAL_ExtendedIntro",
        rule_effect="Anticipation, world-building, patience test - earning the beginning",
    )


def generate_temporal_abrupt_ending() -> GeneratedTemporal:
    """
    TEMPORAL_AbruptEnding
    End suddenly without resolution for shock.
    """
    return GeneratedTemporal(
        pacing="Normal then STOP",
        section_timing=[
            {"section": "Normal song", "duration": "Standard", "notes": "Build expectation of ending"},
            {"section": "Cut", "duration": "Instant", "notes": "No fade, no resolution - STOP"},
        ],
        pause_strategy="No pause - the absence of ending IS the ending",
        transition_style="No transition to nothing - the cut is total",
        time_feel="Normal until the last millisecond",
        special_moments=[
            {"moment": "The cut", "timing": "Mid-phrase ideally", "effect": "Maximum shock"},
            {"moment": "What's left out", "timing": "Never happens", "effect": "The resolution we needed but didn't get"},
        ],
        rule_broken="TEMPORAL_AbruptEnding",
        rule_effect="Shock, incompleteness, sudden loss - the story cut short",
    )


def generate_temporal_time_stretch() -> GeneratedTemporal:
    """
    TEMPORAL_TimeStretch
    Stretch or compress time perception for altered reality.
    """
    return GeneratedTemporal(
        pacing="Elastic - time bends",
        section_timing=[
            {"section": "Normal section", "duration": "4 bars feels like 4 bars", "notes": "Establish baseline"},
            {"section": "Stretched section", "duration": "4 bars feels like 16", "notes": "Tempo halves, sparse arrangement"},
            {"section": "Compressed section", "duration": "4 bars feels like 2", "notes": "Double time feel, dense"},
        ],
        pause_strategy="Pauses feel eternal in stretched sections",
        transition_style="Reality shifts at section boundaries",
        time_feel="Disorienting - 'how long has this been playing?'",
        special_moments=[
            {"moment": "First stretch", "timing": "When reality shifts", "effect": "Disassociation onset"},
            {"moment": "Return to normal", "timing": "If it happens", "effect": "Grounding, or more disorientation"},
        ],
        rule_broken="TEMPORAL_TimeStretch",
        rule_effect="Altered reality, time distortion, dream state - time becomes unreliable",
    )


def generate_temporal_loop_hypnosis() -> GeneratedTemporal:
    """
    TEMPORAL_LoopHypnosis
    Loop beyond comfortable repetition for trance state.
    """
    return GeneratedTemporal(
        pacing="Static - the loop IS time",
        section_timing=[
            {"section": "The loop", "duration": "8-32 bars repeated 8+ times", "notes": "Longer than comfortable"},
            {"section": "Micro-variations", "duration": "Within loop", "notes": "Tiny changes maintain engagement"},
        ],
        pause_strategy="No pauses - the loop is continuous",
        transition_style="No traditional transitions - the loop continues or stops",
        time_feel="Hypnotic, meditative, eventually maddening",
        special_moments=[
            {"moment": "When comfort breaks", "timing": "Around repetition 4-5", "effect": "From pleasant to obsessive"},
            {"moment": "If loop breaks", "timing": "After extended repetition", "effect": "Shock, relief, or loss"},
        ],
        rule_broken="TEMPORAL_LoopHypnosis",
        rule_effect="Hypnotic, meditative, obsessive - circular thoughts that won't stop",
    )


def generate_temporal_breath_pauses() -> GeneratedTemporal:
    """
    TEMPORAL_BreathPauses
    Insert pauses like held breath for anticipation.
    """
    return GeneratedTemporal(
        pacing="Normal with significant interruptions",
        section_timing=[
            {"section": "Normal flow", "duration": "Standard", "notes": "Establish rhythm"},
            {"section": "Breath pause", "duration": "1-4 beats of silence", "notes": "Everything stops"},
            {"section": "Resume", "duration": "Standard", "notes": "Continue or transform"},
        ],
        pause_strategy="Strategic silences before key moments - inhale before speaking",
        transition_style="Pauses ARE the transitions",
        time_feel="Punctuated, breathless, gathering courage",
        special_moments=[
            {"moment": "Pre-chorus pause", "timing": "Before the release", "effect": "Anticipation builds"},
            {"moment": "Mid-word pause", "timing": "Maximum tension", "effect": "Holding back what's hard to say"},
            {"moment": "Post-climax pause", "timing": "After the revelation", "effect": "Processing, impact"},
        ],
        rule_broken="TEMPORAL_BreathPauses",
        rule_effect="Tension, anticipation, gathering courage - the moment before speaking",
    )


def generate_temporal_accelerando_decay() -> GeneratedTemporal:
    """
    TEMPORAL_AccelerandoDecay
    Speed up then collapse for panic and exhaustion.
    """
    return GeneratedTemporal(
        pacing="Accelerating then collapsing",
        section_timing=[
            {"section": "Normal", "duration": "Standard tempo", "notes": "Establish baseline"},
            {"section": "Acceleration", "duration": "Gradual then rapid tempo increase", "notes": "Panic setting in"},
            {"section": "Breaking point", "duration": "Maximum speed, unsustainable", "notes": "Can't keep up"},
            {"section": "Collapse", "duration": "Sudden slowdown or stop", "notes": "Exhaustion, surrender"},
        ],
        pause_strategy="No pauses during acceleration - no rest allowed",
        transition_style="Tempo IS the transition - faster = more desperate",
        time_feel="Frantic then exhausted - a sprint that ends in collapse",
        special_moments=[
            {"moment": "When acceleration starts", "timing": "Anxiety trigger", "effect": "Things spinning out"},
            {"moment": "The break", "timing": "Can't be sustained", "effect": "The giving up"},
            {"moment": "The aftermath", "timing": "Whatever remains", "effect": "Exhaustion or peace"},
        ],
        rule_broken="TEMPORAL_AccelerandoDecay",
        rule_effect="Panic, collapse, exhaustion - overwhelm leading to surrender",
    )
