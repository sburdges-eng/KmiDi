"""
Melody processors for generating melodic guidelines with intentional rule-breaking.

This module contains functions that generate melody patterns that break
traditional melodic rules for emotional and creative effect.
"""

from .base import GeneratedMelody


def generate_melody_avoid_resolution(key: str, mode: str = "major") -> GeneratedMelody:
    """
    MELODY_AvoidResolution
    End phrases on non-tonic tones for incompleteness.
    """
    return GeneratedMelody(
        contour="Ascending or arch - reaching but never arriving",
        interval_character="Step-wise with occasional leaps that don't resolve",
        phrase_structure="Phrases end on 2nd, 6th, or 7th scale degree",
        resolution_behavior="NEVER resolve to tonic at phrase end - hang on tensions",
        rhythmic_character="Phrases trail off or sustain rather than land",
        range_notes="Mid-range preferred - high notes imply climax/resolution",
        motif_ideas=[
            "End verse phrases on the 2nd (re) - eternal questioning",
            "Use 7th as final note - perpetual leading",
            "Sustain through expected resolution point",
            "Let phrases fade rather than conclude",
        ],
        performance_notes=[
            "Vocal should sound like thought continuing beyond the phrase",
            "Avoid finality in delivery - everything is ongoing",
            "Consider breath placement that suggests continuation",
        ],
        rule_broken="MELODY_AvoidResolution",
        rule_effect="Incompleteness, searching, yearning - the question without answer",
    )


def generate_melody_excessive_repetition(key: str, mode: str = "major") -> GeneratedMelody:
    """
    MELODY_ExcessiveRepetition
    Repeat melodic cell obsessively beyond comfort.
    """
    return GeneratedMelody(
        contour="Circular - the same shape repeating",
        interval_character="Simple, memorable cell (2-4 notes) repeated obsessively",
        phrase_structure="Same phrase 8+ times with minimal variation",
        resolution_behavior="Resolution exists but becomes meaningless through repetition",
        rhythmic_character="Locked, mechanical, ritualistic",
        range_notes="Narrow range - trapped within a few notes",
        motif_ideas=[
            "3-note descending cell repeated throughout entire verse",
            "Single phrase that IS the chorus - nothing else",
            "Micro-variations only (dynamics, timing) - melody stays identical",
            "Consider the mantra quality - the repetition IS the meaning",
        ],
        performance_notes=[
            "Each repetition should feel both inevitable and maddening",
            "Subtle emotional shift through repetitions despite identical notes",
            "The monotony should become hypnotic, then uncomfortable",
            "Let desperation or acceptance creep in through delivery, not melody",
        ],
        rule_broken="MELODY_ExcessiveRepetition",
        rule_effect="Hypnotic, obsessive, ritualistic - spiral thoughts that can't escape",
    )


def generate_melody_angular_intervals(key: str, mode: str = "major") -> GeneratedMelody:
    """
    MELODY_AngularIntervals
    Use wide, awkward interval leaps for discomfort.
    """
    return GeneratedMelody(
        contour="Jagged, unpredictable - avoiding smooth motion",
        interval_character="Tritones, 7ths, 9ths - the 'wrong' intervals",
        phrase_structure="Fragmented by the leaps - hard to sing along",
        resolution_behavior="Leaps may resolve but to unexpected places",
        rhythmic_character="Syncopated or irregular to emphasize the discomfort",
        range_notes="Wide range required - exploiting the extremes",
        motif_ideas=[
            "Tritone leap at emotional peak - the devil's interval for the unspeakable",
            "Minor 9th drop for sudden isolation/falling",
            "Avoid all stepwise motion - every move is a jump",
            "7th up followed by tritone down - complete destabilization",
        ],
        performance_notes=[
            "Don't smooth out the leaps - the awkwardness is the point",
            "Let voice crack or strain on difficult intervals",
            "The discomfort in singing mirrors the emotional discomfort",
            "Consider speaking sections where melody becomes too difficult",
        ],
        rule_broken="MELODY_AngularIntervals",
        rule_effect="Discomfort, unease, alienation - something is deeply wrong",
    )


def generate_melody_anti_climax(key: str, mode: str = "major") -> GeneratedMelody:
    """
    MELODY_AntiClimax
    Build up then resolve downward/weakly instead of triumphantly.
    """
    return GeneratedMelody(
        contour="Ascending build that deflates - the arch collapses",
        interval_character="Ascending steps that reverse into descending minor intervals",
        phrase_structure="Long build → weak landing, often below starting point",
        resolution_behavior="Resolves DOWN when up is expected - deflation",
        rhythmic_character="Accelerating then suddenly slower/quieter",
        range_notes="Build to high range, but climax arrives low or mid",
        motif_ideas=[
            "Pre-chorus rises a 5th, chorus enters a 3rd BELOW where verse started",
            "Build to the highest note of the song... on a weak syllable, quickly",
            "The big moment arrives as a whisper, not a shout",
            "Melodic peak is a question (rising) that answers itself weakly (falling)",
        ],
        performance_notes=[
            "The deflation should feel like disappointment, resignation, or acceptance",
            "Energy must build authentically so the anticlimax lands",
            "Don't telegraph the anticlimax - let it surprise",
            "The weak resolution should feel inevitable in retrospect",
        ],
        rule_broken="MELODY_AntiClimax",
        rule_effect="Disappointment, deflation, resignation - failed expectations",
    )


def generate_melody_monotone_drone(key: str, mode: str = "major") -> GeneratedMelody:
    """
    MELODY_MonotoneDrone
    Minimal melodic movement, near-monotone for numbness.
    """
    return GeneratedMelody(
        contour="Flat line - horizontal, minimal pitch change",
        interval_character="Unisons and minor 2nds only - micro-movements",
        phrase_structure="Continuous drone with tiny inflections",
        resolution_behavior="No resolution because no tension created",
        rhythmic_character="Speech rhythms over static pitch",
        range_notes="2-3 note range maximum - intentionally limited",
        motif_ideas=[
            "Entire verse on a single note with occasional half-step dips",
            "Recitation tone - like chanting or praying",
            "Harmony moves underneath static melody - world shifts, voice doesn't",
            "Micro-ornaments (quarter-tone bends) are the only 'melody'",
        ],
        performance_notes=[
            "The monotone is dissociation made audible",
            "Emotion comes through timbre and dynamics, not pitch",
            "Consider the state between speaking and singing",
            "Numbness, meditation, or shutdown - voice has given up expression",
        ],
        rule_broken="MELODY_MonotoneDrone",
        rule_effect="Numbness, dissociation, meditation - emotional shutdown",
    )


def generate_melody_fragmented_phrases(key: str, mode: str = "major") -> GeneratedMelody:
    """
    MELODY_FragmentedPhrases
    Break melody into disconnected fragments for fractured thought.
    """
    return GeneratedMelody(
        contour="Interrupted, stop-start, phrases cut short",
        interval_character="Normal intervals but phrases never complete",
        phrase_structure="2-3 words then silence, restart differently",
        resolution_behavior="Phrases end abruptly before resolution possible",
        rhythmic_character="Irregular, gasping, catching breath",
        range_notes="Full range but used in disconnected bursts",
        motif_ideas=[
            "Sentence melody cut mid-word, resume somewhere else",
            "Silences are as long as the sung fragments",
            "Each fragment in different part of range - no continuity",
            "The melody keeps trying to start but can't sustain",
        ],
        performance_notes=[
            "Sound like someone trying to speak through difficulty",
            "Interruptions should feel involuntary, not artistic",
            "Gasps, false starts, abandoned thoughts",
            "The fragmentation IS the trauma made audible",
        ],
        rule_broken="MELODY_FragmentedPhrases",
        rule_effect="Fractured thought, interrupted speech, trauma - difficulty expressing",
    )
