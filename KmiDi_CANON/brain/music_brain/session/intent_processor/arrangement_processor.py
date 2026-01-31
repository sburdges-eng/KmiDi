"""
Arrangement processors for generating song structure and production guidelines.

This module contains functions that generate arrangement and production plans
that break traditional structural and mixing rules for emotional and creative effect.
"""

from .base import GeneratedArrangement, GeneratedProduction
from music_brain.session.intent_schema import RULE_BREAKING_EFFECTS


def generate_arrangement_structural_mismatch(narrative_arc: str) -> GeneratedArrangement:
    """
    ARRANGEMENT_StructuralMismatch
    Use unexpected structure for genre.
    """
    if narrative_arc == "Sudden Shift":
        # Long build, immediate payoff, then reflection
        sections = [
            {"name": "Intro", "bars": 8, "energy": 0.3, "notes": "Restrained, building"},
            {"name": "Verse 1", "bars": 16, "energy": 0.4, "notes": "Constrained energy"},
            {"name": "Build", "bars": 8, "energy": 0.7, "notes": "Rising tension"},
            {"name": "DROP", "bars": 4, "energy": 1.0, "notes": "THE SHIFT - maximum impact"},
            {"name": "Release", "bars": 16, "energy": 0.6, "notes": "Aftermath, processing"},
            {"name": "Outro", "bars": 8, "energy": 0.3, "notes": "Gentle landing"},
        ]
        arc = [0.3, 0.4, 0.7, 1.0, 0.6, 0.3]

    elif narrative_arc == "Slow Reveal":
        # Through-composed, no repetition
        sections = [
            {"name": "Movement I", "bars": 16, "energy": 0.3, "notes": "Introduction of theme"},
            {"name": "Movement II", "bars": 16, "energy": 0.5, "notes": "Development"},
            {"name": "Movement III", "bars": 12, "energy": 0.7, "notes": "Complication"},
            {"name": "Movement IV", "bars": 8, "energy": 0.4, "notes": "The reveal"},
            {"name": "Coda", "bars": 8, "energy": 0.2, "notes": "Resolution"},
        ]
        arc = [0.3, 0.5, 0.7, 0.4, 0.2]

    elif narrative_arc == "Repetitive Despair":
        # Same section repeating with minor variations
        sections = [
            {"name": "Loop A", "bars": 8, "energy": 0.5, "notes": "The cycle begins"},
            {"name": "Loop A'", "bars": 8, "energy": 0.55, "notes": "Slight variation"},
            {"name": "Loop A''", "bars": 8, "energy": 0.6, "notes": "Building frustration"},
            {"name": "Loop A'''", "bars": 8, "energy": 0.5, "notes": "Back to start - trapped"},
            {"name": "Loop A''''", "bars": 8, "energy": 0.45, "notes": "Fading energy"},
        ]
        arc = [0.5, 0.55, 0.6, 0.5, 0.45]

    else:  # Default Climb-to-Climax
        sections = [
            {"name": "Intro", "bars": 4, "energy": 0.2, "notes": "Minimal"},
            {"name": "Verse", "bars": 16, "energy": 0.4, "notes": "Building"},
            {"name": "Pre-Chorus", "bars": 8, "energy": 0.6, "notes": "Rising"},
            {"name": "Chorus", "bars": 16, "energy": 0.8, "notes": "Arrival"},
            {"name": "Bridge", "bars": 8, "energy": 0.5, "notes": "Brief retreat"},
            {"name": "Final Chorus", "bars": 16, "energy": 1.0, "notes": "Peak"},
            {"name": "Outro", "bars": 8, "energy": 0.3, "notes": "Descent"},
        ]
        arc = [0.2, 0.4, 0.6, 0.8, 0.5, 1.0, 0.3]

    return GeneratedArrangement(
        sections=sections,
        dynamic_arc=arc,
        rule_broken="ARRANGEMENT_StructuralMismatch",
        rule_effect="Structure serves the story, not genre convention",
    )


def generate_arrangement_extreme_dynamics() -> GeneratedArrangement:
    """
    ARRANGEMENT_ExtremeDynamicRange
    Exceed normal dynamic limits for dramatic impact.
    """
    sections = [
        {"name": "Whisper", "bars": 8, "energy": 0.1, "notes": "Nearly silent, intimate"},
        {"name": "Build", "bars": 8, "energy": 0.4, "notes": "Gradual increase"},
        {"name": "EXPLOSION", "bars": 4, "energy": 1.0, "notes": "Maximum possible volume"},
        {"name": "Silence", "bars": 2, "energy": 0.0, "notes": "Complete stop"},
        {"name": "Resolution", "bars": 16, "energy": 0.5, "notes": "Normal level feels loud after silence"},
    ]

    return GeneratedArrangement(
        sections=sections,
        dynamic_arc=[0.1, 0.4, 1.0, 0.0, 0.5],
        rule_broken="ARRANGEMENT_ExtremeDynamicRange",
        rule_effect="The silence after the explosion is deafening",
    )


def generate_arrangement_unbalanced_dynamics() -> GeneratedArrangement:
    """
    ARRANGEMENT_UnbalancedDynamics
    Keep specific element intentionally too loud or quiet throughout.
    """
    sections = [
        {
            "name": "Intro",
            "bars": 8,
            "energy": 0.5,
            "notes": "Establish the imbalance immediately",
            "mix_notes": "Bass is 6dB too loud - intentional weight"
        },
        {
            "name": "Verse",
            "bars": 16,
            "energy": 0.5,
            "notes": "Maintain the same imbalance",
            "mix_notes": "Don't fix the bass - the weight IS the point"
        },
        {
            "name": "Chorus",
            "bars": 8,
            "energy": 0.7,
            "notes": "Increase energy but keep imbalance",
            "mix_notes": "Everything else comes up, bass stays dominant"
        },
        {
            "name": "Bridge",
            "bars": 8,
            "energy": 0.4,
            "notes": "Brief moment of balance",
            "mix_notes": "Balance temporarily - makes return to imbalance more impactful"
        },
        {
            "name": "Final",
            "bars": 8,
            "energy": 0.8,
            "notes": "Return to imbalance, even more extreme",
            "mix_notes": "Bass now 8dB too loud - obsession made sonic"
        },
    ]

    return GeneratedArrangement(
        sections=sections,
        dynamic_arc=[0.5, 0.5, 0.7, 0.4, 0.8],
        rule_broken="ARRANGEMENT_UnbalancedDynamics",
        rule_effect="The imbalance creates obsessive focus - one element dominates attention",
    )


def generate_arrangement_buried_vocals() -> GeneratedArrangement:
    """
    ARRANGEMENT_BuriedVocals
    Place vocals intentionally below/behind instruments.
    """
    sections = [
        {
            "name": "Intro",
            "bars": 8,
            "energy": 0.4,
            "notes": "Instrumental setup",
            "vocal_level": "none"
        },
        {
            "name": "Verse 1",
            "bars": 16,
            "energy": 0.5,
            "notes": "Vocals emerge but buried",
            "vocal_level": "-6dB from instruments",
            "vocal_notes": "Vocal is texture, not focus. Words half-heard."
        },
        {
            "name": "Chorus",
            "bars": 8,
            "energy": 0.7,
            "notes": "Vocals rise slightly but never dominate",
            "vocal_level": "-3dB from instruments",
            "vocal_notes": "Still fighting through - intimacy through distance"
        },
        {
            "name": "Verse 2",
            "bars": 16,
            "energy": 0.5,
            "notes": "Return to deeply buried",
            "vocal_level": "-8dB from instruments",
            "vocal_notes": "Deeper burial - dissociation intensifies"
        },
        {
            "name": "Bridge - Exposed",
            "bars": 8,
            "energy": 0.3,
            "notes": "Strip away instruments - vocal finally clear",
            "vocal_level": "0dB - solo or near-solo",
            "vocal_notes": "The one moment of clarity - devastating impact"
        },
        {
            "name": "Final Chorus",
            "bars": 8,
            "energy": 0.8,
            "notes": "Instruments return, vocal buried again",
            "vocal_level": "-6dB from instruments",
            "vocal_notes": "Return to burial after clarity - the cost of vulnerability"
        },
    ]

    return GeneratedArrangement(
        sections=sections,
        dynamic_arc=[0.4, 0.5, 0.7, 0.5, 0.3, 0.8],
        rule_broken="ARRANGEMENT_BuriedVocals",
        rule_effect="Intimacy through distance - forcing the listener to lean in",
    )


def generate_arrangement_premature_climax() -> GeneratedArrangement:
    """
    ARRANGEMENT_PrematureClimax
    Put the emotional/sonic peak earlier than expected, then deal with aftermath.
    """
    sections = [
        {
            "name": "Intro",
            "bars": 4,
            "energy": 0.3,
            "notes": "Brief setup"
        },
        {
            "name": "Build",
            "bars": 8,
            "energy": 0.6,
            "notes": "Rapid escalation"
        },
        {
            "name": "CLIMAX",
            "bars": 4,
            "energy": 1.0,
            "notes": "THE PEAK - happens at 30% through the song, not 75%"
        },
        {
            "name": "Aftermath I",
            "bars": 16,
            "energy": 0.5,
            "notes": "Processing what just happened"
        },
        {
            "name": "Aftermath II",
            "bars": 16,
            "energy": 0.4,
            "notes": "Continued reflection, gradual descent"
        },
        {
            "name": "Resolution",
            "bars": 8,
            "energy": 0.3,
            "notes": "Quiet ending - the aftermath IS the story"
        },
    ]

    return GeneratedArrangement(
        sections=sections,
        dynamic_arc=[0.3, 0.6, 1.0, 0.5, 0.4, 0.3],
        rule_broken="ARRANGEMENT_PrematureClimax",
        rule_effect="The aftermath is the story - what happens after the moment of impact",
    )


def generate_production_guidelines(
    rule_to_break: str,
    vulnerability: str,
    imagery: str
) -> GeneratedProduction:
    """Generate production guidelines based on intent."""

    # Base guidelines
    eq_notes = []
    dynamics_notes = []
    space_notes = []
    vocal_treatment = ""

    # Rule-specific modifications
    if rule_to_break == "PRODUCTION_ExcessiveMud":
        eq_notes = [
            "DO NOT cut 200-400Hz - let the mud exist",
            "The weight is the point",
            "Consider BOOSTING low-mids for claustrophobia",
        ]
        dynamics_notes = ["Heavy compression to emphasize density"]
        space_notes = ["Minimal reverb - keep it close and suffocating"]
        vocal_treatment = "Slightly buried, fighting through the mud"

    elif rule_to_break == "PRODUCTION_PitchImperfection":
        eq_notes = ["Natural, minimal processing"]
        dynamics_notes = ["Light compression to preserve dynamics"]
        space_notes = ["Room sound acceptable"]
        vocal_treatment = "NO pitch correction - the drift IS the emotion"

    elif rule_to_break == "PRODUCTION_BuriedVocals":
        eq_notes = ["Roll off some highs on vocal for distance"]
        dynamics_notes = ["Compress heavily to make it part of the texture"]
        space_notes = ["Heavy reverb on vocal, less on instruments"]
        vocal_treatment = "Sit BEHIND the instruments - intimacy through distance"

    elif rule_to_break == "PRODUCTION_RoomNoise":
        eq_notes = ["Don't filter out room tone"]
        dynamics_notes = ["Let natural dynamics exist"]
        space_notes = ["The room IS the reverb"]
        vocal_treatment = "Record in the space, not the booth"

    elif rule_to_break == "PRODUCTION_Distortion":
        eq_notes = ["Saturate the mids", "Let it clip intentionally"]
        dynamics_notes = ["Crush the dynamics on specific elements"]
        space_notes = ["Distortion provides its own 'space'"]
        vocal_treatment = "Consider vocal distortion at emotional peaks"

    elif rule_to_break == "PRODUCTION_MonoCollapse":
        eq_notes = ["Check in mono frequently", "Bass and kick center"]
        dynamics_notes = ["Standard"]
        space_notes = ["Narrow stereo field intentionally", "Creates claustrophobia"]
        vocal_treatment = "Dead center, no width"

    elif rule_to_break == "PRODUCTION_LoFiDegradation":
        eq_notes = [
            "Roll off highs aggressively (low-pass around 8-10kHz)",
            "Add subtle low-pass filter modulation for tape wobble",
            "Consider bit-crushing for digital artifacts",
        ]
        dynamics_notes = [
            "Heavy compression with slow attack - pump effect",
            "Let the compression artifacts be audible",
        ]
        space_notes = [
            "Use degraded reverb - spring or plate with noise",
            "Add vinyl crackle or tape hiss as texture layer",
            "Keep it close - lo-fi is intimate",
        ]
        vocal_treatment = "Process through tape emulation or bit crusher - imperfection is memory"

    elif rule_to_break == "PRODUCTION_SilenceAsInstrument":
        eq_notes = ["Standard - make the sound clear so its absence is felt"]
        dynamics_notes = [
            "Use hard gates for sudden dropouts",
            "Automate volume to zero - not fade, CUT",
            "The silence must be absolute, not quiet",
        ]
        space_notes = [
            "Reverb tails should cut with the sound",
            "No ambient bed during silence sections",
            "Consider room tone only - dead silence is unsettling",
        ]
        vocal_treatment = "Cut mid-word for maximum impact - the unfinished thought"

    elif rule_to_break == "PRODUCTION_ClippingPeaks":
        eq_notes = [
            "Don't fix the clipping - it's intentional",
            "Emphasize the clipped frequencies",
            "Consider adding harmonic saturation to spread the damage",
        ]
        dynamics_notes = [
            "Push into the red intentionally",
            "Use hard clipping, not soft limiting",
            "Let transients clip - the crack is the emotion",
        ]
        space_notes = [
            "Dry signal clips better - reverb softens the edge",
            "Consider clipping the reverb return separately",
        ]
        vocal_treatment = "Allow vocal peaks to clip at emotional climax - the voice breaking"

    else:
        # Default based on vulnerability
        if vulnerability == "High":
            eq_notes = ["Gentle, natural EQ", "Don't over-polish"]
            dynamics_notes = ["Preserve natural dynamics"]
            space_notes = ["Intimate reverb, not concert hall"]
            vocal_treatment = "Present but not 'produced'"
        else:
            eq_notes = ["Standard mixing practices"]
            dynamics_notes = ["Appropriate compression"]
            space_notes = ["Genre-appropriate space"]
            vocal_treatment = "Clear and present"

    # Imagery texture modifications
    if "vast" in imagery.lower() or "open" in imagery.lower():
        space_notes.append("Wide stereo field")
        space_notes.append("Long reverb tails")
    elif "muffled" in imagery.lower():
        eq_notes.append("Roll off highs aggressively")
        space_notes.append("Distant, filtered reverb")
    elif "sharp" in imagery.lower():
        eq_notes.append("Emphasize presence frequencies (2-5kHz)")
        dynamics_notes.append("Fast attack compression")

    return GeneratedProduction(
        eq_notes=eq_notes,
        dynamics_notes=dynamics_notes,
        space_notes=space_notes,
        vocal_treatment=vocal_treatment,
        rule_broken=rule_to_break,
        rule_effect=RULE_BREAKING_EFFECTS.get(rule_to_break, {}).get("effect", ""),
    )
