"""
Texture processors for generating sonic texture guidelines with intentional rule-breaking.

This module contains functions that generate texture patterns that break
traditional timbral and layering rules for emotional and creative effect.
"""

from .base import GeneratedTexture


def generate_texture_frequency_masking() -> GeneratedTexture:
    """
    TEXTURE_FrequencyMasking
    Let elements fight for same frequencies for crowded thoughts.
    """
    return GeneratedTexture(
        density_level="Dense - intentionally too many elements",
        frequency_balance="Mid-heavy - everything competing for 200Hz-2kHz",
        element_roles=[
            {"element": "Guitar 1", "role": "Occupies 200-600Hz", "notes": "Don't carve space"},
            {"element": "Guitar 2", "role": "Also 200-600Hz", "notes": "Let them fight"},
            {"element": "Keys", "role": "300-800Hz", "notes": "Adds to the pile"},
            {"element": "Vocal", "role": "Buried in the conflict", "notes": "Fighting to be heard"},
            {"element": "Bass", "role": "Bleeds into mids", "notes": "No high-pass, let it mud"},
        ],
        space_character="Claustrophobic, no separation between elements",
        timbre_notes=[
            "Similar timbres that blend/fight rather than contrast",
            "Avoid clarity - clarity is not the goal here",
            "The listener should work to hear individual elements",
        ],
        interaction_notes=[
            "Elements step on each other - no polite frequency carving",
            "Ducking/sidechaining would solve this - DON'T use it",
            "The chaos is the internal monologue made sonic",
        ],
        rule_broken="TEXTURE_FrequencyMasking",
        rule_effect="Crowded, competitive, overwhelming - internal voices all talking at once",
    )


def generate_texture_sparse_emptiness() -> GeneratedTexture:
    """
    TEXTURE_SparseEmptiness
    Extreme space between elements for isolation.
    """
    return GeneratedTexture(
        density_level="Skeletal - barely anything present",
        frequency_balance="Huge gaps in spectrum - isolated pockets of sound",
        element_roles=[
            {"element": "Voice", "role": "Solo presence", "notes": "Exposed, nowhere to hide"},
            {"element": "Single instrument", "role": "Occasional support", "notes": "Long gaps between entries"},
            {"element": "Space/silence", "role": "Primary element", "notes": "The emptiness IS the texture"},
        ],
        space_character="Vast, empty, isolating - reverb emphasizes loneliness",
        timbre_notes=[
            "Each sound is precious because it's surrounded by nothing",
            "Imperfections are magnified - let them be heard",
            "The voice has nothing to lean on",
        ],
        interaction_notes=[
            "Elements rarely overlap - isolation is maintained",
            "When elements do meet, it should feel significant",
            "The space between notes is as composed as the notes",
        ],
        rule_broken="TEXTURE_SparseEmptiness",
        rule_effect="Isolation, exposure, vulnerability - nowhere to hide",
    )


def generate_texture_dense_wall() -> GeneratedTexture:
    """
    TEXTURE_DenseWall
    Stack elements into undifferentiated mass for overwhelming force.
    """
    return GeneratedTexture(
        density_level="Overwhelming - too much, intentionally",
        frequency_balance="Full spectrum saturation - every frequency filled",
        element_roles=[
            {"element": "Bass", "role": "Foundation extending into mids", "notes": "Massive"},
            {"element": "Guitars/Keys", "role": "Wall of sound", "notes": "Layered, doubled, stacked"},
            {"element": "Drums", "role": "Constant presence", "notes": "Not leading, supporting the mass"},
            {"element": "Vocal", "role": "Part of the wall OR fighting through", "notes": "Choose one"},
            {"element": "Additional layers", "role": "Fill every remaining gap", "notes": "Pads, drones, noise"},
        ],
        space_character="No space - the wall is solid",
        timbre_notes=[
            "Individual elements lose identity in the mass",
            "The wall becomes its own instrument",
            "Saturation and compression to glue everything together",
        ],
        interaction_notes=[
            "No element should be individually distinguishable",
            "The listener is swept up, not analyzing",
            "Catharsis through overwhelming sonic force",
        ],
        rule_broken="TEXTURE_DenseWall",
        rule_effect="Overwhelming force, loss of self, catharsis - swept away by sound",
    )


def generate_texture_conflicting_timbres() -> GeneratedTexture:
    """
    TEXTURE_ConflictingTimbres
    Combine timbres that traditionally clash for wrongness.
    """
    return GeneratedTexture(
        density_level="Moderate - conflict needs space to be heard",
        frequency_balance="Elements occupying similar ranges with different timbres",
        element_roles=[
            {"element": "Acoustic instrument", "role": "Natural, organic", "notes": "Warm, imperfect"},
            {"element": "Harsh synth", "role": "Cold, synthetic", "notes": "Bright, processed"},
            {"element": "Lo-fi element", "role": "Degraded quality", "notes": "Tape, bitcrushed"},
            {"element": "Clean element", "role": "Pristine production", "notes": "Too clean"},
        ],
        space_character="Each element in its own 'world' - they don't belong together",
        timbre_notes=[
            "The conflict should feel wrong, uncomfortable",
            "Don't blend - let the clash be audible",
            "Mix production eras: 60s with modern digital, acoustic with synthetic",
        ],
        interaction_notes=[
            "Elements coexist but don't converse",
            "The wrongness mirrors emotional dissonance",
            "Things that shouldn't be together, are",
        ],
        rule_broken="TEXTURE_ConflictingTimbres",
        rule_effect="Dissonance, wrongness, tension - things that don't belong together",
    )


def generate_texture_single_element_focus() -> GeneratedTexture:
    """
    TEXTURE_SingleElementFocus
    Strip away all but one element for stark truth.
    """
    return GeneratedTexture(
        density_level="Solo - one element carries everything",
        frequency_balance="Only what that single element provides",
        element_roles=[
            {"element": "The chosen one", "role": "Everything", "notes": "Voice OR single instrument"},
            {"element": "Nothing else", "role": "Absence", "notes": "The support doesn't exist"},
        ],
        space_character="Raw, exposed - the element and the room/void",
        timbre_notes=[
            "Every nuance of the single element is exposed",
            "Imperfections become features - nothing to hide behind",
            "The timbre IS the arrangement",
        ],
        interaction_notes=[
            "No interaction - monologue, not dialogue",
            "If anything else enters, it should be devastating",
            "Confession, revelation - the moment of truth",
        ],
        rule_broken="TEXTURE_SingleElementFocus",
        rule_effect="Stark truth, nowhere to hide, confession - naked honesty",
    )


def generate_texture_timbral_drift() -> GeneratedTexture:
    """
    TEXTURE_TimbralDrift
    Gradually morph timbre over time for transformation.
    """
    return GeneratedTexture(
        density_level="Variable - changes with the drift",
        frequency_balance="Shifting over time - what was low becomes high, etc.",
        element_roles=[
            {"element": "Primary voice/instrument", "role": "The thing that transforms", "notes": "Clear at start"},
            {"element": "Processing", "role": "Gradual morphing agent", "notes": "Automation over minutes"},
            {"element": "Environment", "role": "Also shifts", "notes": "Reverb, space changes"},
        ],
        space_character="Evolving - intimate to vast, or vice versa",
        timbre_notes=[
            "Start clean, end destroyed (or reverse)",
            "The transformation should be slow enough to be subliminal",
            "By the end, the sound should be unrecognizable from the start",
        ],
        interaction_notes=[
            "Other elements may drift at different rates",
            "Asynchronous drifting creates unease",
            "The listener realizes change has happened, not as it happens",
        ],
        rule_broken="TEXTURE_TimbralDrift",
        rule_effect="Transformation, unease, evolution - change happening beneath notice",
    )
