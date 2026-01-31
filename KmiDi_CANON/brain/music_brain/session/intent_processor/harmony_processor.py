"""
Harmony processors for generating chord progressions with intentional rule-breaking.

This module contains functions that generate chord progressions that break
traditional harmonic rules for emotional and creative effect.
"""

import random
from .base import GeneratedProgression, _romans_to_chords


def generate_progression_avoid_tonic(key: str, mode: str = "major") -> GeneratedProgression:
    """
    HARMONY_AvoidTonicResolution
    Generate progression that resolves to IV or VI instead of I.
    """
    if mode == "major":
        # End on IV instead of I
        progressions = [
            (['I', 'V', 'vi', 'IV'], "Axis progression ending on IV - unresolved yearning"),
            (['I', 'IV', 'V', 'IV'], "Classic with IV ending - perpetual motion"),
            (['vi', 'IV', 'I', 'vi'], "Start and end on vi - melancholy cycle"),
            (['I', 'V', 'IV', 'vi'], "Deceptive to vi - the hope never lands"),
        ]
    else:
        progressions = [
            (['i', 'VI', 'III', 'VII'], "Minor with bVII ending"),
            (['i', 'iv', 'VI', 'iv'], "Cycling minor, never resolves"),
        ]

    choice = random.choice(progressions)
    romans, effect = choice

    # Convert to actual chords
    chords = _romans_to_chords(romans, key, mode)

    return GeneratedProgression(
        chords=chords,
        key=key,
        mode=mode,
        roman_numerals=romans,
        rule_broken="HARMONY_AvoidTonicResolution",
        rule_effect=effect,
        emotional_arc=["stable", "building", "reaching", "suspended"],
    )


def generate_progression_modal_interchange(key: str, mode: str = "major") -> GeneratedProgression:
    """
    HARMONY_ModalInterchange
    Insert chord borrowed from parallel or related mode.
    """
    if mode == "major":
        # Borrow from parallel minor
        progressions = [
            (['I', 'V', 'iv', 'I'], "iv borrowed from minor - instant melancholy"),
            (['I', 'bVI', 'IV', 'I'], "bVI epic chord - cinematic arrival"),
            (['I', 'IV', 'bVII', 'I'], "bVII rock swagger - avoids cliché V"),
            (['I', 'bIII', 'IV', 'V'], "bIII brightness from minor - unexpected color"),
            (['I', 'V', 'bVI', 'bVII'], "Double borrowed - emotional journey"),
        ]
    else:
        # In minor, borrow from major
        progressions = [
            (['i', 'IV', 'V', 'i'], "Major IV (Dorian) - hope in darkness"),
            (['i', 'bVI', 'III', 'VII'], "Natural minor with major III"),
        ]

    choice = random.choice(progressions)
    romans, effect = choice
    chords = _romans_to_chords(romans, key, mode)

    return GeneratedProgression(
        chords=chords,
        key=key,
        mode=mode,
        roman_numerals=romans,
        rule_broken="HARMONY_ModalInterchange",
        rule_effect=effect,
        emotional_arc=["grounded", "questioning", "shifted", "returned"],
        voice_leading_notes=["Watch chromatic movement in borrowed chord"],
    )


def generate_progression_parallel_motion(key: str, mode: str = "major") -> GeneratedProgression:
    """
    HARMONY_ParallelMotion
    Force parallel 5ths/octaves - power chord style.
    """
    # Power chord progressions
    progressions = [
        (['I5', 'bVII5', 'IV5', 'I5'], "Classic rock parallel 5ths"),
        (['I5', 'IV5', 'V5', 'IV5'], "Power ballad motion"),
        (['i5', 'bVII5', 'bVI5', 'V5'], "Metal descent"),
        (['I5', 'bIII5', 'IV5', 'V5'], "Punk parallel climb"),
    ]

    choice = random.choice(progressions)
    romans, effect = choice
    chords = _romans_to_chords(romans, key, mode)

    return GeneratedProgression(
        chords=chords,
        key=key,
        mode=mode,
        roman_numerals=romans,
        rule_broken="HARMONY_ParallelMotion",
        rule_effect=effect,
        emotional_arc=["power", "defiance", "momentum", "landing"],
        voice_leading_notes=["All voices move in parallel - intentional fusion"],
    )


def generate_progression_unresolved_dissonance(key: str, mode: str = "major") -> GeneratedProgression:
    """
    HARMONY_UnresolvedDissonance
    Leave 7ths, 9ths, tritones hanging.
    """
    progressions = [
        (['Imaj7', 'IVmaj7', 'viim7b5', 'IVmaj7'], "All 7ths, ends on IV7"),
        (['Imaj9', 'vim7', 'IVadd9', 'Vsus4'], "Extensions and sus - nothing fully resolves"),
        (['Im7', 'bVImaj7', 'IVm7', 'bVII7'], "Minor 7th chain - perpetual tension"),
    ]

    choice = random.choice(progressions)
    romans, effect = choice
    chords = _romans_to_chords(romans, key, mode)

    return GeneratedProgression(
        chords=chords,
        key=key,
        mode=mode,
        roman_numerals=romans,
        rule_broken="HARMONY_UnresolvedDissonance",
        rule_effect=effect,
        emotional_arc=["questioning", "reaching", "suspended", "lingering"],
    )


def generate_progression_tritone_substitution(key: str, mode: str = "major") -> GeneratedProgression:
    """
    HARMONY_TritoneSubstitution
    Replace V7 with bII7 for chromatic bass movement.
    """
    progressions = [
        (['Imaj7', 'vim7', 'bII7', 'Imaj7'], "bII7 replaces V7 - chromatic resolution"),
        (['Imaj7', 'IVmaj7', 'bII7', 'I6'], "Tritone sub before tonic - jazz sophistication"),
        (['iim7', 'bII7', 'Imaj7', 'vim7'], "ii-V becomes ii-bII - smooth chromatic bass"),
        (['Imaj7', 'bVI7', 'bII7', 'I'], "Double tritone subs - maximum color"),
    ]

    choice = random.choice(progressions)
    romans, effect = choice
    chords = _romans_to_chords(romans, key, mode)

    return GeneratedProgression(
        chords=chords,
        key=key,
        mode=mode,
        roman_numerals=romans,
        rule_broken="HARMONY_TritoneSubstitution",
        rule_effect=effect,
        emotional_arc=["grounded", "tension", "chromatic", "resolution"],
        voice_leading_notes=["Bass moves by half-step - emphasize this chromatic movement"],
    )


def generate_progression_polytonality(key: str, mode: str = "major") -> GeneratedProgression:
    """
    HARMONY_Polytonality
    Stack chords from different keys for tension and disorientation.
    """
    # Express as slash chords or compound chords
    progressions = [
        (['I/bII', 'IV/V', 'bVI/bVII', 'I'], "Polytonal clashes - internal conflict"),
        (['Imaj7#11', 'bVImaj7#11', 'IVmaj7#11', 'V7#9'], "Lydian stacks - dreamlike dissonance"),
        (['I', 'I+bV', 'IV', 'IV+bI'], "Bitonal moments - reality shifting"),
        (['Im', 'IM/Im', 'IVm/IVM', 'Vm'], "Major/minor superimposition - emotional duality"),
    ]

    choice = random.choice(progressions)
    romans, effect = choice
    chords = _romans_to_chords(romans, key, mode)

    return GeneratedProgression(
        chords=chords,
        key=key,
        mode=mode,
        roman_numerals=romans,
        rule_broken="HARMONY_Polytonality",
        rule_effect=effect,
        emotional_arc=["stable", "fractured", "disoriented", "reintegrated"],
        voice_leading_notes=["Let the clash be heard - don't bury it in reverb"],
    )
