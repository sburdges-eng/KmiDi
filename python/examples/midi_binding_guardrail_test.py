"""
MIDI binding guardrail: notes, velocity, timing, branch count must be fixed.
They must NEVER be derived from intent, value states, escalation, or constraints.
If someone ties MIDI to intent here, CI fails.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui_gate.midi_binding import (
    BRANCH_MIDI_NOTES,
    BRANCH_MIDI_NOTES_VARIANT,
    FIXED_VELOCITY,
    FIXED_DURATION_MS,
    FIXED_DURATION_MS_VARIANT,
    EXPECTED_BRANCH_COUNT,
    get_midi_for_branch,
)


def test_branch_count_fixed() -> None:
    """Branch count must equal expected. No dynamic derivation."""
    assert len(BRANCH_MIDI_NOTES) == EXPECTED_BRANCH_COUNT, (
        f"Branch count {len(BRANCH_MIDI_NOTES)} != expected {EXPECTED_BRANCH_COUNT}"
    )


def test_midi_notes_fixed() -> None:
    """MIDI notes must be fixed per branch. Not computed from intent."""
    assert BRANCH_MIDI_NOTES[0] == (60, 64, 67)
    assert BRANCH_MIDI_NOTES[1] == (62, 65, 69)
    assert BRANCH_MIDI_NOTES[2] == (65, 69, 72)


def test_midi_notes_variant_fixed() -> None:
    """Pitch variation mapping must be fixed. First inversions."""
    assert BRANCH_MIDI_NOTES_VARIANT[0] == (64, 67, 72)
    assert BRANCH_MIDI_NOTES_VARIANT[1] == (65, 69, 74)
    assert BRANCH_MIDI_NOTES_VARIANT[2] == (69, 72, 77)


def test_timing_variant_fixed() -> None:
    """Timing variation duration must be fixed. Symbolic only."""
    assert FIXED_DURATION_MS == 400
    assert FIXED_DURATION_MS_VARIANT == 600


def test_velocity_constant() -> None:
    """Velocity must be constant. No shaping or humanization."""
    for i in range(EXPECTED_BRANCH_COUNT):
        for pv in (False, True):
            for tv in (False, True):
                events = get_midi_for_branch(i, pitch_variation=pv, timing_variation=tv)
                for _pitch, vel, _dur in events:
                    assert vel == FIXED_VELOCITY, f"Velocity {vel} != constant {FIXED_VELOCITY}"


def test_timing_constant() -> None:
    """Duration must be fixed per path: 400ms default, 600ms when timing_variation."""
    for i in range(EXPECTED_BRANCH_COUNT):
        for pv in (False, True):
            for tv in (False, True):
                events = get_midi_for_branch(i, pitch_variation=pv, timing_variation=tv)
                expected_dur = FIXED_DURATION_MS_VARIANT if tv else FIXED_DURATION_MS
                for _pitch, _vel, dur in events:
                    assert dur == expected_dur, (
                        f"Duration {dur} != expected {expected_dur} (tv={tv})"
                    )


def test_same_input_same_output() -> None:
    """Deterministic: same branch + pitch_variation + timing_variation → same MIDI."""
    for i in range(EXPECTED_BRANCH_COUNT):
        for pv in (False, True):
            for tv in (False, True):
                a = get_midi_for_branch(i, pitch_variation=pv, timing_variation=tv)
                b = get_midi_for_branch(i, pitch_variation=pv, timing_variation=tv)
                assert a == b, f"Branch {i} pv={pv} tv={tv}: non-deterministic output"


def main() -> int:
    try:
        test_branch_count_fixed()
        test_midi_notes_fixed()
        test_midi_notes_variant_fixed()
        test_timing_variant_fixed()
        test_velocity_constant()
        test_timing_constant()
        test_same_input_same_output()
        print("midi binding guardrail: PASS")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
