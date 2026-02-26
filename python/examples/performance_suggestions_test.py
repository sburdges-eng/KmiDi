"""
C2 — Performance Suggestions acceptance tests.
AC-C2-1 through AC-C2-10.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minimal_listening_core import (
    AbstractInput,
    IntentAspect,
    intent_engine_process,
)
from minimal_listening_core.performance_suggestions import (
    ChordMaterial,
    request_suggestions,
    density_unchanged,
)


def _abstain_result():
    inp = AbstractInput(input_type="", aspect=IntentAspect.HARMONIC, magnitude=0.0)
    return intent_engine_process(inp, None)


def _valid_result():
    inp = AbstractInput(input_type="chord_change", aspect=IntentAspect.HARMONIC, magnitude=0.7)
    return intent_engine_process(inp, None)


def _two_chords() -> list:
    return [
        ChordMaterial(pitches=(60, 64, 67), start_beat=0.0, duration=4.0),
        ChordMaterial(pitches=(55, 59, 62), start_beat=4.0, duration=4.0),
    ]


def test_ac_c2_1_exactly_2_or_3_variations() -> int:
    """AC-C2-1: When user requests suggestions with valid input, return exactly 2 or 3 variations."""
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=True)
    assert 2 <= len(variations) <= 3, f"Expected 2-3 variations, got {len(variations)}"
    return 0


def test_ac_c2_2_same_chord_count() -> int:
    """AC-C2-2: Each variation has the same chord count as the input material."""
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=True)
    for var in variations:
        assert len(var) == len(chords), f"Expected {len(chords)} chords, got {len(var)}"
    return 0


def test_ac_c2_3_same_rhythmic_density() -> int:
    """AC-C2-3: Each variation has the same rhythmic density (notes per beat)."""
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=True)
    for var in variations:
        for vc, ic in zip(var, chords):
            assert vc.note_count() == ic.note_count(), "Note count must not increase"
    return 0


def test_ac_c2_4_same_tempo() -> int:
    """AC-C2-4: Each variation has the same tempo (duration preserved)."""
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=True)
    for var in variations:
        for vc, ic in zip(var, chords):
            assert vc.duration == ic.duration and vc.start_beat == ic.start_beat
    return 0


def test_ac_c2_5_no_added_notes() -> int:
    """AC-C2-5: No variation adds notes; only voicing/inversion changes."""
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=True)
    for var in variations:
        for vc, ic in zip(var, chords):
            assert len(vc.pitches) == len(ic.pitches), "Must not add notes"
    return 0


def test_ac_c2_6_opt_in_only() -> int:
    """AC-C2-6: Suggestions never produced automatically; require explicit user request."""
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=False)
    assert variations == [], "No suggestions without user request"
    return 0


def test_ac_c2_7_abstain_returns_empty() -> int:
    """AC-C2-7: When input is empty or ABSTAIN, request for suggestions returns empty."""
    chords = _two_chords()
    abstain = _abstain_result()
    variations = request_suggestions(chords, abstain, user_requested=True)
    assert variations == [], "ABSTAIN must return no suggestions"
    return 0


def test_ac_c2_7b_empty_chords_returns_empty() -> int:
    """AC-C2-7: Empty chords → empty suggestions."""
    result = _valid_result()
    variations = request_suggestions([], result, user_requested=True)
    assert variations == [], "Empty chords must return no suggestions"
    return 0


def test_ac_c2_9_density_unchanged() -> int:
    """AC-C2-9: density of each suggestion ≤ density of input."""
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=True)
    for var in variations:
        assert density_unchanged(var, chords), "Density must not increase"
    return 0


def test_ac_c2_10_no_auto_execution() -> int:
    """AC-C2-10: Suggestions are not executed (played, applied) without explicit user action.
    This is a contract test: request_suggestions returns data only; caller must apply.
    """
    chords = _two_chords()
    result = _valid_result()
    variations = request_suggestions(chords, result, user_requested=True)
    assert variations  # We got data
    # No side effects: no global state mutation, no playback. Pure function.
    return 0


def main() -> int:
    r = 0
    r |= test_ac_c2_1_exactly_2_or_3_variations()
    r |= test_ac_c2_2_same_chord_count()
    r |= test_ac_c2_3_same_rhythmic_density()
    r |= test_ac_c2_4_same_tempo()
    r |= test_ac_c2_5_no_added_notes()
    r |= test_ac_c2_6_opt_in_only()
    r |= test_ac_c2_7_abstain_returns_empty()
    r |= test_ac_c2_7b_empty_chords_returns_empty()
    r |= test_ac_c2_9_density_unchanged()
    r |= test_ac_c2_10_no_auto_execution()
    if r == 0:
        print("performance_suggestions_test: PASS")
    return r


if __name__ == "__main__":
    sys.exit(main())
