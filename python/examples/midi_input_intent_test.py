"""
C1: MIDI Input → Intent Mapping test.
Empty MIDI → ABSTAIN. Notes present → intent with harmonic_tension PRESENT.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minimal_listening_core import (
    IntentAspect,
    ValueType,
    intent_engine_process,
    midi_notes_to_abstract_input,
)


def test_empty_midi_abstain() -> None:
    """Empty MIDI must produce ABSTAIN."""
    inp = midi_notes_to_abstract_input([])
    result = intent_engine_process(inp, None)
    assert result.get_metadata() == "abstain", "Empty MIDI must produce ABSTAIN"


def test_notes_present_harmonic() -> None:
    """Notes present must produce intent with harmonic_tension PRESENT."""
    notes = [(60, 80, 0.0), (64, 80, 0.0), (67, 80, 0.0)]
    inp = midi_notes_to_abstract_input(notes)
    result = intent_engine_process(inp, None)
    assert result.get_metadata() == "processed", "Notes must produce processed result"
    states = result.get_value_states()
    ht = states.get(ValueType.HARMONIC_TENSION)
    assert ht.is_present(), "harmonic_tension must be PRESENT"
    assert result.get_intent().has_aspect(IntentAspect.HARMONIC), "Intent must have HARMONIC aspect"


def main() -> int:
    try:
        test_empty_midi_abstain()
        test_notes_present_harmonic()
        print("midi input intent test: PASS")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
