"""
B2: Playback guardrails test. CI must detect forbidden playback.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minimal_listening_core import AbstractInput, IntentAspect, intent_engine_process
from ui_gate.playback_guardrails import assert_playback_allowed, can_produce_output


def test_can_produce_output_abstain() -> int:
    inp = AbstractInput(input_type="", aspect=IntentAspect.HARMONIC, magnitude=0.0)
    result = intent_engine_process(inp, None)
    assert not can_produce_output(result), "ABSTAIN must not allow output"
    return 0


def test_can_produce_output_valid() -> int:
    inp = AbstractInput(input_type="chord_change", aspect=IntentAspect.HARMONIC, magnitude=0.7)
    result = intent_engine_process(inp, None)
    assert can_produce_output(result), "Valid result must allow output"
    return 0


def test_assert_playback_forbidden_without_user() -> int:
    """Debug build must assert when user_initiated=False."""
    inp = AbstractInput(input_type="chord_change", aspect=IntentAspect.HARMONIC, magnitude=0.7)
    result = intent_engine_process(inp, None)
    try:
        assert_playback_allowed(user_initiated=False, result=result, caller="test")
        if __debug__:
            return 1  # Assert should have fired
    except AssertionError:
        pass  # Expected
    return 0


def test_assert_playback_forbidden_after_abstain() -> int:
    """Debug build must assert when result is ABSTAIN."""
    inp = AbstractInput(input_type="", aspect=IntentAspect.HARMONIC, magnitude=0.0)
    result = intent_engine_process(inp, None)
    try:
        assert_playback_allowed(user_initiated=True, result=result, caller="test")
        if __debug__:
            return 1  # Assert should have fired
    except AssertionError:
        pass  # Expected
    return 0


def main() -> int:
    r = 0
    r |= test_can_produce_output_abstain()
    r |= test_can_produce_output_valid()
    r |= test_assert_playback_forbidden_without_user()
    r |= test_assert_playback_forbidden_after_abstain()
    if r == 0:
        print("playback guardrails test: PASS")
    return r


if __name__ == "__main__":
    sys.exit(main())
