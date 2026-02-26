"""Invariant test: more_groove without rhythmic intent -> NONE. With rhythmic -> 3 branches."""

import sys

from minimal_listening_core import (
    AbstractInput,
    IntentAspect,
    ValueType,
    intent_engine_process,
)


def test_more_groove_without_rhythmic_intent() -> int:
    input_obj = AbstractInput(
        input_type="chord_change",
        aspect=IntentAspect.HARMONIC,
        magnitude=0.7,
        escalation_token="more_groove",
    )
    result = intent_engine_process(input_obj, None)
    esc = result.get_rhythmic_escalation()
    if len(esc) != 0:
        print(f"FAIL: more_groove without rhythmic intent should yield 0 branches, got {len(esc)}")
        return 1
    return 0


def test_more_groove_with_rhythmic_intent() -> int:
    input_obj = AbstractInput(
        input_type="rhythmic",
        aspect=IntentAspect.RHYTHMIC,
        magnitude=0.5,
        escalation_token="more_groove",
    )
    result = intent_engine_process(input_obj, None)
    esc = result.get_rhythmic_escalation()
    if len(esc) != 3:
        print(f"FAIL: more_groove with rhythmic intent should yield 3 branches, got {len(esc)}")
        return 1
    if not esc.satisfies_invariant():
        print("FAIL: rhythmic escalation invariant violated")
        return 1
    return 0


def main() -> int:
    r = test_more_groove_without_rhythmic_intent()
    r |= test_more_groove_with_rhythmic_intent()
    if r == 0:
        print("rhythmic escalation invariant tests: PASS")
    return r


if __name__ == "__main__":
    sys.exit(main())
