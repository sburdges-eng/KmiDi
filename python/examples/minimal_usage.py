"""Minimal usage: chord_change -> harmonic change, no added density."""

import sys

from minimal_listening_core import (
    AbstractInput,
    IntentAspect,
    ValueType,
    intent_engine_process,
    ConstraintSet,
)


def main() -> int:
    input_obj = AbstractInput(
        input_type="chord_change",
        aspect=IntentAspect.HARMONIC,
        magnitude=0.7,
    )

    constraints = ConstraintSet()
    result = intent_engine_process(input_obj, constraints)

    if result is None:
        print("invariant violation: result is None", file=sys.stderr)
        return 1

    states = result.get_value_states()
    harmonic = states.get(ValueType.HARMONIC_TENSION)
    density = states.get(ValueType.RHYTHMIC_DENSITY)

    print("chord_change input -> harmonic_tension: ", end="")
    if harmonic.is_present():
        print(f"PRESENT({harmonic.get_value():.2f})")
    else:
        print("MISSING/INVALID")

    print("chord_change input -> rhythmic_density: ", end="")
    if density.is_missing():
        print("MISSING(no_intent) - preserved absence, no added density")
    elif density.is_present():
        print("PRESENT - ERROR: should not add density")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
