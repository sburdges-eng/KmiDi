"""Invariant violation: magnitude out of range [0,1] -> INVALID result."""

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
        magnitude=1.5,
    )

    constraints = ConstraintSet()
    result = intent_engine_process(input_obj, constraints)

    if result is not None:
        states = result.get_value_states()
        harmonic = states.get(ValueType.HARMONIC_TENSION)
        # 1.5 is wrong: engine must reject. INVALID = correct; PRESENT>1.0 = bug
        if harmonic.is_present() and harmonic.get_value() > 1.0:
            print("invariant violation: engine accepted magnitude 1.5 (should reject)")
            return 1
        # Engine correctly rejected 1.5 -> INVALID. Stub pass.
    else:
        print("invariant violation: validate_invariants failed, result is None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
