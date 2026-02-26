"""
UI Gate invariant test: UI state must mirror core state.
Runs headless checks (no tkinter display) to verify logic.
"""

import os
import sys

# Add python/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minimal_listening_core import (
    AbstractInput,
    IntentAspect,
    intent_engine_process,
)


def test_abstain_display() -> int:
    """UI must show ABSTAIN when core abstains."""
    inp = AbstractInput(input_type="", aspect=IntentAspect.HARMONIC, magnitude=0.0)
    result = intent_engine_process(inp, None)
    assert result is not None
    assert result.get_metadata() == "abstain"
    return 0


def test_escalation_permissions_accurate() -> int:
    """UI must show escalation permissions accurately from core."""
    # chord_change + more_harmonic_color -> harmonic branches
    inp = AbstractInput(
        input_type="chord_change",
        aspect=IntentAspect.HARMONIC,
        magnitude=0.7,
        escalation_token="more_harmonic_color",
    )
    result = intent_engine_process(inp, None)
    assert result is not None
    h = result.get_harmonic_escalation()
    assert len(h) == 2

    # more_groove without rhythmic -> no rhythmic branches
    inp2 = AbstractInput(
        input_type="chord_change",
        aspect=IntentAspect.HARMONIC,
        magnitude=0.7,
        escalation_token="more_groove",
    )
    result2 = intent_engine_process(inp2, None)
    assert result2 is not None
    r = result2.get_rhythmic_escalation()
    assert len(r) == 0
    return 0


def test_escalation_buttons_domain_gated() -> int:
    """Escalation tokens only apply when intent matches domain."""
    # more_harmonic_color + rhythmic intent -> no harmonic branches (wrong domain)
    inp = AbstractInput(
        input_type="rhythmic",
        aspect=IntentAspect.RHYTHMIC,
        magnitude=0.5,
        escalation_token="more_harmonic_color",
    )
    result = intent_engine_process(inp, None)
    assert result is not None
    h = result.get_harmonic_escalation()
    assert len(h) == 0
    return 0


def main() -> int:
    try:
        test_abstain_display()
        test_escalation_permissions_accurate()
        test_escalation_buttons_domain_gated()
        print("UI Gate invariant tests: PASS")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
