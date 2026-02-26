"""
B2: Playback Guardrails & Assertions.

Ensures playback can never bypass listening rules.
- Playback without user action → fail
- Playback after ABSTAIN → fail
- Playback path must check constraints (canProduceOutput)
"""

from __future__ import annotations


def can_produce_output(result) -> bool:
    """Playback must pass this before producing output."""
    if result is None:
        return False
    if result.get_metadata() == "abstain":
        return False
    return result.is_valid()


def assert_playback_allowed(
    *,
    user_initiated: bool,
    result,
    caller: str = "",
) -> None:
    """
    B2: Assert playback is allowed. Call before any playback.
    Fails in debug if:
    - playback without user action
    - playback after ABSTAIN
    - canProduceOutput is False
    """
    if not __debug__:
        return
    assert user_initiated, f"Playback without user action forbidden: {caller}"
    assert result is not None, f"Playback with None result forbidden: {caller}"
    assert result.get_metadata() != "abstain", f"Playback after ABSTAIN forbidden: {caller}"
    assert result.is_valid(), f"Playback when invalid forbidden: {caller}"
