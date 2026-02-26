"""
C1 — MIDI Input → Intent Mapping.

Infer minimal intent from MIDI notes. No escalation, no playback.
Performance ≠ permission; silence remains silence.
"""

from __future__ import annotations

from typing import List, Tuple

from .intent_engine import AbstractInput
from .musical_intent import IntentAspect


def midi_notes_to_abstract_input(
    notes: List[Tuple[int, int, float]],
) -> AbstractInput:
    """
    Map MIDI notes to AbstractInput. Minimal, symbolic.
    Empty notes → ABSTAIN (input_type="").
    Notes present → chord_change, HARMONIC, magnitude 0.7 (fixed).
    """
    if not notes:
        return AbstractInput(
            input_type="",
            aspect=IntentAspect.HARMONIC,
            magnitude=0.0,
            escalation_token=None,
        )
    return AbstractInput(
        input_type="chord_change",
        aspect=IntentAspect.HARMONIC,
        magnitude=0.7,
        escalation_token=None,
    )
