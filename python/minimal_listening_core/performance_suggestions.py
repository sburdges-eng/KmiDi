"""
C2 — Performance Suggestions (2–3 only).

Opt-in only. No density increase. Voicing/inversion variants only.
Silence remains silence. Advisory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .intent_result import IntentResult


@dataclass
class ChordMaterial:
    """Minimal chord representation for suggestion input/output."""

    pitches: tuple[int, ...]
    start_beat: float
    duration: float

    def note_count(self) -> int:
        return len(self.pitches)

    def density(self) -> float:
        """Notes per beat. Used for AC-C2-3."""

        if self.duration <= 0:
            return 0.0
        return len(self.pitches) / self.duration


def _is_abstain(result: IntentResult) -> bool:
    return result.metadata == "abstain"


def _invert_chord(pitches: tuple[int, ...], inversion: int) -> tuple[int, ...]:
    """
    Apply inversion by rotating pitch classes. Same note count, no added notes.
    inversion 0 = root, 1 = first inv, 2 = second inv.
    """
    if not pitches or inversion == 0:
        return pitches
    n = len(pitches)
    inv = inversion % n
    if inv == 0:
        return pitches
    # Rotate: first `inv` notes move up an octave (voice-leading convention)
    sorted_p = sorted(pitches)
    result = [p + 12 if i < inv else p for i, p in enumerate(sorted_p)]
    return tuple(sorted(result))


def request_suggestions(
    chords: List[ChordMaterial],
    intent_result: IntentResult,
    user_requested: bool,
) -> List[List[ChordMaterial]]:
    """
    Return 2–3 MIDI variations. Opt-in only. No density increase.

    Args:
        chords: Current generated output (material to vary).
        intent_result: From C1 intent pipeline.
        user_requested: Must be True to produce suggestions.

    Returns:
        2–3 variations, or [] if: not user_requested, ABSTAIN, or empty chords.
    """
    if not user_requested:
        return []
    if _is_abstain(intent_result):
        return []
    if not chords:
        return []

    num_variations = min(3, max(2, len(chords)))
    variations: List[List[ChordMaterial]] = []
    for inv in range(num_variations):
        var = []
        for c in chords:
            new_pitches = _invert_chord(tuple(c.pitches), inv)
            var.append(
                ChordMaterial(
                    pitches=new_pitches,
                    start_beat=c.start_beat,
                    duration=c.duration,
                )
            )
        variations.append(var)

    return variations


def density_unchanged(variation: List[ChordMaterial], input_chords: List[ChordMaterial]) -> bool:
    """AC-C2-9: density of each suggestion ≤ density of input."""

    if len(variation) != len(input_chords):
        return False
    for vc, ic in zip(variation, input_chords):
        if vc.note_count() > ic.note_count():
            return False
        if vc.duration > 0 and ic.duration > 0:
            if vc.density() > ic.density():
                return False
    return True
