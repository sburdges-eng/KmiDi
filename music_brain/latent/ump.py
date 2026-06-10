"""Minimal MIDI 2.0 UMP (Universal MIDI Packet) packer + validator.

Closes **UMP mapping validation** and the MIDI-2.0 side of the
**Schema expansion and validation** spec item. The platform watchlist
(``docs/research/KMIDI_PLATFORM_WATCHLIST_2026.md``) calls out MIDI 2.0
as a "research-grade" target — this module ships the minimum on-wire
contract so downstream code can emit/consume UMP without a third-party
crate.

Scope:
  - Pack a MIDI 1.0 note-on as the 1-word UMP MT=2 message.
  - Pack a MIDI 2.0 note-on as the 2-word UMP MT=4 message
    (16-bit velocity + 16-bit attribute).
  - ``validate_ump([word, …])`` decodes the top nibble (message type)
    and returns a structured ``UMPMessage`` while enforcing the spec's
    word-count-per-MT rule (MT=2 → 1 word, MT=4 → 2 words).

This is intentionally tiny — full MIDI 2.0 (Property Exchange,
Per-Note Articulation, Profiles) is out of scope for a *validator*. We
ship the wire-level contract; profile management lives elsewhere.

Spec references:
  - MMA UMP spec (Universal MIDI Packet Format), section "Message Type".
  - MIDI Association: "State of MIDI 2.0 updates (2025–2026)".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

UMP_MESSAGE_TYPE_UTILITY = 0x0
UMP_MESSAGE_TYPE_SYSTEM = 0x1
UMP_MESSAGE_TYPE_MIDI_1 = 0x2
UMP_MESSAGE_TYPE_DATA_64 = 0x3
UMP_MESSAGE_TYPE_MIDI_2 = 0x4
UMP_MESSAGE_TYPE_DATA_128 = 0x5


_MT_TO_WORD_COUNT = {
    UMP_MESSAGE_TYPE_UTILITY: 1,
    UMP_MESSAGE_TYPE_SYSTEM: 1,
    UMP_MESSAGE_TYPE_MIDI_1: 1,
    UMP_MESSAGE_TYPE_DATA_64: 2,
    UMP_MESSAGE_TYPE_MIDI_2: 2,
    UMP_MESSAGE_TYPE_DATA_128: 4,
}


class UMPMessageType(IntEnum):
    UTILITY = UMP_MESSAGE_TYPE_UTILITY
    SYSTEM = UMP_MESSAGE_TYPE_SYSTEM
    MIDI_1 = UMP_MESSAGE_TYPE_MIDI_1
    DATA_64 = UMP_MESSAGE_TYPE_DATA_64
    MIDI_2 = UMP_MESSAGE_TYPE_MIDI_2
    DATA_128 = UMP_MESSAGE_TYPE_DATA_128


@dataclass(frozen=True)
class UMPMessage:
    message_type: UMPMessageType
    group: int
    word_count: int
    # MIDI 1/2 note-on convenience fields (None for non-note messages).
    channel: int = -1
    note: int = -1
    velocity: int = -1


def _check_range(name: str, v: int, lo: int, hi: int) -> None:
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {v}")


# ----------------------------------------------------------------------
# Packers
# ----------------------------------------------------------------------


def pack_midi1_note_on(*, group: int, channel: int, note: int, velocity: int) -> int:
    """Pack a MIDI 1.0 note-on into one 32-bit UMP word (MT=2)."""
    _check_range("group", group, 0, 15)
    _check_range("channel", channel, 0, 15)
    _check_range("note", note, 0, 127)
    _check_range("velocity", velocity, 0, 127)
    mt = UMP_MESSAGE_TYPE_MIDI_1
    status = 0x90 | (channel & 0xF)  # note-on opcode
    return (
        ((mt & 0xF) << 28)
        | ((group & 0xF) << 24)
        | (status << 16)
        | ((note & 0x7F) << 8)
        | (velocity & 0x7F)
    )


def pack_midi2_note_on(
    *, group: int, channel: int, note: int, velocity_16: int, attribute: int
) -> tuple[int, int]:
    """Pack a MIDI 2.0 note-on into two 32-bit UMP words (MT=4)."""
    _check_range("group", group, 0, 15)
    _check_range("channel", channel, 0, 15)
    _check_range("note", note, 0, 127)
    _check_range("velocity_16", velocity_16, 0, 0xFFFF)
    _check_range("attribute", attribute, 0, 0xFFFF)
    mt = UMP_MESSAGE_TYPE_MIDI_2
    status = 0x90 | (channel & 0xF)
    word1 = ((mt & 0xF) << 28) | ((group & 0xF) << 24) | (status << 16) | ((note & 0x7F) << 8)
    word2 = ((velocity_16 & 0xFFFF) << 16) | (attribute & 0xFFFF)
    return word1, word2


# ----------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------


def validate_ump(words: list[int]) -> UMPMessage:
    """Validate a sequence of UMP words and decode the first message."""
    if not words:
        raise ValueError("UMP word sequence is empty")
    head = words[0]
    mt = (head >> 28) & 0xF
    if mt not in _MT_TO_WORD_COUNT:
        raise ValueError(f"unknown UMP message type 0x{mt:X}")
    expected_words = _MT_TO_WORD_COUNT[mt]
    if len(words) < expected_words:
        raise ValueError(
            f"UMP word count for MT=0x{mt:X} must be {expected_words}, " f"got {len(words)}"
        )
    group = (head >> 24) & 0xF
    channel = -1
    note = -1
    velocity = -1
    if mt == UMP_MESSAGE_TYPE_MIDI_1:
        status = (head >> 16) & 0xFF
        channel = status & 0xF
        note = (head >> 8) & 0xFF
        velocity = head & 0xFF
    elif mt == UMP_MESSAGE_TYPE_MIDI_2 and len(words) >= 2:
        status = (head >> 16) & 0xFF
        channel = status & 0xF
        note = (head >> 8) & 0x7F
        velocity = (words[1] >> 16) & 0xFFFF
    return UMPMessage(
        message_type=UMPMessageType(mt),
        group=group,
        word_count=expected_words,
        channel=channel,
        note=note,
        velocity=velocity,
    )


__all__ = [
    "UMP_MESSAGE_TYPE_UTILITY",
    "UMP_MESSAGE_TYPE_SYSTEM",
    "UMP_MESSAGE_TYPE_MIDI_1",
    "UMP_MESSAGE_TYPE_DATA_64",
    "UMP_MESSAGE_TYPE_MIDI_2",
    "UMP_MESSAGE_TYPE_DATA_128",
    "UMPMessage",
    "UMPMessageType",
    "pack_midi1_note_on",
    "pack_midi2_note_on",
    "validate_ump",
]
