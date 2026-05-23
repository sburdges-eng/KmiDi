"""Tests for ``music_brain.latent.ump`` — minimal MIDI 2.0 UMP validator.

Closes **UMP mapping validation** and the **Schema expansion and
validation** spec items for the MIDI 2.0 wire format.
"""

from __future__ import annotations

import pytest

from music_brain.latent.ump import (  # noqa: E402
    UMP_MESSAGE_TYPE_MIDI_1,
    UMP_MESSAGE_TYPE_MIDI_2,
    UMP_MESSAGE_TYPE_UTILITY,
    UMPMessage,
    UMPMessageType,
    pack_midi1_note_on,
    pack_midi2_note_on,
    validate_ump,
)


def test_message_type_constants_match_spec() -> None:
    """Per MMA UMP spec: utility=0x0, system=0x1, midi1=0x2, midi2=0x4."""
    assert UMP_MESSAGE_TYPE_UTILITY == 0x0
    assert UMP_MESSAGE_TYPE_MIDI_1 == 0x2
    assert UMP_MESSAGE_TYPE_MIDI_2 == 0x4


def test_pack_midi1_note_on_layout() -> None:
    word = pack_midi1_note_on(group=0, channel=0, note=60, velocity=100)
    # MT=0x2 in the top nibble.
    assert (word >> 28) & 0xF == 0x2
    # Status byte 0x9 (note-on) with channel 0.
    status = (word >> 16) & 0xFF
    assert status == 0x90
    # Note number.
    assert (word >> 8) & 0xFF == 60
    # Velocity.
    assert word & 0xFF == 100


def test_pack_midi1_validates_ranges() -> None:
    with pytest.raises(ValueError, match="channel"):
        pack_midi1_note_on(group=0, channel=16, note=60, velocity=100)
    with pytest.raises(ValueError, match="note"):
        pack_midi1_note_on(group=0, channel=0, note=128, velocity=100)
    with pytest.raises(ValueError, match="velocity"):
        pack_midi1_note_on(group=0, channel=0, note=60, velocity=200)
    with pytest.raises(ValueError, match="group"):
        pack_midi1_note_on(group=16, channel=0, note=60, velocity=100)


def test_pack_midi2_note_on_returns_two_words() -> None:
    w1, w2 = pack_midi2_note_on(group=0, channel=0, note=60, velocity_16=0x4000, attribute=0)
    # MT=0x4 in top nibble.
    assert (w1 >> 28) & 0xF == 0x4
    # Word 2: high 16 bits = velocity_16, low 16 bits = attribute.
    assert (w2 >> 16) & 0xFFFF == 0x4000
    assert w2 & 0xFFFF == 0


def test_pack_midi2_validates_16bit_velocity() -> None:
    with pytest.raises(ValueError, match="velocity_16"):
        pack_midi2_note_on(group=0, channel=0, note=60, velocity_16=0x10000, attribute=0)


def test_validate_ump_accepts_known_message_types() -> None:
    word = pack_midi1_note_on(group=0, channel=0, note=60, velocity=100)
    msg = validate_ump([word])
    assert msg.message_type == UMPMessageType.MIDI_1
    assert msg.group == 0
    assert msg.word_count == 1


def test_validate_ump_two_word_midi2_message() -> None:
    w1, w2 = pack_midi2_note_on(group=0, channel=0, note=60, velocity_16=0x4000, attribute=0)
    msg = validate_ump([w1, w2])
    assert msg.message_type == UMPMessageType.MIDI_2
    assert msg.word_count == 2


def test_validate_ump_rejects_truncated_midi2_message() -> None:
    w1, _ = pack_midi2_note_on(group=0, channel=0, note=60, velocity_16=0x4000, attribute=0)
    with pytest.raises(ValueError, match="word count"):
        validate_ump([w1])


def test_validate_ump_rejects_unknown_message_type() -> None:
    # mt = 0xF is reserved.
    with pytest.raises(ValueError, match="message type"):
        validate_ump([0xF0000000])


def test_ump_message_round_trip_metadata() -> None:
    word = pack_midi1_note_on(group=3, channel=2, note=72, velocity=64)
    msg = validate_ump([word])
    assert isinstance(msg, UMPMessage)
    assert msg.group == 3
    assert msg.channel == 2
    assert msg.note == 72
    assert msg.velocity == 64
