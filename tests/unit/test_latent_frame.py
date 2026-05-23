"""Tests for ``music_brain.latent.latent_frame.LatentFrame``.

Closes the **Shared latent contract** spec item identified as the
highest-leverage gap during the 2026-05-22 latent-core inventory: every
encoder (audio_jepa, chord_jepa, emotion_probe) emitted its own tensor
without a shared normalization / time-indexing / provenance envelope.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402


def _frame(**overrides):
    defaults = dict(
        audio_z=torch.zeros(4, 8, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.0, 0.0),
        time_index=0,
        provenance=IntentProvenance(source=IntentSource.ML_AUDIO,
                                    user_override_weight=0.0),
        metadata={"chunk_ms": 32},
    )
    defaults.update(overrides)
    return LatentFrame(**defaults)


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------

def test_construction_minimal_smoke() -> None:
    frame = _frame()
    assert frame.audio_z.shape == (4, 8)
    assert frame.time_index == 0
    assert frame.chord_z is None
    assert frame.emotion_va == (0.0, 0.0)


def test_audio_z_must_be_2d() -> None:
    with pytest.raises(ValueError, match="audio_z"):
        _frame(audio_z=torch.zeros(8, dtype=torch.float32))


def test_audio_z_must_be_float32() -> None:
    with pytest.raises(ValueError, match="dtype"):
        _frame(audio_z=torch.zeros(4, 8, dtype=torch.float64))


def test_chord_z_when_present_must_be_2d() -> None:
    with pytest.raises(ValueError, match="chord_z"):
        _frame(chord_z=torch.zeros(8, dtype=torch.float32))


def test_emotion_va_in_range() -> None:
    with pytest.raises(ValueError, match="emotion_va"):
        _frame(emotion_va=(1.5, 0.0))
    with pytest.raises(ValueError, match="emotion_va"):
        _frame(emotion_va=(0.0, -1.1))


def test_time_index_non_negative() -> None:
    with pytest.raises(ValueError, match="time_index"):
        _frame(time_index=-1)


def test_metadata_is_immutable_view() -> None:
    md = {"chunk_ms": 32}
    frame = _frame(metadata=md)
    # Mutating the source dict must not mutate the frame's view.
    md["chunk_ms"] = 9999
    assert frame.metadata["chunk_ms"] == 32
    # And the exposed view rejects direct mutation.
    with pytest.raises(TypeError):
        frame.metadata["new_key"] = 1  # type: ignore[index]


# ----------------------------------------------------------------------
# Immutability
# ----------------------------------------------------------------------

def test_frame_is_frozen() -> None:
    frame = _frame()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        frame.time_index = 5  # type: ignore[misc]


# ----------------------------------------------------------------------
# Serialization round-trip
# ----------------------------------------------------------------------

def test_dict_round_trip_audio_only() -> None:
    torch.manual_seed(0)
    src = _frame(audio_z=torch.randn(3, 4, dtype=torch.float32),
                 emotion_va=(0.5, -0.25),
                 time_index=7,
                 metadata={"chunk_ms": 32, "engine": "audio_jepa"})
    payload = src.to_dict()
    restored = LatentFrame.from_dict(payload)
    assert torch.equal(src.audio_z, restored.audio_z)
    assert restored.chord_z is None
    assert restored.emotion_va == (0.5, -0.25)
    assert restored.time_index == 7
    assert restored.provenance.source == src.provenance.source
    assert restored.metadata["engine"] == "audio_jepa"


def test_dict_round_trip_with_chord() -> None:
    torch.manual_seed(0)
    src = _frame(audio_z=torch.randn(2, 8, dtype=torch.float32),
                 chord_z=torch.randn(5, 16, dtype=torch.float32))
    restored = LatentFrame.from_dict(src.to_dict())
    assert torch.equal(src.audio_z, restored.audio_z)
    assert torch.equal(src.chord_z, restored.chord_z)


# ----------------------------------------------------------------------
# Time-index helpers
# ----------------------------------------------------------------------

def test_advance_returns_new_frame_with_incremented_index() -> None:
    src = _frame(time_index=3)
    later = src.advance(time_index=4, audio_z=torch.zeros(4, 8, dtype=torch.float32))
    assert src.time_index == 3  # original unchanged
    assert later.time_index == 4
    assert later.provenance is src.provenance  # propagated


def test_advance_rejects_non_monotonic_step() -> None:
    src = _frame(time_index=3)
    with pytest.raises(ValueError, match="monotonic"):
        src.advance(time_index=2, audio_z=torch.zeros(4, 8, dtype=torch.float32))


def test_advance_rejects_dim_mismatch() -> None:
    src = _frame(audio_z=torch.zeros(4, 8, dtype=torch.float32))
    with pytest.raises(ValueError, match="feature dim"):
        src.advance(time_index=4,
                    audio_z=torch.zeros(4, 16, dtype=torch.float32))
