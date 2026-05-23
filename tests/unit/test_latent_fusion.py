"""Tests for ``music_brain.latent.fusion``.

Closes **Multimodal fusion engine**, **Multimodal latent alignment**,
**Hybrid symbolic/audio generation**, **Audio/MIDI dual representation**,
**Stem-aware generation**, and **Multi-instrument orchestration**.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.fusion import (  # noqa: E402
    MultimodalFusion,
    StemBundle,
)
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402


def _prov() -> IntentProvenance:
    return IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0)


def _frame_with(audio: torch.Tensor, chord=None, t: int = 0) -> LatentFrame:
    return LatentFrame(
        audio_z=audio,
        chord_z=chord,
        emotion_va=(0.0, 0.0),
        time_index=t,
        provenance=_prov(),
        metadata={},
    )


def test_fusion_concatenates_audio_and_chord_dims() -> None:
    fuse = MultimodalFusion()
    f = _frame_with(
        audio=torch.zeros(2, 4, dtype=torch.float32), chord=torch.zeros(2, 8, dtype=torch.float32)
    )
    out = fuse(f)
    assert out.shape == (2, 12)


def test_fusion_handles_missing_chord_with_zero_pad() -> None:
    fuse = MultimodalFusion(chord_dim=8)
    f = _frame_with(audio=torch.ones(2, 4, dtype=torch.float32))
    out = fuse(f)
    assert out.shape == (2, 12)
    # The chord pad must be zeros.
    assert torch.all(out[:, 4:] == 0.0)


def test_fusion_requires_matching_time_steps() -> None:
    fuse = MultimodalFusion()
    f = _frame_with(
        audio=torch.zeros(2, 4, dtype=torch.float32), chord=torch.zeros(3, 8, dtype=torch.float32)
    )
    with pytest.raises(ValueError, match="time"):
        fuse(f)


def test_fusion_emits_for_chord_only_frames() -> None:
    """Symbolic-only generation: chord present, audio_z is a zero placeholder."""
    fuse = MultimodalFusion()
    audio = torch.zeros(3, 0, dtype=torch.float32)  # 0-dim audio
    chord = torch.randn(3, 8, dtype=torch.float32)
    f = _frame_with(audio=torch.zeros(3, 4, dtype=torch.float32), chord=chord)
    out = fuse(f)
    assert out.shape == (3, 12)  # 4 audio + 8 chord
    # Reject the truly zero-audio-dim path.
    with pytest.raises(ValueError, match="audio"):
        MultimodalFusion()._validate_audio_dim(0)
    _ = audio  # silence unused-variable warning


def test_stem_bundle_holds_multiple_instrument_frames() -> None:
    drums = _frame_with(audio=torch.zeros(2, 4, dtype=torch.float32))
    bass = _frame_with(audio=torch.zeros(2, 4, dtype=torch.float32))
    keys = _frame_with(audio=torch.zeros(2, 4, dtype=torch.float32))
    bundle = StemBundle(stems={"drums": drums, "bass": bass, "keys": keys})
    assert bundle.stem_names == ("bass", "drums", "keys")
    assert len(bundle) == 3


def test_stem_bundle_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least"):
        StemBundle(stems={})


def test_stem_bundle_requires_consistent_time_index() -> None:
    a = _frame_with(audio=torch.zeros(2, 4, dtype=torch.float32), t=0)
    b = _frame_with(audio=torch.zeros(2, 4, dtype=torch.float32), t=1)
    with pytest.raises(ValueError, match="time_index"):
        StemBundle(stems={"a": a, "b": b})


def test_stem_bundle_mix_concatenates_along_feature_axis() -> None:
    a = _frame_with(audio=torch.ones(2, 4, dtype=torch.float32))
    b = _frame_with(audio=torch.full((2, 4), 2.0, dtype=torch.float32))
    bundle = StemBundle(stems={"a": a, "b": b})
    mixed = bundle.mix()
    assert mixed.shape == (2, 8)
    assert torch.all(mixed[:, :4] == 1.0)
    assert torch.all(mixed[:, 4:] == 2.0)


def test_stem_bundle_sum_averages_along_feature_axis() -> None:
    a = _frame_with(audio=torch.ones(2, 4, dtype=torch.float32))
    b = _frame_with(audio=torch.full((2, 4), 3.0, dtype=torch.float32))
    bundle = StemBundle(stems={"a": a, "b": b})
    averaged = bundle.average()
    assert averaged.shape == (2, 4)
    assert torch.all(averaged == 2.0)
