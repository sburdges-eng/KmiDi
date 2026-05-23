"""Tests for ``music_brain.latent.motif`` — motif persistence + recurrence.

Closes **Semantic motif persistence**, **Motif recurrence tracking**,
and **Arrangement continuity preservation** (via motif callbacks).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402
from music_brain.latent.motif import (  # noqa: E402
    MotifTracker,
    MotifRecurrence,
)


def _frame(audio, *, t: int = 0) -> LatentFrame:
    return LatentFrame(
        audio_z=audio, chord_z=None, emotion_va=(0.0, 0.0),
        time_index=t,
        provenance=IntentProvenance(source=IntentSource.UI_DIRECT,
                                    user_override_weight=1.0),
        metadata={},
    )


def test_track_creates_new_motif_on_first_observation() -> None:
    mt = MotifTracker(similarity_threshold=0.95)
    z = torch.tensor([[1.0, 0.0, 0.0]])
    rec = mt.observe(_frame(z, t=0))
    assert isinstance(rec, MotifRecurrence)
    assert rec.is_new
    assert rec.occurrences == 1


def test_repeat_observation_increments_occurrence_count() -> None:
    mt = MotifTracker(similarity_threshold=0.95)
    z = torch.tensor([[1.0, 0.0, 0.0]])
    mt.observe(_frame(z, t=0))
    rec = mt.observe(_frame(z, t=4))
    assert not rec.is_new
    assert rec.occurrences == 2
    assert rec.time_indices == (0, 4)


def test_dissimilar_motif_gets_new_id() -> None:
    mt = MotifTracker(similarity_threshold=0.99)
    z1 = torch.tensor([[1.0, 0.0, 0.0]])
    z2 = torch.tensor([[0.0, 1.0, 0.0]])
    a = mt.observe(_frame(z1, t=0))
    b = mt.observe(_frame(z2, t=4))
    assert a.motif_id != b.motif_id
    assert b.is_new


def test_motif_count_grows_with_unique_observations() -> None:
    mt = MotifTracker(similarity_threshold=0.99)
    for i in range(3):
        z = torch.zeros(1, 3)
        z[0, i] = 1.0
        mt.observe(_frame(z, t=i))
    assert mt.motif_count == 3


def test_recurring_motif_does_not_create_new_entry() -> None:
    mt = MotifTracker(similarity_threshold=0.95)
    z = torch.tensor([[1.0, 0.0, 0.0]])
    mt.observe(_frame(z, t=0))
    mt.observe(_frame(z, t=4))
    mt.observe(_frame(z, t=8))
    assert mt.motif_count == 1
    rec = mt.most_recurrent()
    assert rec.occurrences == 3
    assert rec.time_indices == (0, 4, 8)


def test_observe_rejects_dim_mismatch_after_first() -> None:
    mt = MotifTracker(similarity_threshold=0.95)
    mt.observe(_frame(torch.tensor([[1.0, 0.0, 0.0]]), t=0))
    with pytest.raises(ValueError, match="dim"):
        mt.observe(_frame(torch.tensor([[1.0, 0.0]]), t=4))


def test_similarity_threshold_validated() -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        MotifTracker(similarity_threshold=-0.1)
    with pytest.raises(ValueError, match="similarity_threshold"):
        MotifTracker(similarity_threshold=1.5)


def test_most_recurrent_picks_highest_occurrence_count() -> None:
    mt = MotifTracker(similarity_threshold=0.99)
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    mt.observe(_frame(a, t=0))
    mt.observe(_frame(b, t=4))
    mt.observe(_frame(b, t=8))
    rec = mt.most_recurrent()
    assert rec.occurrences == 2
    # Vectors equal to b's representative.
    assert torch.allclose(rec.representative, b.squeeze(0), atol=1e-6)


def test_most_recurrent_on_empty_returns_none() -> None:
    mt = MotifTracker()
    assert mt.most_recurrent() is None


def test_zero_vector_does_not_crash() -> None:
    mt = MotifTracker(similarity_threshold=0.95)
    z = torch.zeros(1, 3)
    rec = mt.observe(_frame(z, t=0))
    # Treated as a new motif but cosine sim is undefined → it must not
    # produce NaN scores.
    assert rec.is_new
