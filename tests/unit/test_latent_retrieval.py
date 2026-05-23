"""Tests for ``music_brain.latent.retrieval``.

Closes **Vector memory retrieval**, **Retrieval-augmented generation**,
**Emotion-conditioned retrieval**, **Embedding-indexed user history**,
**Semantic retrieval steering**, and **Context persistence beyond
token window** (via the persistent vector store under the hood).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402
from music_brain.latent.retrieval import (  # noqa: E402
    LatentMemory,
    pool_audio_z,
)


def _prov() -> IntentProvenance:
    return IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0)


def _frame(audio: torch.Tensor, *, time_index: int = 0, emotion_va=(0.0, 0.0)) -> LatentFrame:
    return LatentFrame(
        audio_z=audio,
        chord_z=None,
        emotion_va=emotion_va,
        time_index=time_index,
        provenance=_prov(),
        metadata={},
    )


def test_pool_audio_z_returns_mean_unit_vector() -> None:
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    pooled = pool_audio_z(z)
    assert pooled.shape == (2,)
    # Mean is (0.5, 0.5); pool_audio_z does NOT normalize — the
    # vector store handles normalization on insert.
    assert torch.allclose(pooled, torch.tensor([0.5, 0.5]))


def test_remember_and_recall_round_trip() -> None:
    mem = LatentMemory(dim=4)
    f1 = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    mem.remember("session-1", f1)
    hits = mem.recall(f1, top_k=1)
    assert len(hits) == 1
    assert hits[0].id == "session-1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_recall_orders_by_cosine_similarity() -> None:
    mem = LatentMemory(dim=4)
    mem.remember("a", _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
    mem.remember("b", _frame(torch.tensor([[0.0, 1.0, 0.0, 0.0]])))
    mem.remember("c", _frame(torch.tensor([[0.0, 0.0, 1.0, 0.0]])))
    q = _frame(torch.tensor([[1.0, 0.1, 0.0, 0.0]]))
    hits = mem.recall(q, top_k=2)
    assert [h.id for h in hits] == ["a", "b"]


def test_emotion_conditioned_recall_boosts_close_va_matches() -> None:
    """A retrieval that biases by emotion proximity should rank
    similar-VA frames above pure-cosine ones."""
    mem = LatentMemory(dim=4)
    # Two frames with identical embedding but different VA tags.
    same_z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mem.remember("happy", _frame(same_z, emotion_va=(0.9, 0.9)))
    mem.remember("sad", _frame(same_z, emotion_va=(-0.9, -0.9)))

    q_happy = _frame(same_z, emotion_va=(0.85, 0.85))
    hits = mem.recall(q_happy, top_k=2, emotion_weight=0.5)
    assert hits[0].id == "happy"  # emotion proximity broke the cosine tie


def test_remember_with_explicit_metadata() -> None:
    mem = LatentMemory(dim=4)
    f = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    mem.remember("evt", f, metadata={"user_id": "u42", "satisfaction": 0.85})
    hits = mem.recall(f, top_k=1)
    assert hits[0].metadata["user_id"] == "u42"
    assert hits[0].metadata["satisfaction"] == 0.85


def test_persistence_round_trip(tmp_path) -> None:
    mem = LatentMemory(dim=4)
    f = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    mem.remember("evt", f, metadata={"u": 1})
    out = tmp_path / "mem.npz"
    mem.save(out)
    restored = LatentMemory.load(out)
    hits = restored.recall(f, top_k=1)
    assert hits[0].id == "evt"
    assert hits[0].metadata["u"] == 1


def test_remember_rejects_dim_mismatch() -> None:
    mem = LatentMemory(dim=4)
    with pytest.raises(ValueError, match="dim"):
        mem.remember("x", _frame(torch.tensor([[1.0, 0.0, 0.0]])))  # 3-dim


def test_empty_memory_recall_returns_empty() -> None:
    mem = LatentMemory(dim=4)
    f = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert mem.recall(f, top_k=3) == []


def test_filter_by_user_id() -> None:
    mem = LatentMemory(dim=4)
    mem.remember("u1-a", _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]])), metadata={"user_id": "u1"})
    mem.remember("u2-a", _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]])), metadata={"user_id": "u2"})
    q = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    hits = mem.recall(q, top_k=2, user_id="u1")
    assert [h.id for h in hits] == ["u1-a"]


def test_emotion_weight_clamped_to_valid_range() -> None:
    mem = LatentMemory(dim=4)
    f = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    mem.remember("a", f)
    with pytest.raises(ValueError, match="emotion_weight"):
        mem.recall(f, top_k=1, emotion_weight=-0.1)
    with pytest.raises(ValueError, match="emotion_weight"):
        mem.recall(f, top_k=1, emotion_weight=1.5)


def test_user_scoped_recall_finds_user_hits_even_when_far_from_global_top_k() -> None:
    """Regression: a user with a few vectors must still get top_k hits
    when other users dominate the nearest neighbors. The old code
    fetched top_k*4 globally then post-filtered, so a "far" user got
    zero results. Reported by Codex on PR #196."""
    mem = LatentMemory(dim=4)
    # 20 vectors for user-popular, all close to (1, 0, 0, 0).
    for i in range(20):
        v = torch.tensor([[1.0, 0.001 * i, 0.0, 0.0]])
        mem.remember(f"pop-{i}", _frame(v), metadata={"user_id": "popular"})
    # 2 vectors for user-rare, far from query.
    mem.remember(
        "rare-a",
        _frame(torch.tensor([[0.0, 0.0, 1.0, 0.0]])),
        metadata={"user_id": "rare"},
    )
    mem.remember(
        "rare-b",
        _frame(torch.tensor([[0.0, 0.0, 0.9, 0.1]])),
        metadata={"user_id": "rare"},
    )
    # Query near (1, 0, 0, 0). user_id="rare" hits sit far below the
    # global top-k window — but user-scoped recall must still return them.
    q = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    hits = mem.recall(q, top_k=2, user_id="rare")
    assert sorted(h.id for h in hits) == ["rare-a", "rare-b"]
    for h in hits:
        assert h.metadata["user_id"] == "rare"


def test_user_scoped_recall_empty_for_unknown_user() -> None:
    mem = LatentMemory(dim=4)
    mem.remember(
        "u1-a",
        _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]])),
        metadata={"user_id": "u1"},
    )
    q = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert mem.recall(q, top_k=3, user_id="ghost") == []


def test_load_rebuilds_user_index_for_duplicate_embeddings(tmp_path) -> None:
    """Regression: two users with identical pooled embeddings must both
    appear in the rebuilt _user_index after load. The old top-1-search
    rebuild would collapse them onto whichever id won the tie-break.
    Reported by Cursor Bugbot on PR #196."""
    mem = LatentMemory(dim=4)
    z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mem.remember("alice-1", _frame(z), metadata={"user_id": "alice"})
    mem.remember("bob-1", _frame(z), metadata={"user_id": "bob"})
    out = tmp_path / "mem.npz"
    mem.save(out)
    restored = LatentMemory.load(out)
    q = _frame(z)
    alice_hits = restored.recall(q, top_k=2, user_id="alice")
    bob_hits = restored.recall(q, top_k=2, user_id="bob")
    assert [h.id for h in alice_hits] == ["alice-1"]
    assert [h.id for h in bob_hits] == ["bob-1"]


def test_pool_audio_z_rejects_non_2d_tensor() -> None:
    with pytest.raises(ValueError, match="2-D"):
        pool_audio_z(torch.zeros(4))


def test_numpy_round_trip_through_memory() -> None:
    """Sanity: the store's underlying numpy dtype stays float32."""
    mem = LatentMemory(dim=4)
    mem.remember("a", _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
    arr = mem.snapshot_vectors()
    assert arr.dtype == np.float32
    assert arr.shape == (1, 4)
