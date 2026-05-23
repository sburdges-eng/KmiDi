"""Tests for ``music_brain.latent.rollback``.

Closes **Realtime rollback system** and **Realtime audition-safe
regeneration**. The rollback ring lets a realtime caller checkpoint
generation state before a speculative path and revert in O(1) if
the user audits it and rejects.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.kv_cache import KVCache  # noqa: E402
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402
from music_brain.latent.rollback import RollbackRing  # noqa: E402


def _prov() -> IntentProvenance:
    return IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0)


def _frame(t: int) -> LatentFrame:
    return LatentFrame(
        audio_z=torch.zeros(1, 4, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.0, 0.0),
        time_index=t,
        provenance=_prov(),
        metadata={"t": t},
    )


def test_starts_empty() -> None:
    r = RollbackRing(capacity=4)
    assert len(r) == 0
    assert r.head_index is None


def test_checkpoint_increments_length() -> None:
    r = RollbackRing(capacity=4)
    r.checkpoint(_frame(0))
    r.checkpoint(_frame(1))
    assert len(r) == 2
    assert r.head_index == 1


def test_capacity_evicts_oldest() -> None:
    r = RollbackRing(capacity=3)
    for t in range(5):
        r.checkpoint(_frame(t))
    assert len(r) == 3
    # Oldest two (t=0, t=1) were evicted.
    indices = [c.time_index for c in r.snapshot()]
    assert indices == [2, 3, 4]


def test_rollback_returns_frame_and_shrinks() -> None:
    r = RollbackRing(capacity=4)
    f0, f1, f2 = _frame(0), _frame(1), _frame(2)
    r.checkpoint(f0)
    r.checkpoint(f1)
    r.checkpoint(f2)
    out = r.rollback_to(1)
    assert out.time_index == 1
    # The "1" checkpoint stays as the new head; "2" is dropped.
    assert len(r) == 2
    assert r.head_index == 1


def test_rollback_to_zero_resets_ring() -> None:
    r = RollbackRing(capacity=4)
    r.checkpoint(_frame(0))
    r.checkpoint(_frame(1))
    r.rollback_to(0)
    assert len(r) == 1
    assert r.head_index == 0


def test_rollback_unknown_index_raises() -> None:
    r = RollbackRing(capacity=4)
    r.checkpoint(_frame(0))
    with pytest.raises(KeyError, match="checkpoint"):
        r.rollback_to(42)


def test_attaches_kv_cache_and_truncates_on_rollback() -> None:
    r = RollbackRing(capacity=4)
    cache = KVCache(num_heads=1, head_dim=4, max_len=16)
    cache.append(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4))
    r.checkpoint(_frame(0), kv_cache=cache)
    cache.append(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    r.checkpoint(_frame(1), kv_cache=cache)
    cache.append(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4))
    r.checkpoint(_frame(2), kv_cache=cache)
    assert cache.length == 9

    r.rollback_to(0)
    # KV cache should be truncated back to the length at checkpoint 0.
    assert cache.length == 3


def test_rejects_non_monotonic_checkpoint() -> None:
    r = RollbackRing(capacity=4)
    r.checkpoint(_frame(3))
    with pytest.raises(ValueError, match="monotonic"):
        r.checkpoint(_frame(2))


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        RollbackRing(capacity=0)


def test_snapshot_returns_copy() -> None:
    r = RollbackRing(capacity=4)
    r.checkpoint(_frame(0))
    snap = r.snapshot()
    snap.append(_frame(99))  # type: ignore[arg-type]
    # Ring's own length must not have changed.
    assert len(r) == 1
