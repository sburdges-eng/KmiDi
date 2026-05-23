"""Tests for ``music_brain.latent.kv_cache.KVCache``.

Closes the **KV-cache optimization** spec item. The cache is the
substrate that makes incremental decoding (the next test file)
cheaper than O(T^2): once a (k, v) tensor has been computed for an
intent slot or an emitted audio frame, it never recomputes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.latent.kv_cache import KVCache  # noqa: E402


def _cache(num_heads: int = 2, head_dim: int = 4, max_len: int = 16) -> KVCache:
    return KVCache(num_heads=num_heads, head_dim=head_dim, max_len=max_len)


def test_starts_empty() -> None:
    c = _cache()
    assert c.length == 0
    assert c.capacity == 16


def test_append_grows_length() -> None:
    c = _cache()
    k = torch.zeros(1, 2, 3, 4)
    v = torch.zeros(1, 2, 3, 4)
    c.append(k, v)
    assert c.length == 3
    c.append(k, v)
    assert c.length == 6


def test_append_returns_full_keys_values() -> None:
    c = _cache(num_heads=2, head_dim=4, max_len=8)
    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    k_all, v_all = c.append(k1, v1)
    assert k_all.shape == (1, 2, 3, 4)
    assert torch.equal(k_all, k1)

    k2 = torch.randn(1, 2, 2, 4)
    v2 = torch.randn(1, 2, 2, 4)
    k_all2, v_all2 = c.append(k2, v2)
    assert k_all2.shape == (1, 2, 5, 4)
    assert torch.equal(k_all2[:, :, :3, :], k1)
    assert torch.equal(k_all2[:, :, 3:, :], k2)


def test_capacity_overflow_raises() -> None:
    c = _cache(max_len=4)
    k = torch.zeros(1, 2, 5, 4)
    v = torch.zeros(1, 2, 5, 4)
    with pytest.raises(ValueError, match="capacity"):
        c.append(k, v)


def test_truncate_shrinks_length_without_reallocation() -> None:
    c = _cache()
    c.append(torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    assert c.length == 5
    c.truncate(3)
    assert c.length == 3
    k_all, _ = c.append(torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
    assert k_all.shape == (1, 2, 5, 4)


def test_truncate_to_zero_resets() -> None:
    c = _cache()
    c.append(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
    c.truncate(0)
    assert c.length == 0


def test_truncate_negative_or_above_length_raises() -> None:
    c = _cache()
    c.append(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
    with pytest.raises(ValueError, match="truncate"):
        c.truncate(-1)
    with pytest.raises(ValueError, match="truncate"):
        c.truncate(10)


def test_shape_mismatch_rejected_after_first_append() -> None:
    c = _cache(num_heads=2, head_dim=4)
    c.append(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
    # Wrong num_heads.
    with pytest.raises(ValueError, match="num_heads"):
        c.append(torch.zeros(1, 3, 1, 4), torch.zeros(1, 3, 1, 4))
    # Wrong head_dim.
    with pytest.raises(ValueError, match="head_dim"):
        c.append(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8))


def test_k_and_v_shape_must_agree() -> None:
    c = _cache()
    with pytest.raises(ValueError, match="shape mismatch"):
        c.append(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 5, 4))


def test_snapshot_returns_a_copy_not_a_view() -> None:
    c = _cache()
    k = torch.randn(1, 2, 2, 4)
    v = torch.randn(1, 2, 2, 4)
    c.append(k, v)
    k_snap, _ = c.snapshot()
    k_snap.zero_()
    # The cache's internal buffer must not have been touched.
    k_again, _ = c.snapshot()
    assert not torch.allclose(k_again, torch.zeros_like(k_again))


def test_reuse_saves_recomputation() -> None:
    """The whole point of a KV-cache: once appended, never recompute."""
    c = _cache(max_len=32)
    k_full = torch.randn(1, 2, 10, 4)
    v_full = torch.randn(1, 2, 10, 4)
    # Append in two chunks; assemble out of cache.
    c.append(k_full[:, :, :7, :], v_full[:, :, :7, :])
    c.append(k_full[:, :, 7:, :], v_full[:, :, 7:, :])
    k_out, v_out = c.snapshot()
    assert torch.equal(k_out, k_full)
    assert torch.equal(v_out, v_full)
