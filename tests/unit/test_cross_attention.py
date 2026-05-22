"""Tests for ``music_brain.cross_attention.CrossAttention``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.cross_attention import CrossAttention  # noqa: E402


def _attn(query_dim: int = 16, kv_dim: int = None, heads: int = 4) -> CrossAttention:
    torch.manual_seed(0)
    return CrossAttention(query_dim, kv_dim=kv_dim, num_heads=heads)


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------

def test_query_dim_positive() -> None:
    with pytest.raises(ValueError, match="query_dim"):
        CrossAttention(0)


def test_num_heads_must_divide_query_dim() -> None:
    with pytest.raises(ValueError, match="evenly divide"):
        CrossAttention(16, num_heads=5)


def test_dropout_range() -> None:
    with pytest.raises(ValueError, match="dropout"):
        CrossAttention(16, dropout=1.5)


def test_default_kv_dim_matches_query_dim() -> None:
    attn = _attn(query_dim=16)
    assert attn.kv_dim == 16


def test_explicit_kv_dim_used() -> None:
    attn = _attn(query_dim=16, kv_dim=32)
    assert attn.kv_dim == 32


# ----------------------------------------------------------------------
# Forward shape
# ----------------------------------------------------------------------

def test_output_and_attn_shapes() -> None:
    attn = _attn(query_dim=16, kv_dim=8, heads=4)
    q = torch.randn(2, 5, 16)
    kv = torch.randn(2, 7, 8)
    out, weights = attn(q, kv)
    assert out.shape == (2, 5, 16)
    assert weights.shape == (2, 5, 7)


def test_attn_weights_sum_to_one_per_query_position() -> None:
    attn = _attn()
    q = torch.randn(1, 3, 16)
    kv = torch.randn(1, 4, 16)
    _, weights = attn(q, kv)
    # Averaging over heads of softmax distributions yields rows that still sum to 1.
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------

def test_key_padding_mask_zeros_out_masked_positions() -> None:
    attn = _attn(query_dim=8, heads=2)
    q = torch.randn(1, 3, 8)
    kv = torch.randn(1, 4, 8)
    # Mask out positions 2 and 3.
    pad = torch.tensor([[False, False, True, True]])
    _, weights = attn(q, kv, key_padding_mask=pad)
    # Masked columns must have zero weight everywhere.
    assert (weights[..., 2] == 0).all()
    assert (weights[..., 3] == 0).all()
    # Unmasked columns still sum to 1 per query.
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)


def test_attn_mask_additive_blocks_specific_positions() -> None:
    attn = _attn(query_dim=8, heads=2)
    q = torch.randn(1, 2, 8)
    kv = torch.randn(1, 3, 8)
    mask = torch.zeros(2, 3)
    mask[0, 1] = float("-inf")  # block (query 0 -> kv 1) only
    _, weights = attn(q, kv, attn_mask=mask)
    assert weights[0, 0, 1].item() == 0.0
    # Other entries remain non-zero.
    assert weights[0, 1, 1].item() > 0.0


# ----------------------------------------------------------------------
# Determinism / gradient flow
# ----------------------------------------------------------------------

def test_forward_is_deterministic_with_fixed_seed() -> None:
    attn = _attn()
    torch.manual_seed(42)
    q = torch.randn(2, 4, 16)
    kv = torch.randn(2, 6, 16)
    out1, _ = attn(q, kv)
    out2, _ = attn(q, kv)
    assert torch.allclose(out1, out2)


def test_gradients_flow_through_all_projections() -> None:
    attn = _attn()
    q = torch.randn(1, 3, 16, requires_grad=True)
    kv = torch.randn(1, 4, 16, requires_grad=True)
    out, _ = attn(q, kv)
    loss = out.pow(2).mean()
    loss.backward()
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
        for p in proj.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum().item() > 0
    assert q.grad is not None and kv.grad is not None


def test_parameter_count_matches_expected() -> None:
    attn = _attn(query_dim=16, kv_dim=16, heads=4)
    total = sum(p.numel() for p in attn.parameters())
    # 4 linear layers, each 16x16 (no bias) = 4 * 256 = 1024.
    assert total == 1024


def test_dropout_does_nothing_when_training_false() -> None:
    attn = CrossAttention(16, num_heads=4, dropout=0.99)
    q = torch.randn(1, 3, 16)
    kv = torch.randn(1, 4, 16)
    out1, _ = attn(q, kv, training=False)
    out2, _ = attn(q, kv, training=False)
    assert torch.allclose(out1, out2)
