"""Tests for ``music_brain.latent.decoding`` — constraint-aware sampling.

Closes the **Constraint-based decoding**, **Top-k / top-p constrained
decoding**, and **Greedy low-jitter decoding** spec items, plus the
**Constraint-aware generation** umbrella.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.latent.decoding import (  # noqa: E402
    DecodeConfig,
    greedy_argmax,
    sample_with_constraints,
)

# ----------------------------------------------------------------------
# Greedy low-jitter decoder
# ----------------------------------------------------------------------


def test_greedy_picks_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    out = greedy_argmax(logits)
    assert out.tolist() == [1]


def test_greedy_supports_batch_input() -> None:
    logits = torch.tensor([[1.0, 5.0, 0.0], [10.0, 1.0, 1.0]])
    out = greedy_argmax(logits)
    assert out.tolist() == [1, 0]


def test_greedy_honors_forbidden_tokens() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    out = greedy_argmax(logits, forbidden_tokens=[1])
    assert out.tolist() == [3]  # next-best after token 1


def test_greedy_is_deterministic_under_repeated_calls() -> None:
    logits = torch.randn(8, 16)
    out_a = greedy_argmax(logits)
    out_b = greedy_argmax(logits)
    assert torch.equal(out_a, out_b)


# ----------------------------------------------------------------------
# DecodeConfig validation
# ----------------------------------------------------------------------


def test_decode_config_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        DecodeConfig(temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        DecodeConfig(temperature=-1.0)


def test_decode_config_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        DecodeConfig(top_k=-1)


def test_decode_config_rejects_invalid_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        DecodeConfig(top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        DecodeConfig(top_p=1.5)


# ----------------------------------------------------------------------
# Constraint sampler
# ----------------------------------------------------------------------


def test_sample_with_no_constraints_is_softmax_distribution() -> None:
    """With temperature=1.0, top_k=0 (off), top_p=1.0 (off), and a
    fixed seed, the sampler must produce the same token as a pure
    multinomial sample of the softmax."""
    torch.manual_seed(0)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    cfg = DecodeConfig(temperature=1.0, top_k=0, top_p=1.0)
    rng = torch.Generator().manual_seed(42)
    out = sample_with_constraints(logits, cfg, generator=rng)
    assert out.shape == (1,)
    assert 0 <= int(out.item()) < 4


def test_top_k_zeros_tail() -> None:
    """top_k=2 must pick only from the top 2 logits."""
    torch.manual_seed(0)
    # Logit-1 dominates; logit-3 dominates the bottom half. With top_k=2,
    # only token 1 and token 3 are eligible.
    logits = torch.tensor([[1.0, 10.0, 1.0, 9.0]])
    cfg = DecodeConfig(temperature=1.0, top_k=2)
    rng = torch.Generator().manual_seed(0)
    eligible = set()
    for _ in range(50):
        eligible.add(int(sample_with_constraints(logits, cfg, generator=rng).item()))
    assert eligible.issubset({1, 3})


def test_top_p_keeps_nucleus_only() -> None:
    """top_p=0.5 keeps just the largest tokens whose cumprob first crosses 0.5.

    With logits [0, 0, 100, 0] the softmax is a one-hot on index 2 — so
    only that token survives top_p filtering regardless of threshold.
    """
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, 100.0, 0.0]])
    cfg = DecodeConfig(temperature=1.0, top_p=0.5)
    rng = torch.Generator().manual_seed(0)
    out = sample_with_constraints(logits, cfg, generator=rng)
    assert int(out.item()) == 2


def test_forbidden_tokens_never_sampled() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    cfg = DecodeConfig(temperature=1.0)
    rng = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(50):
        seen.add(
            int(sample_with_constraints(logits, cfg, forbidden_tokens=[0, 2], generator=rng).item())
        )
    assert seen.isdisjoint({0, 2})
    assert seen.issubset({1, 3})


def test_temperature_zero_falls_back_to_greedy() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    cfg = DecodeConfig(temperature=1e-6)  # effectively greedy
    out = sample_with_constraints(logits, cfg, generator=torch.Generator().manual_seed(0))
    assert int(out.item()) == 1


def test_all_tokens_forbidden_raises() -> None:
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    cfg = DecodeConfig()
    with pytest.raises(ValueError, match="forbidden"):
        sample_with_constraints(logits, cfg, forbidden_tokens=[0, 1, 2])


def test_greedy_all_tokens_forbidden_raises() -> None:
    """Regression: greedy_argmax must mirror sample_with_constraints
    and reject an all-forbidden mask. Otherwise the
    incremental-decode jitter-fallback path can emit a forbidden
    token. Reported by Cursor Bugbot on PR #196 (round 3)."""
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="forbidden"):
        greedy_argmax(logits, forbidden_tokens=[0, 1, 2])
