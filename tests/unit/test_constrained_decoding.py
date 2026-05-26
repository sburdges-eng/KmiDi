"""Constrained decoding primitives (Goal: Constraint-based decoding,
Top-k/top-p constrained decoding, Greedy low-jitter decoding,
Constraint-aware generation)."""

import numpy as np
import pytest

from music_brain.decoding.constrained import (
    apply_mask,
    apply_temperature,
    greedy_argmax,
    sample_from_probs,
    top_k_filter,
    top_p_filter,
)


# ---------------------------------------------------------------------------
# apply_temperature
# ---------------------------------------------------------------------------


def test_temperature_one_returns_softmax_unchanged():
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probs = apply_temperature(logits, temperature=1.0)
    expected = np.exp(logits) / np.exp(logits).sum()
    assert probs == pytest.approx(expected, abs=1e-6)
    assert probs.sum() == pytest.approx(1.0)


def test_temperature_below_one_sharpens_distribution():
    logits = np.array([1.0, 2.0], dtype=np.float32)
    cold = apply_temperature(logits, temperature=0.1)
    warm = apply_temperature(logits, temperature=1.0)
    # Cold (low temperature) → concentrates mass on the argmax.
    assert cold[1] > warm[1]


def test_temperature_above_one_flattens_distribution():
    logits = np.array([1.0, 2.0], dtype=np.float32)
    hot = apply_temperature(logits, temperature=10.0)
    base = apply_temperature(logits, temperature=1.0)
    # Hot → spreads mass; argmax delta shrinks vs uniform.
    assert (hot[1] - hot[0]) < (base[1] - base[0])


def test_temperature_zero_raises():
    """Temperature 0 is greedy — caller should use greedy_argmax directly."""
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(np.array([1.0, 2.0]), temperature=0.0)


# ---------------------------------------------------------------------------
# top_k_filter
# ---------------------------------------------------------------------------


def test_top_k_zeros_out_low_probability_indices():
    probs = np.array([0.1, 0.4, 0.2, 0.3], dtype=np.float32)
    filtered = top_k_filter(probs, k=2)
    # Keeps indices 1 (0.4) and 3 (0.3); 0 and 2 → 0.
    assert filtered[0] == 0.0
    assert filtered[2] == 0.0
    assert filtered[1] > 0.0
    assert filtered[3] > 0.0
    # Renormalised.
    assert filtered.sum() == pytest.approx(1.0)


def test_top_k_with_k_equal_to_length_returns_unchanged():
    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    filtered = top_k_filter(probs, k=4)
    assert filtered == pytest.approx(probs)


def test_top_k_k_one_returns_argmax_only():
    probs = np.array([0.1, 0.4, 0.2, 0.3], dtype=np.float32)
    filtered = top_k_filter(probs, k=1)
    assert filtered[1] == pytest.approx(1.0)
    assert sum(p for p in filtered if p > 0) == pytest.approx(1.0)
    assert int(np.argmax(filtered)) == 1


def test_top_k_zero_raises():
    with pytest.raises(ValueError, match="k"):
        top_k_filter(np.array([0.5, 0.5]), k=0)


# ---------------------------------------------------------------------------
# top_p_filter (nucleus)
# ---------------------------------------------------------------------------


def test_top_p_keeps_smallest_set_summing_above_p():
    probs = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    # p=0.7 → cumulative {0.5, 0.8} crosses on the 2nd → keep top 2.
    filtered = top_p_filter(probs, p=0.7)
    assert filtered[2] == 0.0
    assert filtered[3] == 0.0
    assert filtered[0] > 0.0
    assert filtered[1] > 0.0
    assert filtered.sum() == pytest.approx(1.0)


def test_top_p_one_keeps_everything():
    probs = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    filtered = top_p_filter(probs, p=1.0)
    assert filtered == pytest.approx(probs)


def test_top_p_low_p_still_keeps_at_least_one():
    """Even p=0.01 must keep the argmax — no empty distribution."""
    probs = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    filtered = top_p_filter(probs, p=0.01)
    assert filtered[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


def test_apply_mask_zeros_forbidden_indices_and_renormalises():
    probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    allowed = np.array([True, False, True, True])
    filtered = apply_mask(probs, allowed_mask=allowed)
    assert filtered[1] == 0.0
    assert filtered.sum() == pytest.approx(1.0)


def test_apply_mask_all_forbidden_raises():
    probs = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="mask"):
        apply_mask(probs, allowed_mask=np.array([False, False]))


def test_apply_mask_passthrough_when_mask_is_none():
    probs = np.array([0.5, 0.5], dtype=np.float32)
    assert apply_mask(probs, allowed_mask=None) == pytest.approx(probs)


# ---------------------------------------------------------------------------
# greedy_argmax (deterministic, low-jitter)
# ---------------------------------------------------------------------------


def test_greedy_argmax_returns_max_index():
    probs = np.array([0.1, 0.7, 0.2], dtype=np.float32)
    assert greedy_argmax(probs) == 1


def test_greedy_argmax_breaks_ties_with_lowest_index():
    """Determinism guarantee: equal probs → lowest index. No RNG."""
    probs = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    assert greedy_argmax(probs) == 0


# ---------------------------------------------------------------------------
# sample_from_probs (seeded)
# ---------------------------------------------------------------------------


def test_sample_from_probs_is_deterministic_with_seed():
    probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    seq_a = [sample_from_probs(probs, rng_a) for _ in range(20)]
    seq_b = [sample_from_probs(probs, rng_b) for _ in range(20)]
    assert seq_a == seq_b


def test_sample_from_probs_respects_distribution():
    """Over many draws, frequencies approximate the input probs."""
    probs = np.array([0.1, 0.7, 0.2], dtype=np.float32)
    rng = np.random.default_rng(0)
    draws = np.array([sample_from_probs(probs, rng) for _ in range(2000)])
    freq = np.array([np.mean(draws == i) for i in range(len(probs))])
    assert freq == pytest.approx(probs, abs=0.05)


# ---------------------------------------------------------------------------
# Composed pipeline: temperature → top_p → mask → sample
# ---------------------------------------------------------------------------


def test_composed_pipeline_respects_mask_after_topk_topp():
    """A token zeroed by mask never appears in samples, even if it survived top_k."""
    logits = np.array([3.0, 2.0, 1.5, 1.0], dtype=np.float32)
    probs = apply_temperature(logits, temperature=1.0)
    probs = top_k_filter(probs, k=3)
    probs = top_p_filter(probs, p=0.95)
    # Forbid index 0 (which would otherwise dominate).
    probs = apply_mask(probs, allowed_mask=np.array([False, True, True, True]))
    rng = np.random.default_rng(1)
    draws = [sample_from_probs(probs, rng) for _ in range(500)]
    assert 0 not in set(draws)
