"""Constraint-aware decoding primitives.

A library-agnostic toolkit for shaping probability distributions before
sampling: temperature scaling, top-k truncation, top-p (nucleus) truncation,
allow/forbid masking, deterministic greedy selection, and seedable sampling.

These primitives compose — each takes and returns a 1-D ``np.ndarray`` of
probabilities so they can be chained in any order ahead of
``sample_from_probs`` or ``greedy_argmax``. They are deliberately decoupled
from any specific model runtime: callers feed in their own logits, the
shape stage runs on CPU/NumPy, and sampling honours a caller-provided RNG
so behaviour is fully reproducible.

Goal-list ties: Constraint-based decoding, Top-k/top-p constrained decoding,
Greedy low-jitter decoding, Constraint-aware generation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax over ``logits / temperature``.

    Use temperature 1.0 for unmodified softmax, < 1 to sharpen toward the
    argmax, > 1 to flatten. Temperature 0 raises — use ``greedy_argmax``
    directly for deterministic selection.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0; use greedy_argmax() for 0 / deterministic")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    # Subtract max for numerical stability before exponentiation.
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return (exp / exp.sum()).astype(np.float32)


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the top ``k`` probabilities; zero the rest; renormalise.

    Ties at the k-th position are broken by lowest index (stable).
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    probs = np.asarray(probs, dtype=np.float32)
    if k >= probs.size:
        return probs / probs.sum() if probs.sum() > 0 else probs
    # argpartition gives unsorted top-k indices in O(n).
    top_idx = np.argpartition(probs, -k)[-k:]
    mask = np.zeros_like(probs, dtype=bool)
    mask[top_idx] = True
    out = np.where(mask, probs, 0.0).astype(np.float32)
    total = out.sum()
    if total > 0:
        out = out / total
    return out


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus (top-p) truncation: smallest set whose cumulative mass ≥ p.

    Always retains at least the argmax — caller can't end up with an empty
    distribution. Mass outside the nucleus is zeroed and the rest renormalised.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if p >= 1.0:
        return probs / probs.sum() if probs.sum() > 0 else probs
    # Sort descending by probability.
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    # Smallest k such that cumulative[k-1] >= p. ``searchsorted(left)`` on the
    # ascending cumulative returns that k.
    cutoff = int(np.searchsorted(cumulative, p, side="left")) + 1
    cutoff = max(cutoff, 1)  # always keep at least the argmax
    kept_idx = sorted_idx[:cutoff]
    out = np.zeros_like(probs, dtype=np.float32)
    out[kept_idx] = probs[kept_idx]
    total = out.sum()
    if total > 0:
        out = out / total
    return out


def apply_mask(
    probs: np.ndarray,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Zero indices where ``allowed_mask`` is False, then renormalise.

    Raises when the mask zeroes the entire distribution — callers must
    guarantee at least one allowed index.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if allowed_mask is None:
        return probs
    mask = np.asarray(allowed_mask, dtype=bool)
    if not mask.any():
        raise ValueError("mask forbids every index; refusing to produce empty distribution")
    out = np.where(mask, probs, 0.0).astype(np.float32)
    total = out.sum()
    if total > 0:
        out = out / total
    else:
        # All allowed entries were zero already — degrade to uniform over allowed set.
        out = mask.astype(np.float32)
        out = out / out.sum()
    return out


def greedy_argmax(probs: np.ndarray) -> int:
    """Deterministic argmax. Ties broken by lowest index.

    No RNG and no system clock — useful for low-jitter scheduling where the
    same input must always pick the same token.
    """
    probs = np.asarray(probs, dtype=np.float32)
    # np.argmax already returns the lowest index on ties.
    return int(np.argmax(probs))


def sample_from_probs(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample a single index from ``probs`` using the caller's seeded RNG.

    The RNG is threaded explicitly so behaviour is reproducible — passing
    the same seeded generator twice yields the same draw sequence.
    """
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if total <= 0:
        raise ValueError("probability distribution sums to zero")
    probs = probs / total
    return int(rng.choice(len(probs), p=probs))
