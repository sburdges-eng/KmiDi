"""Multimodal latent fusion primitives.

Four NumPy strategies for combining 2+ embedding vectors into one, each
appropriate for a different shape of fusion problem:

  - ``concat_fuse(*v)`` — preserves all information from every modality;
    growth in dimension is the cost.
  - ``weighted_sum(vs, weights)`` — linear blend with implicit weight
    normalisation; same dimension as inputs.
  - ``gated_fuse(a, b, gate)`` — linear interpolation \\alpha·a + (1-\\alpha)·b
    where \\alpha can be scalar or per-element. The host-side primitive
    for "Latent emotional steering": slide a global mood-mix knob, or
    steer per-dimension.
  - ``normalized_average(vs)`` — L2-normalised mean direction; the
    standard "align modalities on the unit sphere" recipe.

Stdlib + numpy.

Goal-list ties: Multimodal fusion engine (direct), Latent emotional
steering (gated_fuse), Multimodal latent alignment (normalized_average).
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

_DEFAULT_EPS = 1e-12


def concat_fuse(*vectors: np.ndarray) -> np.ndarray:
    """Concatenate vectors along the last axis."""
    if not vectors:
        raise ValueError("concat_fuse requires at least one vector")
    return np.concatenate([np.asarray(v, dtype=np.float32) for v in vectors]).astype(np.float32)


def weighted_sum(vectors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted linear blend. Weights are normalised to sum to 1.

    Raises if ``vectors`` and ``weights`` differ in length, if any vector
    has a different shape, or if all weights sum to zero.
    """
    if len(vectors) != len(weights):
        raise ValueError(f"length mismatch: {len(vectors)} vectors vs {len(weights)} weights")
    if not vectors:
        raise ValueError("weighted_sum requires at least one vector")
    arrs = [np.asarray(v, dtype=np.float32) for v in vectors]
    shape = arrs[0].shape
    for a in arrs[1:]:
        if a.shape != shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {shape}")
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError("weights sum to zero — cannot normalise")
    w = w / total
    out = np.zeros(shape, dtype=np.float32)
    for arr, scalar in zip(arrs, w):
        out += (arr * scalar).astype(np.float32)
    return out


def gated_fuse(
    a: np.ndarray,
    b: np.ndarray,
    gate: Union[float, np.ndarray],
) -> np.ndarray:
    """Linear interpolation ``gate * a + (1 - gate) * b``.

    ``gate`` may be a scalar (global) or a vector of the same shape as
    ``a`` and ``b`` (per-element steering). Values must lie in ``[0, 1]``.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if isinstance(gate, np.ndarray):
        g = gate.astype(np.float32)
        if g.shape != a.shape:
            raise ValueError(f"gate shape mismatch: {g.shape} vs {a.shape}")
        if np.any(g < 0) or np.any(g > 1):
            raise ValueError("gate vector entries must lie in [0, 1]")
    else:
        if not 0.0 <= gate <= 1.0:
            raise ValueError(f"gate must lie in [0, 1]; got {gate}")
        g = np.float32(gate)
    return (g * a + (1.0 - g) * b).astype(np.float32)


def normalized_average(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of the inputs, then L2-normalised.

    Zero-mean output (vectors cancel out) is returned as zero instead of
    NaN. The recipe of "average then re-normalise to the unit sphere" is
    the standard alignment step for late multimodal fusion.
    """
    if not vectors:
        raise ValueError("normalized_average requires at least one vector")
    arrs = [np.asarray(v, dtype=np.float32) for v in vectors]
    shape = arrs[0].shape
    for a in arrs[1:]:
        if a.shape != shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {shape}")
    mean = np.mean(arrs, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    if norm <= _DEFAULT_EPS:
        return np.zeros_like(mean)
    return (mean / norm).astype(np.float32)
