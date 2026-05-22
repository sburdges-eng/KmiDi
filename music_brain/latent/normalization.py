"""Latent-space normalization primitives.

Stateless NumPy helpers for working with embedding vectors and batches:
unit-length projection, layer-norm, mean-centering, norm clipping, and
training-time standardization. Used as the shared base for retrieval,
projection, and calibration layers — keeps every component agreeing on
the same numerical conventions instead of each rolling its own.

Goal-list ties: Latent normalization and regularization,
Personalized latent calibration (standardize), Multimodal latent alignment
(l2_normalize as a shared pre-step).
"""

from __future__ import annotations

import numpy as np

_DEFAULT_EPS = 1e-12


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = _DEFAULT_EPS) -> np.ndarray:
    """Project ``x`` to unit L2 norm along ``axis``.

    Zero-norm vectors are left as zero (no NaN). The output dtype is float32.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    safe = np.where(norm > eps, norm, 1.0)
    out = (x / safe).astype(np.float32)
    # Zero-vector inputs stay zero (np.where above kept them as x/1.0 == x == 0).
    return out


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-row mean-0, variance-1 normalization (no learned affine).

    Constant rows produce all-zero output instead of NaN.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)).astype(np.float32)


def center(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Subtract the per-axis mean. Variance is preserved."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=axis, keepdims=True)
    return (x - mean).astype(np.float32)


def clip_norm(x: np.ndarray, max_norm: float, axis: int = -1) -> np.ndarray:
    """Scale ``x`` down so that ``||x||_2 <= max_norm`` along ``axis``.

    Vectors already within the cap are returned unchanged. Zero vectors
    pass through unchanged (no NaN).
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be > 0")
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    # Scale factor: min(1, max_norm/norm); zero-norm becomes 1 (no scaling).
    safe = np.where(norm > 0, norm, 1.0)
    scale = np.minimum(1.0, max_norm / safe)
    return (x * scale).astype(np.float32)


def min_max_scale(x: np.ndarray, eps: float = _DEFAULT_EPS) -> np.ndarray:
    """Linearly rescale to ``[0, 1]`` across the entire array.

    Constant input (max == min) returns zeros.
    """
    x = np.asarray(x, dtype=np.float32)
    lo = x.min()
    hi = x.max()
    span = hi - lo
    if span < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / span).astype(np.float32)


def standardize(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply ``(x - mean) / std`` using caller-supplied training statistics.

    Zero-std features fall back to a small epsilon so the operation never
    produces NaN — useful for features that were constant in the training
    set but might vary at inference.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    safe_std = np.where(std > eps, std, eps)
    return ((x - mean) / safe_std).astype(np.float32)
