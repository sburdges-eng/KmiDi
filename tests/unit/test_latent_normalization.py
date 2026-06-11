"""Latent normalization primitives (Goal: Latent normalization and regularization,
Personalized latent calibration, Multimodal latent alignment)."""

import numpy as np
import pytest

from music_brain.latent.normalization import (
    center,
    clip_norm,
    l2_normalize,
    layer_norm,
    min_max_scale,
    standardize,
)

# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


def test_l2_normalize_vector_to_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)  # length 5
    out = l2_normalize(v)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out == pytest.approx([0.6, 0.8])


def test_l2_normalize_batch_per_row():
    batch = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(batch, axis=-1)
    assert np.linalg.norm(out, axis=-1) == pytest.approx([1.0, 1.0])


def test_l2_normalize_zero_vector_returns_zero():
    """An all-zero vector has no direction — return as-is, not NaN."""
    v = np.zeros(4, dtype=np.float32)
    out = l2_normalize(v)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# layer_norm
# ---------------------------------------------------------------------------


def test_layer_norm_zero_mean_unit_variance_per_row():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    out = layer_norm(x)
    assert out.mean(axis=-1) == pytest.approx([0.0], abs=1e-5)
    assert out.std(axis=-1) == pytest.approx([1.0], abs=1e-4)


def test_layer_norm_constant_row_returns_zeros():
    """All-same input → variance 0 → eps prevents div-by-zero, output ≈ 0."""
    x = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    out = layer_norm(x, eps=1e-5)
    assert out == pytest.approx(np.zeros((1, 3)), abs=1e-3)


# ---------------------------------------------------------------------------
# center
# ---------------------------------------------------------------------------


def test_center_subtracts_per_row_mean():
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    out = center(x, axis=-1)
    assert out.mean(axis=-1) == pytest.approx([0.0, 0.0])
    # Original variance preserved.
    assert out[0] == pytest.approx([-1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# clip_norm
# ---------------------------------------------------------------------------


def test_clip_norm_scales_long_vector_down():
    v = np.array([6.0, 8.0], dtype=np.float32)  # length 10
    out = clip_norm(v, max_norm=5.0)
    assert np.linalg.norm(out) == pytest.approx(5.0)
    # Direction preserved.
    assert out == pytest.approx([3.0, 4.0])


def test_clip_norm_leaves_short_vector_untouched():
    v = np.array([1.0, 2.0], dtype=np.float32)
    out = clip_norm(v, max_norm=10.0)
    assert out == pytest.approx(v)


def test_clip_norm_zero_vector_returns_zero():
    v = np.zeros(3, dtype=np.float32)
    out = clip_norm(v, max_norm=1.0)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# min_max_scale
# ---------------------------------------------------------------------------


def test_min_max_scale_to_unit_interval():
    x = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    out = min_max_scale(x)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)


def test_min_max_scale_constant_input_returns_zeros():
    x = np.array([7.0, 7.0, 7.0], dtype=np.float32)
    out = min_max_scale(x)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# standardize (training-time mean/std applied at inference)
# ---------------------------------------------------------------------------


def test_standardize_uses_provided_mean_and_std():
    """Reusing training-time statistics at inference: x' = (x - mean) / std."""
    mean = np.array([1.0, 2.0], dtype=np.float32)
    std = np.array([0.5, 2.0], dtype=np.float32)
    x = np.array([[1.5, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = standardize(x, mean=mean, std=std)
    assert out[0] == pytest.approx([1.0, 1.0])
    assert out[1] == pytest.approx([0.0, -1.0])


def test_standardize_zero_std_falls_back_to_eps():
    """A feature with zero training-time std (constant) should not produce NaN."""
    mean = np.array([1.0, 2.0], dtype=np.float32)
    std = np.array([0.0, 2.0], dtype=np.float32)
    x = np.array([[1.5, 4.0]], dtype=np.float32)
    out = standardize(x, mean=mean, std=std)
    assert np.all(np.isfinite(out))
