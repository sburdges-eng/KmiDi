"""Multimodal fusion primitives (Goal: Multimodal fusion engine,
Latent emotional steering via gated fusion, Multimodal latent alignment)."""

import numpy as np
import pytest

from music_brain.latent.fusion import (
    concat_fuse,
    gated_fuse,
    normalized_average,
    weighted_sum,
)


# ---------------------------------------------------------------------------
# concat_fuse
# ---------------------------------------------------------------------------


def test_concat_fuse_joins_along_last_axis():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    out = concat_fuse(a, b)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_concat_fuse_handles_three_or_more_inputs():
    out = concat_fuse(
        np.array([1.0], dtype=np.float32),
        np.array([2.0, 3.0], dtype=np.float32),
        np.array([4.0], dtype=np.float32),
    )
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_concat_fuse_at_least_one_arg():
    with pytest.raises(ValueError, match="at least one"):
        concat_fuse()


# ---------------------------------------------------------------------------
# weighted_sum
# ---------------------------------------------------------------------------


def test_weighted_sum_with_equal_weights():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    out = weighted_sum([a, b], weights=[0.5, 0.5])
    assert out.tolist() == [2.0, 3.0]


def test_weighted_sum_normalises_weights_implicitly():
    """Caller may pass un-normalised weights — function normalises."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    out = weighted_sum([a, b], weights=[3.0, 1.0])  # 3:1 ratio
    assert out[0] == pytest.approx(0.75)
    assert out[1] == pytest.approx(0.25)


def test_weighted_sum_shape_mismatch_raises():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        weighted_sum([a, b], weights=[0.5, 0.5])


def test_weighted_sum_length_mismatch_raises():
    a = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        weighted_sum([a, a], weights=[0.5])


def test_weighted_sum_zero_weights_raises():
    a = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="weights"):
        weighted_sum([a, a], weights=[0.0, 0.0])


# ---------------------------------------------------------------------------
# gated_fuse
# ---------------------------------------------------------------------------


def test_gated_fuse_scalar_gate_linear_interpolation():
    """gate=0.7 → 0.7*a + 0.3*b."""
    a = np.array([1.0, 1.0], dtype=np.float32)
    b = np.array([0.0, 0.0], dtype=np.float32)
    out = gated_fuse(a, b, gate=0.7)
    assert out == pytest.approx([0.7, 0.7])


def test_gated_fuse_vector_gate_elementwise():
    a = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    gate = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = gated_fuse(a, b, gate=gate)
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_gated_fuse_gate_outside_unit_range_raises():
    a = np.zeros(2, dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="gate"):
        gated_fuse(a, b, gate=1.5)
    with pytest.raises(ValueError, match="gate"):
        gated_fuse(a, b, gate=np.array([0.5, 1.7], dtype=np.float32))


# ---------------------------------------------------------------------------
# normalized_average
# ---------------------------------------------------------------------------


def test_normalized_average_yields_unit_length_output():
    a = np.array([3.0, 0.0], dtype=np.float32)  # length 3
    b = np.array([0.0, 4.0], dtype=np.float32)  # length 4
    out = normalized_average([a, b])
    assert np.linalg.norm(out) == pytest.approx(1.0)


def test_normalized_average_preserves_direction_of_aligned_inputs():
    """When all inputs already point the same way, normalized average == l2_norm of one."""
    a = np.array([3.0, 0.0], dtype=np.float32)
    out = normalized_average([a, a, a])
    assert out == pytest.approx([1.0, 0.0])


def test_normalized_average_zero_input_returns_zero():
    out = normalized_average([np.zeros(3, dtype=np.float32)])
    assert np.all(out == 0.0)


def test_normalized_average_zero_inputs_raises():
    with pytest.raises(ValueError, match="at least one"):
        normalized_average([])
