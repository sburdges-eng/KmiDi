"""LinearProjection — small affine bridge between embedding spaces.

Goal: Conditioning bridge projection layer (direct),
Soft prompt or adapter injection (small projections are the adapter shape)."""

from pathlib import Path

import numpy as np
import pytest

from music_brain.latent.projection import LinearProjection

# ---------------------------------------------------------------------------
# Construction & shape
# ---------------------------------------------------------------------------


def test_projection_shape_matches_constructor_args():
    proj = LinearProjection(in_dim=8, out_dim=4, seed=0)
    assert proj.weight.shape == (8, 4)
    assert proj.bias.shape == (4,)


def test_projection_seed_is_deterministic():
    a = LinearProjection(in_dim=4, out_dim=2, seed=42)
    b = LinearProjection(in_dim=4, out_dim=2, seed=42)
    np.testing.assert_array_equal(a.weight, b.weight)
    np.testing.assert_array_equal(a.bias, b.bias)


def test_projection_different_seeds_differ():
    a = LinearProjection(in_dim=4, out_dim=2, seed=0)
    b = LinearProjection(in_dim=4, out_dim=2, seed=1)
    # Extremely unlikely to coincide given Glorot init scale.
    assert not np.allclose(a.weight, b.weight)


def test_invalid_dims_raise():
    with pytest.raises(ValueError):
        LinearProjection(in_dim=0, out_dim=4)
    with pytest.raises(ValueError):
        LinearProjection(in_dim=4, out_dim=0)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def test_apply_single_vector_returns_projected_shape():
    proj = LinearProjection(in_dim=4, out_dim=2, seed=0)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = proj.apply(x)
    assert y.shape == (2,)


def test_apply_batch_returns_projected_batch_shape():
    proj = LinearProjection(in_dim=4, out_dim=2, seed=0)
    batch = np.ones((5, 4), dtype=np.float32)
    y = proj.apply(batch)
    assert y.shape == (5, 2)


def test_apply_input_dim_mismatch_raises():
    proj = LinearProjection(in_dim=4, out_dim=2, seed=0)
    bad = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="dim"):
        proj.apply(bad)


def test_apply_uses_weight_and_bias():
    """Set known weight/bias and verify apply == x @ W + b."""
    proj = LinearProjection(in_dim=2, out_dim=3, seed=0)
    proj.weight = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    proj.bias = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    y = proj.apply(np.array([1.0, 2.0], dtype=np.float32))
    # x @ W: [1*1+2*0, 1*0+2*1, 1*(-1)+2*1] = [1, 2, 1]
    # + bias: [1.5, 2.5, 1.5]
    assert y == pytest.approx([1.5, 2.5, 1.5])


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_to_json_and_from_json_round_trip():
    a = LinearProjection(in_dim=4, out_dim=2, seed=7)
    payload = a.to_json()
    b = LinearProjection.from_json(payload)
    np.testing.assert_allclose(a.weight, b.weight, rtol=1e-6)
    np.testing.assert_allclose(a.bias, b.bias, rtol=1e-6)


def test_from_json_with_invalid_payload_raises():
    with pytest.raises(ValueError):
        LinearProjection.from_json("not-json")


def test_save_load_round_trip(tmp_path: Path):
    a = LinearProjection(in_dim=4, out_dim=2, seed=7)
    a.save(tmp_path / "proj.json")
    b = LinearProjection.load(tmp_path / "proj.json")
    np.testing.assert_allclose(a.weight, b.weight, rtol=1e-6)


# ---------------------------------------------------------------------------
# Glorot init magnitude sanity
# ---------------------------------------------------------------------------


def test_glorot_init_keeps_variance_in_reasonable_range():
    """Glorot variance ~ 2/(in+out). For 64x64 → ~0.03; std ~ 0.17.
    We don't assert exact theoretical variance, just that values are
    bounded and non-uniform."""
    proj = LinearProjection(in_dim=64, out_dim=64, seed=0)
    flat = proj.weight.flatten()
    # No value should explode; Glorot keeps values bounded.
    assert np.max(np.abs(flat)) < 2.0
    # Variance is nonzero (not all the same value).
    assert flat.std() > 1e-4
