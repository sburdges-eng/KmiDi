"""Tests for ``music_brain.latent.regularization``.

Closes the **Latent normalization / regularization** spec item. The
existing JEPA + world-model stack emits raw latents with no norm or
variance constraints, leaving them vulnerable to feature collapse and
unbounded growth during world-model rollouts.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from music_brain.latent.regularization import (  # noqa: E402
    L2NormProjection,
    VarianceFloor,
)

# ----------------------------------------------------------------------
# L2NormProjection
# ----------------------------------------------------------------------


def test_l2_norm_projects_above_radius_back_to_radius() -> None:
    proj = L2NormProjection(radius=1.0)
    torch.manual_seed(0)
    z = torch.randn(4, 8, dtype=torch.float32) * 10.0  # large norms
    out = proj(z)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.all(norms <= 1.0 + 1e-5)
    # And every row that was above 1.0 should now sit *at* the radius.
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_l2_norm_leaves_inside_ball_alone() -> None:
    proj = L2NormProjection(radius=1.0)
    z = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    out = proj(z)
    assert torch.allclose(out, z, atol=1e-6)


def test_l2_norm_preserves_direction() -> None:
    proj = L2NormProjection(radius=1.0)
    z = torch.tensor([[3.0, 4.0]], dtype=torch.float32)  # norm 5
    out = proj(z)
    # Direction (3, 4) → unit (0.6, 0.8)
    assert torch.allclose(out, torch.tensor([[0.6, 0.8]]), atol=1e-6)


def test_l2_norm_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        L2NormProjection(radius=0.0)
    with pytest.raises(ValueError, match="radius"):
        L2NormProjection(radius=-1.0)


def test_l2_norm_handles_zero_vector_without_nan() -> None:
    proj = L2NormProjection(radius=1.0)
    z = torch.zeros(2, 4, dtype=torch.float32)
    out = proj(z)
    assert torch.all(torch.isfinite(out))
    assert torch.allclose(out, z)  # zero stays zero


def test_l2_norm_is_differentiable_at_boundary() -> None:
    proj = L2NormProjection(radius=1.0)
    z = torch.tensor([[3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    out = proj(z)
    out.sum().backward()
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))


# ----------------------------------------------------------------------
# VarianceFloor
# ----------------------------------------------------------------------


def test_variance_floor_zero_loss_when_variance_above_floor() -> None:
    floor = VarianceFloor(min_var=1e-3)
    torch.manual_seed(0)
    z = torch.randn(64, 8, dtype=torch.float32)  # variance ~1 across batch
    z_out, aux = floor(z)
    assert torch.equal(z_out, z)  # input passes through untouched
    assert aux.item() == pytest.approx(0.0, abs=1e-5)


def test_variance_floor_positive_loss_when_below_floor() -> None:
    floor = VarianceFloor(min_var=1.0)
    # Collapse: every row identical → zero variance.
    z = torch.ones(8, 4, dtype=torch.float32)
    z_out, aux = floor(z)
    assert torch.equal(z_out, z)
    # All 4 feature dims have variance 0; loss = sum(relu(1.0 - 0.0)) = 4.0
    assert aux.item() == pytest.approx(4.0, abs=1e-5)


def test_variance_floor_loss_is_zero_for_partial_collapse_above_floor() -> None:
    floor = VarianceFloor(min_var=1e-3)
    z = torch.zeros(8, 3, dtype=torch.float32)
    z[:, 0] = torch.randn(8)  # only dim 0 has variance
    _, aux = floor(z)
    # Dims 1 and 2 have zero variance, each contributing min_var = 1e-3 → 2e-3
    assert aux.item() == pytest.approx(2e-3, abs=1e-5)


def test_variance_floor_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="min_var"):
        VarianceFloor(min_var=0.0)
    with pytest.raises(ValueError, match="min_var"):
        VarianceFloor(min_var=-1e-3)


def test_variance_floor_requires_at_least_two_batch_samples() -> None:
    floor = VarianceFloor(min_var=1e-3)
    with pytest.raises(ValueError, match="batch"):
        floor(torch.zeros(1, 4, dtype=torch.float32))


def test_variance_floor_is_differentiable() -> None:
    floor = VarianceFloor(min_var=1.0)
    z = torch.ones(8, 4, dtype=torch.float32, requires_grad=True)
    _, aux = floor(z)
    aux.backward()
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_l2_norm_deterministic_across_calls() -> None:
    proj = L2NormProjection(radius=2.5)
    torch.manual_seed(42)
    z = torch.randn(4, 8, dtype=torch.float32) * 100.0
    out_a = proj(z)
    out_b = proj(z)
    assert torch.equal(out_a, out_b)
    assert math.isclose(out_a.norm(dim=-1).max().item(), 2.5, abs_tol=1e-5)
