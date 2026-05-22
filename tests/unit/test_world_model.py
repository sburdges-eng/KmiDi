"""Tests for ``music_brain.world_model.WorldModel``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.world_model import WorldModel  # noqa: E402


def _wm(state_dim: int = 8, action_dim: int = 0, hidden_dim: int = None) -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------

def test_state_dim_positive() -> None:
    with pytest.raises(ValueError, match="state_dim"):
        WorldModel(state_dim=0)


def test_action_dim_non_negative() -> None:
    with pytest.raises(ValueError, match="action_dim"):
        WorldModel(state_dim=8, action_dim=-1)


def test_default_hidden_matches_state_dim() -> None:
    wm = _wm(state_dim=10)
    assert wm.hidden_dim == 10


def test_explicit_hidden_dim_used() -> None:
    wm = _wm(state_dim=8, hidden_dim=32)
    assert wm.hidden_dim == 32


# ----------------------------------------------------------------------
# step()
# ----------------------------------------------------------------------

def test_step_unconditioned_shape() -> None:
    wm = _wm(state_dim=8)
    s = torch.randn(3, 8)
    next_s, h = wm.step(s)
    assert next_s.shape == (3, 8)
    assert h.shape == (3, 8)


def test_step_conditioned_requires_action() -> None:
    wm = _wm(state_dim=8, action_dim=4)
    s = torch.randn(2, 8)
    with pytest.raises(ValueError, match="action is required"):
        wm.step(s)


def test_step_conditioned_shape() -> None:
    wm = _wm(state_dim=8, action_dim=4)
    s = torch.randn(2, 8)
    a = torch.randn(2, 4)
    next_s, h = wm.step(s, action=a)
    assert next_s.shape == (2, 8)
    assert h.shape == (2, 8)


def test_step_action_batch_must_match_state() -> None:
    wm = _wm(state_dim=8, action_dim=4)
    s = torch.randn(2, 8)
    a = torch.randn(3, 4)  # mismatched batch
    with pytest.raises(ValueError, match="batch dim"):
        wm.step(s, action=a)


def test_step_default_hidden_is_zeros() -> None:
    """First call with hidden=None must seed with zeros (deterministic start)."""
    wm = _wm(state_dim=8)
    s = torch.randn(1, 8)
    _, h1 = wm.step(s, hidden=None)
    # Re-run with explicit zero hidden — outputs must match.
    _, h2 = wm.step(s, hidden=torch.zeros(1, 8))
    assert torch.allclose(h1, h2)


def test_step_is_deterministic_with_same_inputs() -> None:
    wm = _wm(state_dim=8)
    s = torch.randn(2, 8)
    h = torch.randn(2, 8)
    a1, _ = wm.step(s, hidden=h)
    a2, _ = wm.step(s, hidden=h)
    assert torch.allclose(a1, a2)


# ----------------------------------------------------------------------
# rollout()
# ----------------------------------------------------------------------

def test_rollout_unconditioned_requires_steps() -> None:
    wm = _wm(state_dim=8)
    with pytest.raises(ValueError, match="steps is required"):
        wm.rollout(torch.zeros(1, 8))


def test_rollout_unconditioned_shape() -> None:
    wm = _wm(state_dim=8)
    s0 = torch.randn(2, 8)
    traj = wm.rollout(s0, steps=5)
    assert traj.shape == (2, 5, 8)


def test_rollout_conditioned_requires_actions() -> None:
    wm = _wm(state_dim=8, action_dim=4)
    with pytest.raises(ValueError, match="actions are required"):
        wm.rollout(torch.zeros(1, 8), steps=3)


def test_rollout_conditioned_shape_matches_action_horizon() -> None:
    wm = _wm(state_dim=8, action_dim=4)
    s0 = torch.randn(2, 8)
    actions = torch.randn(2, 6, 4)
    traj = wm.rollout(s0, actions=actions)
    assert traj.shape == (2, 6, 8)


def test_rollout_first_step_matches_explicit_step_call() -> None:
    wm = _wm(state_dim=8, action_dim=4)
    s0 = torch.randn(1, 8)
    a = torch.randn(1, 4)
    traj = wm.rollout(s0, actions=a.unsqueeze(1))
    next_s, _ = wm.step(s0, action=a)
    assert torch.allclose(traj[:, 0, :], next_s)


def test_rollout_states_carry_through_time() -> None:
    """Different rollout steps produce different states (dynamics aren't trivially identity)."""
    wm = _wm(state_dim=8)
    s0 = torch.randn(1, 8)
    traj = wm.rollout(s0, steps=4)
    # Adjacent steps should differ — a GRU with random init evolves the state.
    assert not torch.allclose(traj[:, 0, :], traj[:, 1, :])


# ----------------------------------------------------------------------
# Backprop
# ----------------------------------------------------------------------

def test_gradients_flow_to_gru_and_out_proj() -> None:
    wm = _wm(state_dim=8, action_dim=4)
    s0 = torch.randn(1, 8, requires_grad=True)
    actions = torch.randn(1, 3, 4, requires_grad=True)
    traj = wm.rollout(s0, actions=actions)
    traj.pow(2).mean().backward()
    for p in wm.parameters():
        assert p.grad is not None
        # At least some parameters must accumulate non-zero gradient.
    grad_sum = sum(p.grad.abs().sum().item() for p in wm.parameters())
    assert grad_sum > 0
