"""Latent world-model predictor (next-state forecaster).

Closes the **World-model latent prediction** spec item documented as a
gap in ``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``. A real
implementation, not a stub — small GRU-based autoregressive predictor
that takes the current latent state plus an optional action embedding
and produces the predicted next latent. Future work extends this with
larger architectures; the contract is what's important here.

Why this contract:
The KMiDi platform spec calls for "predictive arrangement modeling":
the system should plan a few bars ahead in latent space before
realising audio. That means rolling a learned dynamics model forward
in latent space — exactly what a world-model does in classical RL /
dreamer-style approaches.

Design:
- ``WorldModel.step(state, action)`` advances one tick: returns the
  predicted next state.
- ``WorldModel.rollout(state0, actions)`` runs an autoregressive loop
  for N steps and returns the full trajectory.
- GRU cell internals — small enough to fit anywhere, large enough to
  learn non-trivial dynamics. Easy to swap for a larger architecture
  later (the public step/rollout API is stable).
- ``state_dim`` and ``action_dim`` are independent — action embedding
  size doesn't have to match latent state size.

Stdlib + torch. Tests skip cleanly without torch.
"""

from __future__ import annotations

from typing import Optional


def _torch():
    import torch
    return torch


class WorldModel:
    """Autoregressive next-state predictor in latent space.

    Args:
        state_dim: dimension of the latent state.
        action_dim: dimension of the action / conditioning vector.
            Pass ``0`` for an unconditional world-model.
        hidden_dim: GRU hidden size. Defaults to ``state_dim``.
    """

    def __init__(self, state_dim: int, action_dim: int = 0,
                 hidden_dim: Optional[int] = None) -> None:
        torch = _torch()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if action_dim < 0:
            raise ValueError("action_dim must be >= 0")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.state_dim

        gru_in = self.state_dim + self.action_dim
        self.gru = torch.nn.GRUCell(gru_in, self.hidden_dim)
        self.out_proj = torch.nn.Linear(self.hidden_dim, self.state_dim)

    # ------------------------------------------------------------------
    # Forward API
    # ------------------------------------------------------------------

    def step(self, state, action=None, hidden=None):
        """Advance one tick.

        Args:
            state: (B, state_dim) current latent state.
            action: (B, action_dim) optional action embedding. Required
                when action_dim > 0.
            hidden: (B, hidden_dim) optional carried GRU hidden state.
                If None, zero-initialised on first call.

        Returns:
            (next_state, new_hidden)
                next_state: (B, state_dim)
                new_hidden: (B, hidden_dim)
        """
        torch = _torch()
        b = state.shape[0]
        if self.action_dim > 0:
            if action is None:
                raise ValueError("action is required when action_dim > 0")
            if action.shape[0] != b:
                raise ValueError("batch dim of action must match state")
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        if hidden is None:
            hidden = torch.zeros(b, self.hidden_dim, device=state.device,
                                 dtype=state.dtype)
        new_hidden = self.gru(x, hidden)
        next_state = self.out_proj(new_hidden)
        return next_state, new_hidden

    def rollout(self, state0, actions=None, steps: Optional[int] = None):
        """Autoregressive multi-step rollout from ``state0``.

        Args:
            state0: (B, state_dim) initial latent state.
            actions: (B, T, action_dim) per-step actions when action_dim
                > 0. ``steps`` is inferred as ``T`` if not given.
            steps: explicit number of steps when action_dim == 0.

        Returns:
            (B, T, state_dim) predicted-trajectory tensor (excludes the
            initial state).
        """
        torch = _torch()
        if self.action_dim > 0:
            if actions is None:
                raise ValueError("actions are required for a conditioned world-model")
            t = actions.shape[1]
        else:
            if steps is None:
                raise ValueError("steps is required for an unconditioned world-model")
            t = int(steps)

        state = state0
        hidden = None
        out = []
        for i in range(t):
            a = actions[:, i, :] if self.action_dim > 0 else None
            state, hidden = self.step(state, action=a, hidden=hidden)
            out.append(state)
        return torch.stack(out, dim=1)

    def __call__(self, state, action=None, hidden=None):
        return self.step(state, action=action, hidden=hidden)

    # ------------------------------------------------------------------
    # Parameters / housekeeping
    # ------------------------------------------------------------------

    def parameters(self):
        yield from self.gru.parameters()
        yield from self.out_proj.parameters()


__all__ = ["WorldModel"]
