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

    def __init__(
        self, state_dim: int, action_dim: int = 0, hidden_dim: Optional[int] = None
    ) -> None:
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
            hidden = torch.zeros(b, self.hidden_dim, device=state.device, dtype=state.dtype)
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
    # LatentFrame adapter
    # ------------------------------------------------------------------

    def rollout_frames(self, initial, actions=None, steps: Optional[int] = None):
        """Roll out a trajectory of ``LatentFrame``s from an initial frame.

        Wraps ``rollout`` and stamps each step's ``time_index`` /
        propagates ``provenance`` + ``emotion_va`` so consumers can
        feed the trajectory directly into ``music_brain.latent.streaming``.

        Args:
            initial: ``LatentFrame`` whose ``audio_z`` (T_in, D)
                supplies the initial state. Mean-pooled across the
                time axis to a single (D,) latent vector.
            actions: (T, action_dim) per-step actions when action_dim > 0.
            steps: explicit step count when action_dim == 0.

        Returns:
            list[LatentFrame] of length T, each with ``audio_z`` shape
            (1, D) and ``time_index`` = initial.time_index + 1 + i.
        """
        # Imported lazily to avoid a top-level cycle: the latent module
        # imports IntentProvenance from intent_ir, which in turn does
        # not depend on world_model.
        from music_brain.latent.latent_frame import LatentFrame  # noqa: PLC0415

        if initial.audio_feature_dim != self.state_dim:
            raise ValueError(
                f"LatentFrame audio_feature_dim {initial.audio_feature_dim} "
                f"!= WorldModel.state_dim {self.state_dim}"
            )
        # Pool the initial frame's time axis to a single state vector,
        # then add the batch dim WorldModel.rollout expects.
        state0 = initial.audio_z.mean(dim=0, keepdim=True)  # (1, D)
        if self.action_dim > 0:
            if actions is None:
                raise ValueError("actions are required when action_dim > 0")
            actions_b = actions.unsqueeze(0)  # (1, T, action_dim)
            traj = self.rollout(state0, actions=actions_b)  # (1, T, D)
        else:
            traj = self.rollout(state0, steps=steps)  # (1, T, D)
        out = []
        t = traj.shape[1]
        for i in range(t):
            step_z = traj[0, i : i + 1, :].contiguous()  # (1, D)
            out.append(
                LatentFrame(
                    audio_z=step_z,
                    chord_z=None,
                    emotion_va=initial.emotion_va,
                    time_index=initial.time_index + 1 + i,
                    provenance=initial.provenance,
                    metadata={"source": "world_model.rollout_frames", "step": i},
                )
            )
        return out

    # ------------------------------------------------------------------
    # Parameters / housekeeping
    # ------------------------------------------------------------------

    def parameters(self):
        yield from self.gru.parameters()
        yield from self.out_proj.parameters()


__all__ = ["WorldModel"]
