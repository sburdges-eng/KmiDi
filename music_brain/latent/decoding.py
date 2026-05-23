"""Constraint-aware token sampling for the incremental decoder.

Closes the **Constraint-based decoding**, **Top-k / top-p constrained
decoding**, **Greedy low-jitter decoding**, and **Constraint-aware
generation** spec items.

Two public surfaces:
  - ``greedy_argmax(logits, *, forbidden_tokens)`` — the zero-jitter
    deterministic path. Used by the realtime audio loop where any
    sampling-induced output divergence is unacceptable.
  - ``sample_with_constraints(logits, cfg, *, forbidden_tokens, generator)``
    — the configurable path supporting temperature, top-k, top-p and
    forbidden-token masking. Pass a ``torch.Generator`` for
    reproducibility.

Both honor a forbidden-tokens list that the caller assembles from the
``IntentConstraints.forbidden_engines_mask`` field or domain logic
(e.g., the chord predictor refuses to emit a chord outside the active
mode). Forbidden tokens are masked *before* the softmax so the
distribution renormalizes over the remaining vocabulary; this is the
behavior every modern HF/llama.cpp sampler implements and matches
caller expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


def _torch():
    import torch
    return torch


NEG_INF = float("-inf")


@dataclass(frozen=True)
class DecodeConfig:
    """Sampling configuration.

    Args:
        temperature: softmax temperature. Must be > 0.
        top_k: keep only the top-k logits; 0 disables.
        top_p: nucleus threshold in (0, 1]; 1.0 disables.
    """
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if not (self.temperature > 0.0):
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


def _mask_forbidden(logits, forbidden_tokens: Optional[Iterable[int]]):
    if not forbidden_tokens:
        return logits
    torch = _torch()
    idx = torch.as_tensor(list(forbidden_tokens), dtype=torch.long,
                          device=logits.device)
    out = logits.clone()
    out[..., idx] = NEG_INF
    return out


def greedy_argmax(logits, *, forbidden_tokens: Optional[Iterable[int]] = None):
    """Return the deterministic argmax token id.

    Zero allocations on the hot path beyond the forbidden-mask clone
    (which is a no-op when no constraints are set).
    """
    masked = _mask_forbidden(logits, forbidden_tokens)
    return masked.argmax(dim=-1)


def sample_with_constraints(logits, cfg: DecodeConfig, *,
                            forbidden_tokens: Optional[Iterable[int]] = None,
                            generator=None):
    """Sample one token per row from ``logits`` under the given constraints.

    Args:
        logits: ``(B, V)`` raw logits.
        cfg: decoder configuration.
        forbidden_tokens: tokens to mask before sampling.
        generator: ``torch.Generator`` for reproducible sampling. When
            ``None`` the default RNG is used.

    Returns:
        ``(B,)`` long tensor of sampled token ids.
    """
    torch = _torch()
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D (B, V); got shape {tuple(logits.shape)}")
    masked = _mask_forbidden(logits, forbidden_tokens)
    if torch.all(masked == NEG_INF):
        raise ValueError("all tokens forbidden — vocabulary mask is empty")

    # Cheap greedy path when temperature is effectively zero. Avoids
    # NaN from dividing by ~0 and is the canonical low-jitter shortcut.
    if cfg.temperature < 1e-4:
        return masked.argmax(dim=-1)

    scaled = masked / cfg.temperature

    # top-k: keep only the top-k entries per row, mask the rest.
    if cfg.top_k > 0:
        k = min(cfg.top_k, scaled.shape[-1])
        topk_vals, _ = torch.topk(scaled, k=k, dim=-1)
        threshold = topk_vals[..., -1:].clone()
        scaled = torch.where(scaled < threshold,
                             torch.full_like(scaled, NEG_INF),
                             scaled)

    # top-p: keep the smallest set of tokens whose cumulative softmax
    # mass first crosses the threshold.
    if cfg.top_p < 1.0:
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Keep tokens where the *prior* cumulative mass is below top_p,
        # plus the boundary token. Equivalent to: keep while cumulative
        # - prob_i < top_p.
        keep_sorted = (cumulative - sorted_probs) < cfg.top_p
        # Always keep at least the top-1.
        keep_sorted[..., 0] = True
        # Scatter back to original index space.
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(-1, sorted_idx, keep_sorted)
        scaled = torch.where(keep, scaled, torch.full_like(scaled, NEG_INF))

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


__all__ = ["DecodeConfig", "greedy_argmax", "sample_with_constraints"]
