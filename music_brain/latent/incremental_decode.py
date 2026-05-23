"""Incremental decode loop composing scheduler + sampler + KV-cache.

Closes the **Incremental decoding pipeline** spec item. This is the
realtime entrypoint for token-level generation: the caller hands in a
"logit function" that, given the current step index and the previous
token, returns the next-step logits — and the loop:

  1. Asks the (optional) ``JitterBoundedScheduler`` for the step
     budget. If headroom is thin, it switches to ``greedy_argmax`` for
     zero-jitter operation.
  2. Otherwise samples via ``sample_with_constraints`` with the
     supplied ``DecodeConfig`` and ``forbidden_tokens``.
  3. Yields a ``DecodeStep`` per emitted token.
  4. Stops on ``max_steps`` or when ``eos_token`` is produced.

The model owns its KV-cache (passed in by side-channel through the
``logit_fn`` closure). The loop has no opinion on the model
architecture — it just enforces the realtime decode contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from music_brain.latent.decoding import (
    DecodeConfig,
    greedy_argmax,
    sample_with_constraints,
)
from music_brain.latent.scheduler import JitterBoundedScheduler


LogitFn = Callable[[int, Optional[int]], "object"]
# `LogitFn(step_idx, previous_token) -> logits: torch.Tensor of shape (1, V)`


@dataclass(frozen=True)
class DecodeStep:
    step_index: int
    token_id: int
    mode: str  # "greedy" | "sampled"
    elapsed_ms: float


def incremental_decode(
    logit_fn: LogitFn,
    *,
    max_steps: int,
    cfg: Optional[DecodeConfig] = None,
    forbidden_tokens: Optional[list[int]] = None,
    eos_token: Optional[int] = None,
    scheduler: Optional[JitterBoundedScheduler] = None,
    generator=None,
) -> Iterator[DecodeStep]:
    """Run the realtime decode loop, yielding one token per step."""
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    cfg = cfg or DecodeConfig()
    previous_token: Optional[int] = None

    for step_idx in range(max_steps):
        decision = scheduler.next_step() if scheduler is not None else None
        force_greedy = decision is not None and decision.recommend_greedy

        t0 = time.perf_counter()
        logits = logit_fn(step_idx, previous_token)
        if force_greedy:
            token = int(greedy_argmax(
                logits, forbidden_tokens=forbidden_tokens).item())
            mode = "greedy"
        else:
            token = int(sample_with_constraints(
                logits, cfg, forbidden_tokens=forbidden_tokens,
                generator=generator).item())
            mode = "sampled"
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if scheduler is not None:
            scheduler.record_step_ms(dt_ms)

        yield DecodeStep(step_index=step_idx, token_id=token, mode=mode,
                         elapsed_ms=dt_ms)
        previous_token = token
        if eos_token is not None and token == eos_token:
            return


__all__ = ["DecodeStep", "incremental_decode"]
