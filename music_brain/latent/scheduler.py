"""Jitter-bounded scheduler for the realtime decoder loop.

Closes **Low-jitter scheduling**, **Dynamic latency budgeting**, and
**Sub-16ms inference targets**. The scheduler is a small piece of
control logic — it does not run any model code itself. Each tick the
decoder loop:

  1. Asks the scheduler for a ``StepDecision``.
  2. Runs the next step (full-quality sampler, or the greedy fallback
     when ``recommend_greedy`` is set).
  3. Reports the observed step latency back via ``record_step_ms``.

The scheduler keeps a rolling window of recent step times and
computes:

  - ``mean_ms``: smoothed average of the window.
  - ``observed_p95_ms``: 95th-percentile latency proxy
    (``mean + p95_factor * stdev`` to avoid a sort on the hot path).
  - ``headroom_ms``: ``budget_ms - mean_ms``.

The decoder uses ``headroom_ms`` to switch decoding modes:

  - Plenty of headroom → sample with full constraints.
  - Headroom thin (< ``greedy_threshold_ms``) → fall back to
    greedy_argmax (zero-jitter, no multinomial / sort overhead).
  - Mean ms above budget → report ``within_budget = False`` so the
    caller can spawn a lighter model or drop a frame.

This is intentionally a *measurement-driven* policy. The decoder
caller owns the actual fallback action — the scheduler just surfaces
the signal.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StepDecision:
    within_budget: bool
    recommend_greedy: bool
    headroom_ms: float
    observed_p95_ms: float


class JitterBoundedScheduler:
    """Track step latency and surface a per-step decode-mode decision.

    Args:
        budget_ms: per-step latency budget. Default 16.0 — the sub-16ms
            target from the platform spec (one ~60Hz frame).
        window: rolling-window length over which latency stats are
            computed.
        greedy_threshold_ms: when ``headroom_ms`` falls below this,
            recommend the greedy fallback even if still within budget.
        p95_factor: multiplier on stdev for the p95 proxy. 1.65 is the
            normal-distribution 95% one-tail value.
    """

    def __init__(self, *, budget_ms: float = 16.0, window: int = 32,
                 greedy_threshold_ms: float = 2.0,
                 p95_factor: float = 1.65) -> None:
        if not (budget_ms > 0.0):
            raise ValueError(f"budget_ms must be > 0, got {budget_ms}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if greedy_threshold_ms < 0.0:
            raise ValueError(
                f"greedy_threshold_ms must be >= 0, got {greedy_threshold_ms}")
        self._budget = float(budget_ms)
        self._window: deque[float] = deque(maxlen=int(window))
        self._greedy_threshold = float(greedy_threshold_ms)
        self._p95_factor = float(p95_factor)

    # ------------------------------------------------------------------
    # Configuration introspection
    # ------------------------------------------------------------------

    @property
    def budget_ms(self) -> float:
        return self._budget

    @property
    def history_len(self) -> int:
        return len(self._window)

    @property
    def mean_ms(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def stdev_ms(self) -> float:
        if len(self._window) < 2:
            return 0.0
        m = self.mean_ms
        var = sum((x - m) ** 2 for x in self._window) / (len(self._window) - 1)
        return math.sqrt(var)

    @property
    def observed_p95_ms(self) -> float:
        return self.mean_ms + self._p95_factor * self.stdev_ms

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def record_step_ms(self, ms: float) -> None:
        if ms < 0.0:
            raise ValueError("step latency must be >= 0")
        self._window.append(float(ms))

    def next_step(self) -> StepDecision:
        # Cold start: optimistic — full headroom, full quality.
        if not self._window:
            return StepDecision(within_budget=True, recommend_greedy=False,
                                headroom_ms=self._budget,
                                observed_p95_ms=0.0)
        mean = self.mean_ms
        p95 = self.observed_p95_ms
        headroom = self._budget - mean
        within = p95 <= self._budget
        greedy = (not within) or (headroom < self._greedy_threshold)
        return StepDecision(within_budget=within,
                            recommend_greedy=greedy,
                            headroom_ms=headroom,
                            observed_p95_ms=p95)

    def reset(self) -> None:
        self._window.clear()


class StepTimer:
    """Context manager that auto-records to a scheduler on exit.

    Usage::

        with StepTimer(scheduler):
            run_decode_step()
    """

    def __init__(self, scheduler: JitterBoundedScheduler) -> None:
        self._scheduler = scheduler
        self._t0 = 0.0

    def __enter__(self) -> "StepTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._scheduler.record_step_ms(dt_ms)


__all__ = ["JitterBoundedScheduler", "StepDecision", "StepTimer"]
