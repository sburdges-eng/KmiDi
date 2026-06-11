"""LatencyBudget — wall-clock + manual-consumption time tracking for
realtime-ish generation paths.

Pairs a monotonic wall clock with explicit ``consume_ms()`` calls so
generators can answer two questions cheaply:

  - "Do I have time to run this stage?" → :meth:`has_budget`
  - "Am I out of time?" → :meth:`is_exhausted` / :attr:`remaining_ms`

The class does *not* enforce real-time deadlines on its own — it's a
bookkeeping primitive. Callers decide what to skip, retry, or fall back
to when ``has_budget`` says no.

Goal-list ties: Dynamic latency budgeting (direct),
Sub-16ms inference targets (callers can gate stages on the remaining
budget), Low-jitter scheduling (deterministic decision points instead of
arbitrary timeouts).

Stdlib only.
"""

from __future__ import annotations

import time


class BudgetExceeded(Exception):
    """Raised by :meth:`LatencyBudget.consume_or_raise` when a step would
    push consumed-plus-elapsed past the total budget."""


class LatencyBudget:
    """Track remaining budget = ``total - (elapsed + consumed)``.

    ``elapsed_ms`` is measured against a monotonic clock at construction;
    ``consumed_ms`` is incremented by explicit :meth:`consume_ms` calls so
    callers can amortise budget across stages that don't run end-to-end
    against the wall clock.
    """

    def __init__(self, total_ms: float) -> None:
        if total_ms <= 0:
            raise ValueError("total_ms must be > 0")
        self._total_ms = float(total_ms)
        self._start = time.monotonic()
        self._consumed_ms = 0.0

    @property
    def total_ms(self) -> float:
        return self._total_ms

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0

    @property
    def consumed_ms(self) -> float:
        return self._consumed_ms

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self._total_ms - self.elapsed_ms - self._consumed_ms)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining_ms <= 0.0

    @property
    def fraction_used(self) -> float:
        """``(elapsed + consumed) / total``, clamped to ``[0, 1]``."""
        used = (self.elapsed_ms + self._consumed_ms) / self._total_ms
        return min(1.0, max(0.0, used))

    def has_budget(self, required_ms: float) -> bool:
        """True iff a step claiming ``required_ms`` would still fit."""
        return required_ms <= self.remaining_ms

    def consume_ms(self, ms: float) -> None:
        """Add ``ms`` to the manually consumed total."""
        if ms < 0:
            raise ValueError("ms must be >= 0")
        self._consumed_ms += float(ms)

    def consume_or_raise(self, ms: float) -> None:
        """Consume ``ms`` only if it fits; otherwise raise :class:`BudgetExceeded`."""
        if not self.has_budget(ms):
            raise BudgetExceeded(
                f"requested {ms:.2f}ms but only {self.remaining_ms:.2f}ms left "
                f"of {self._total_ms:.2f}ms budget"
            )
        self.consume_ms(ms)
