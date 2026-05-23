"""Tests for ``music_brain.latent.scheduler.JitterBoundedScheduler``.

Closes **Low-jitter scheduling**, **Dynamic latency budgeting**, and
**Sub-16ms inference targets**. The scheduler tracks a rolling window
of step latencies, computes a budget headroom, and tells the decoder
whether the next step should run at full quality or fall back to a
cheaper greedy path.
"""

from __future__ import annotations

import pytest

from music_brain.latent.scheduler import (  # noqa: E402
    JitterBoundedScheduler,
    StepDecision,
)


def test_default_budget_is_16ms() -> None:
    s = JitterBoundedScheduler()
    assert s.budget_ms == pytest.approx(16.0)


def test_records_step_and_advances_history() -> None:
    s = JitterBoundedScheduler(budget_ms=16.0, window=4)
    s.record_step_ms(5.0)
    s.record_step_ms(7.0)
    assert s.history_len == 2
    assert s.mean_ms == pytest.approx(6.0)


def test_history_window_is_bounded() -> None:
    s = JitterBoundedScheduler(window=3)
    for v in [10.0, 20.0, 30.0, 40.0]:
        s.record_step_ms(v)
    # Oldest sample (10) has rolled off.
    assert s.history_len == 3
    assert s.mean_ms == pytest.approx((20 + 30 + 40) / 3)


def test_within_budget_when_mean_under_budget() -> None:
    s = JitterBoundedScheduler(budget_ms=16.0)
    s.record_step_ms(5.0)
    s.record_step_ms(6.0)
    d = s.next_step()
    assert d.within_budget is True
    assert d.recommend_greedy is False
    assert d.headroom_ms == pytest.approx(16.0 - 5.5, rel=1e-3)


def test_recommends_greedy_when_headroom_thin() -> None:
    s = JitterBoundedScheduler(budget_ms=16.0, greedy_threshold_ms=4.0)
    s.record_step_ms(13.0)
    s.record_step_ms(14.0)
    d = s.next_step()
    # Mean is 13.5; headroom is 2.5 — below the 4ms threshold.
    assert d.headroom_ms < 4.0
    assert d.recommend_greedy is True


def test_reports_budget_violation_when_jitter_spikes() -> None:
    s = JitterBoundedScheduler(budget_ms=16.0, p95_factor=1.0)
    s.record_step_ms(20.0)
    s.record_step_ms(22.0)
    d = s.next_step()
    assert d.within_budget is False
    assert d.recommend_greedy is True


def test_empty_history_is_optimistic() -> None:
    """A cold scheduler with no history reports full headroom."""
    s = JitterBoundedScheduler(budget_ms=16.0)
    d = s.next_step()
    assert d.within_budget is True
    assert d.headroom_ms == pytest.approx(16.0)


def test_reset_clears_history() -> None:
    s = JitterBoundedScheduler()
    s.record_step_ms(10.0)
    s.record_step_ms(11.0)
    s.reset()
    assert s.history_len == 0


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="budget_ms"):
        JitterBoundedScheduler(budget_ms=0.0)
    with pytest.raises(ValueError, match="window"):
        JitterBoundedScheduler(window=0)
    with pytest.raises(ValueError, match="greedy_threshold"):
        JitterBoundedScheduler(greedy_threshold_ms=-1.0)


def test_step_decision_is_a_dataclass() -> None:
    s = JitterBoundedScheduler()
    d = s.next_step()
    assert isinstance(d, StepDecision)
    # Fields are accessible by name.
    _ = d.within_budget, d.recommend_greedy, d.headroom_ms, d.observed_p95_ms
