"""LatencyBudget primitive (Goal: Dynamic latency budgeting,
Sub-16ms inference targets, Low-jitter scheduling)."""

import time

import pytest

from music_brain.generation.latency_budget import BudgetExceeded, LatencyBudget

# ---------------------------------------------------------------------------
# Basic timing
# ---------------------------------------------------------------------------


def test_budget_remaining_decreases_as_time_passes():
    budget = LatencyBudget(total_ms=50.0)
    initial = budget.remaining_ms
    time.sleep(0.005)
    later = budget.remaining_ms
    assert later < initial
    assert later > 0


def test_budget_elapsed_ms_increases_monotonically():
    budget = LatencyBudget(total_ms=100.0)
    samples = []
    for _ in range(3):
        samples.append(budget.elapsed_ms)
        time.sleep(0.002)
    assert samples == sorted(samples)
    assert samples[-1] > samples[0]


# ---------------------------------------------------------------------------
# has_budget
# ---------------------------------------------------------------------------


def test_has_budget_true_when_required_within_remaining():
    budget = LatencyBudget(total_ms=50.0)
    assert budget.has_budget(required_ms=10.0) is True


def test_has_budget_false_when_required_exceeds_remaining():
    budget = LatencyBudget(total_ms=5.0)
    assert budget.has_budget(required_ms=100.0) is False


def test_has_budget_uses_consumed_budget():
    """Manually consumed budget reduces what's available for has_budget."""
    budget = LatencyBudget(total_ms=100.0)
    budget.consume_ms(80.0)
    # Wall clock barely advanced; consumed reduces remaining further.
    assert budget.has_budget(required_ms=30.0) is False
    assert budget.has_budget(required_ms=10.0) is True


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------


def test_consume_ms_reduces_remaining():
    budget = LatencyBudget(total_ms=100.0)
    before = budget.remaining_ms
    budget.consume_ms(20.0)
    after = budget.remaining_ms
    assert before - after >= 19.0  # 20 minus wall-clock noise


def test_consume_or_raise_passes_when_within_budget():
    budget = LatencyBudget(total_ms=100.0)
    budget.consume_or_raise(20.0)
    assert budget.remaining_ms > 0


def test_consume_or_raise_raises_when_over_budget():
    budget = LatencyBudget(total_ms=10.0)
    with pytest.raises(BudgetExceeded):
        budget.consume_or_raise(50.0)


# ---------------------------------------------------------------------------
# is_exhausted
# ---------------------------------------------------------------------------


def test_is_exhausted_false_initially():
    budget = LatencyBudget(total_ms=100.0)
    assert budget.is_exhausted is False


def test_is_exhausted_true_after_consuming_full_budget():
    budget = LatencyBudget(total_ms=100.0)
    budget.consume_ms(100.0)
    assert budget.is_exhausted is True


def test_is_exhausted_true_after_wall_clock_drains_budget():
    budget = LatencyBudget(total_ms=5.0)
    time.sleep(0.01)
    assert budget.is_exhausted is True


# ---------------------------------------------------------------------------
# fraction_used (for adaptive scheduling)
# ---------------------------------------------------------------------------


def test_fraction_used_starts_near_zero():
    budget = LatencyBudget(total_ms=100.0)
    assert budget.fraction_used == pytest.approx(0.0, abs=0.01)


def test_fraction_used_one_when_exhausted():
    budget = LatencyBudget(total_ms=10.0)
    budget.consume_ms(10.0)
    assert budget.fraction_used >= 1.0


def test_fraction_used_clamps_to_one_when_over_budget():
    budget = LatencyBudget(total_ms=10.0)
    budget.consume_ms(50.0)
    assert budget.fraction_used == 1.0
