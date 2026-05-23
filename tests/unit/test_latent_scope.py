"""Tests for ``music_brain.latent.scope`` — generation scope isolation.

Closes **Generation scope isolation** and **Adaptive state-aware
delivery** (scope is the contract the orchestrator uses to deliver
adaptive results without touching state outside the boundary).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402
from music_brain.latent.scope import (  # noqa: E402
    GenerationScope,
    ScopeViolation,
)


def _prov() -> IntentProvenance:
    return IntentProvenance(source=IntentSource.UI_DIRECT,
                            user_override_weight=1.0)


def _frame(t: int) -> LatentFrame:
    return LatentFrame(
        audio_z=torch.zeros(1, 4, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.0, 0.0),
        time_index=t,
        provenance=_prov(),
        metadata={},
    )


def test_scope_validates_bar_range_on_construction() -> None:
    with pytest.raises(ValueError, match="bar"):
        GenerationScope(start_bar=8, end_bar=4)
    with pytest.raises(ValueError, match="bar"):
        GenerationScope(start_bar=-1, end_bar=4)


def test_scope_contains_frame_inside_range() -> None:
    scope = GenerationScope(start_bar=4, end_bar=8)
    assert scope.contains_bar(4)
    assert scope.contains_bar(7)
    assert not scope.contains_bar(8)  # end_bar is exclusive
    assert not scope.contains_bar(3)


def test_collect_filters_out_of_scope_frames() -> None:
    scope = GenerationScope(start_bar=2, end_bar=5)
    frames = [_frame(t) for t in range(8)]
    collected = scope.collect(frames, bar_of=lambda f: f.time_index)
    assert [f.time_index for f in collected] == [2, 3, 4]


def test_enforce_raises_on_out_of_scope_write() -> None:
    scope = GenerationScope(start_bar=2, end_bar=5)
    with pytest.raises(ScopeViolation):
        scope.enforce_bar(7)


def test_enforce_passes_for_in_scope_writes() -> None:
    scope = GenerationScope(start_bar=2, end_bar=5)
    scope.enforce_bar(2)
    scope.enforce_bar(4)  # no exception


def test_overlap_returns_intersection() -> None:
    a = GenerationScope(start_bar=2, end_bar=6)
    b = GenerationScope(start_bar=4, end_bar=10)
    out = a.overlap(b)
    assert out is not None
    assert out.start_bar == 4
    assert out.end_bar == 6


def test_overlap_returns_none_when_disjoint() -> None:
    a = GenerationScope(start_bar=0, end_bar=2)
    b = GenerationScope(start_bar=4, end_bar=6)
    assert a.overlap(b) is None


def test_length_bars() -> None:
    scope = GenerationScope(start_bar=3, end_bar=11)
    assert scope.length_bars == 8


def test_scope_is_frozen() -> None:
    scope = GenerationScope(start_bar=0, end_bar=4)
    with pytest.raises(Exception):
        scope.start_bar = 5  # type: ignore[misc]


def test_intent_id_propagates_into_violation() -> None:
    scope = GenerationScope(start_bar=2, end_bar=5, intent_id=42)
    try:
        scope.enforce_bar(7)
    except ScopeViolation as exc:
        assert exc.intent_id == 42
        assert exc.attempted_bar == 7
