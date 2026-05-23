"""Tests for ``music_brain.latent.section_planner``.

Closes **Bar/beat structural awareness**, **Section-aware generation**,
**Phrase-boundary prediction**, **Structural phrase planning**,
**Hierarchical generation architecture** (top-down section → bar →
beat), and **Intent-level planning**.
"""

from __future__ import annotations

import pytest

from music_brain.latent.section_planner import (  # noqa: E402
    BarPosition,
    Section,
    SectionPlan,
    SectionPlanner,
    SectionRole,
)


def test_bar_position_is_a_frozen_dataclass() -> None:
    bp = BarPosition(bar=2, beat=3, beats_per_bar=4)
    with pytest.raises(Exception):
        bp.bar = 5  # type: ignore[misc]


def test_bar_position_validates_beat_range() -> None:
    with pytest.raises(ValueError, match="beat"):
        BarPosition(bar=0, beat=4, beats_per_bar=4)
    with pytest.raises(ValueError, match="beat"):
        BarPosition(bar=0, beat=-1, beats_per_bar=4)


def test_section_default_phrase_boundary_is_on_section_change() -> None:
    plan = SectionPlanner(beats_per_bar=4).plan_default()
    # plan_default uses a standard verse-chorus arrangement (16 bars).
    assert plan.total_bars == 16
    # Every section start is a phrase boundary.
    boundary_bars = plan.phrase_boundary_bars()
    assert all(b in boundary_bars for b in [0, 4, 8, 12])


def test_section_lookup_by_bar_index() -> None:
    plan = SectionPlanner(beats_per_bar=4).plan_default()
    assert plan.section_at_bar(0).role == SectionRole.INTRO
    assert plan.section_at_bar(8).role == SectionRole.CHORUS


def test_section_lookup_out_of_range_returns_none() -> None:
    plan = SectionPlanner(beats_per_bar=4).plan_default()
    assert plan.section_at_bar(100) is None


def test_phrase_boundary_within_section_at_half() -> None:
    """An 8-bar section emits a soft phrase boundary at bar 4."""
    plan = SectionPlan(
        sections=(Section(role=SectionRole.VERSE, start_bar=0, length_bars=8),),
        beats_per_bar=4,
    )
    bounds = plan.phrase_boundary_bars(emit_internal=True)
    assert 0 in bounds  # section start
    assert 4 in bounds  # mid-section soft boundary


def test_phrase_boundary_off_by_default() -> None:
    plan = SectionPlan(
        sections=(Section(role=SectionRole.VERSE, start_bar=0, length_bars=8),),
        beats_per_bar=4,
    )
    assert plan.phrase_boundary_bars() == (0,)


def test_section_plan_rejects_overlapping_sections() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SectionPlan(
            sections=(
                Section(role=SectionRole.VERSE, start_bar=0, length_bars=4),
                Section(role=SectionRole.CHORUS, start_bar=2, length_bars=4),
            ),
            beats_per_bar=4,
        )


def test_section_plan_rejects_gap_between_sections() -> None:
    with pytest.raises(ValueError, match="gap"):
        SectionPlan(
            sections=(
                Section(role=SectionRole.VERSE, start_bar=0, length_bars=4),
                Section(role=SectionRole.CHORUS, start_bar=5, length_bars=4),
            ),
            beats_per_bar=4,
        )


def test_total_bars_sums_section_lengths() -> None:
    plan = SectionPlan(
        sections=(
            Section(role=SectionRole.VERSE, start_bar=0, length_bars=4),
            Section(role=SectionRole.CHORUS, start_bar=4, length_bars=8),
        ),
        beats_per_bar=4,
    )
    assert plan.total_bars == 12


def test_bar_to_section_boundary_predicts_next_boundary() -> None:
    plan = SectionPlanner(beats_per_bar=4).plan_default()
    # At bar 3 we are 1 bar away from the next section start at bar 4.
    assert plan.bars_until_next_boundary(bar=3) == 1
    # At a section boundary itself, distance is 0.
    assert plan.bars_until_next_boundary(bar=4) == 0
    # Past the end → -1 sentinel.
    assert plan.bars_until_next_boundary(bar=plan.total_bars) == -1


def test_position_step_advances_beat() -> None:
    bp = BarPosition(bar=0, beat=0, beats_per_bar=4)
    advanced = bp.step()
    assert advanced.bar == 0 and advanced.beat == 1
    last = BarPosition(bar=0, beat=3, beats_per_bar=4).step()
    assert last.bar == 1 and last.beat == 0


def test_section_role_enum_is_stable() -> None:
    assert SectionRole.INTRO.value == "intro"
    assert SectionRole.CHORUS.value == "chorus"
