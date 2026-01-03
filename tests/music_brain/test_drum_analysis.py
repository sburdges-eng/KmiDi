from dataclasses import dataclass

import pytest

from music_brain.groove.drum_analysis import AnalysisConfig, DrumAnalyzer
from music_brain.groove.drum_humanizer import DrumHumanizer


@dataclass
class DummyNote:
    onset_ticks: int
    velocity: int
    pitch: int
    channel: int = 9


@pytest.fixture
def analyzer() -> DrumAnalyzer:
    return DrumAnalyzer()


def test_returns_empty_profile_when_no_drum_notes(analyzer: DrumAnalyzer) -> None:
    notes = [DummyNote(onset_ticks=0, velocity=90, pitch=60, channel=0)]
    profile = analyzer.analyze(notes)
    assert profile.snare.flam_count == 0
    assert profile.hihat.is_alternating is False
    assert profile.ghost_note_density == 0.0


def test_detects_flam_cluster(analyzer: DrumAnalyzer) -> None:
    # Two close snare hits should be tagged as a flam.
    notes = [
        DummyNote(onset_ticks=0, velocity=40, pitch=38),
        DummyNote(onset_ticks=10, velocity=90, pitch=38),
    ]
    profile = analyzer.analyze(notes)
    assert profile.snare.flam_count == 1
    assert profile.snare.has_buzz_rolls is False


def test_detects_buzz_roll(analyzer: DrumAnalyzer) -> None:
    notes = [
        DummyNote(onset_ticks=0, velocity=70, pitch=38),
        DummyNote(onset_ticks=8, velocity=72, pitch=38),
        DummyNote(onset_ticks=16, velocity=75, pitch=38),
    ]
    profile = analyzer.analyze(notes)
    assert profile.snare.has_buzz_rolls is True
    assert len(profile.snare.buzz_roll_regions) == 1


def test_detects_hihat_alternation(analyzer: DrumAnalyzer) -> None:
    # Alternating velocities on an 8th grid; should register alternation.
    notes = [
        DummyNote(onset_ticks=i * 240, velocity=80 if i % 2 == 0 else 70, pitch=42)
        for i in range(8)
    ]
    profile = analyzer.analyze(notes)
    assert profile.hihat.is_alternating is True
    assert profile.hihat.dominant_hand in {"right", "unknown"}


def test_counts_ghost_notes(analyzer: DrumAnalyzer) -> None:
    notes = [
        DummyNote(onset_ticks=i * 120, velocity=30 if i % 2 == 0 else 90, pitch=38)
        for i in range(10)
    ]
    profile = analyzer.analyze(notes)
    assert profile.ghost_note_density > 0.3


def test_humanizer_derives_style_from_analysis() -> None:
    notes = [
        DummyNote(onset_ticks=0, velocity=70, pitch=38),
        DummyNote(onset_ticks=8, velocity=72, pitch=38),
        DummyNote(onset_ticks=16, velocity=75, pitch=38),
    ]
    humanizer = DrumHumanizer()
    plan = humanizer.to_plan(technique_profile=None, notes=notes)
    assert plan["style"] == "technical"
    assert plan["techniques"]


def test_configurable_thresholds_enable_drag_detection() -> None:
    # Looser buzz/drag window should let wider clusters be treated as drags.
    config = AnalysisConfig(buzz_threshold_ms=90.0, drag_threshold_ms=120.0)
    analyzer = DrumAnalyzer(config=config)
    notes = [
        DummyNote(onset_ticks=0, velocity=30, pitch=38),
        DummyNote(onset_ticks=30, velocity=35, pitch=38),
        DummyNote(onset_ticks=60, velocity=95, pitch=38),
    ]
    profile = analyzer.analyze(notes)
    assert profile.snare.drag_count >= 1
    assert profile.snare.primary_technique in {"jazzy", "technical", "standard"}


def test_apply_guide_rules_calls_groove_engine() -> None:
    events = [
        {"start_tick": 0, "velocity": 90, "pitch": 38, "channel": 9},
        {"start_tick": 120, "velocity": 85, "pitch": 42, "channel": 9},
    ]
    humanizer = DrumHumanizer()
    out = humanizer.apply_guide_rules(
        midi=events,
        notes=[DummyNote(onset_ticks=0, velocity=90, pitch=38)],
        seed=123,
    )
    assert isinstance(out, list)
    assert len(out) >= len(events)
    for ev in out:
        assert "velocity" in ev and "start_tick" in ev
