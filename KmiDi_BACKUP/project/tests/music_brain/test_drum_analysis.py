"""
Unit tests for drum_analysis.py
"""

from dataclasses import dataclass

import pytest

from music_brain.groove.drum_analysis import (
    AnalysisConfig,
    DrumAnalyzer,
    DrumTechniqueProfile,
    SnareBounceSignature,
    HiHatAlternation,
    analyze_drum_technique,
)
from music_brain.groove.drum_humanizer import DrumHumanizer

DRUM_CHANNEL = 9
KICK_PITCH = 36
SNARE_PITCH = 38
HIHAT_PITCH = 42


@dataclass
class DummyNote:
    onset_ticks: int
    velocity: int
    pitch: int
    channel: int = DRUM_CHANNEL
    duration_ticks: int = 120


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
        DummyNote(onset_ticks=0, velocity=40, pitch=SNARE_PITCH),
        DummyNote(onset_ticks=10, velocity=90, pitch=SNARE_PITCH),
    ]
    profile = analyzer.analyze(notes)
    assert profile.snare.flam_count == 1
    assert profile.snare.has_buzz_rolls is False


def test_detects_buzz_roll(analyzer: DrumAnalyzer) -> None:
    notes = [
        DummyNote(onset_ticks=0, velocity=70, pitch=SNARE_PITCH),
        DummyNote(onset_ticks=8, velocity=72, pitch=SNARE_PITCH),
        DummyNote(onset_ticks=16, velocity=75, pitch=SNARE_PITCH),
    ]
    profile = analyzer.analyze(notes)
    assert profile.snare.has_buzz_rolls is True
    assert len(profile.snare.buzz_roll_regions) == 1


def test_detects_hihat_alternation(analyzer: DrumAnalyzer) -> None:
    # Alternating velocities on an 8th grid; should register alternation.
    notes = [
        DummyNote(onset_ticks=i * 240, velocity=80 if i %
                  2 == 0 else 70, pitch=HIHAT_PITCH)
        for i in range(8)
    ]
    profile = analyzer.analyze(notes)
    assert profile.hihat.is_alternating is True
    assert profile.hihat.dominant_hand in {"right", "unknown"}


def test_counts_ghost_notes(analyzer: DrumAnalyzer) -> None:
    notes = [
        DummyNote(onset_ticks=i * 120, velocity=30 if i %
                  2 == 0 else 90, pitch=SNARE_PITCH)
        for i in range(10)
    ]
    profile = analyzer.analyze(notes)
    assert profile.ghost_note_density > 0.3


def test_humanizer_derives_style_from_analysis() -> None:
    notes = [
        DummyNote(onset_ticks=0, velocity=70, pitch=SNARE_PITCH),
        DummyNote(onset_ticks=8, velocity=72, pitch=SNARE_PITCH),
        DummyNote(onset_ticks=16, velocity=75, pitch=SNARE_PITCH),
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
        DummyNote(onset_ticks=0, velocity=30, pitch=SNARE_PITCH),
        DummyNote(onset_ticks=30, velocity=35, pitch=SNARE_PITCH),
        DummyNote(onset_ticks=60, velocity=95, pitch=SNARE_PITCH),
    ]
    profile = analyzer.analyze(notes)
    assert profile.snare.drag_count >= 1
    assert profile.snare.primary_technique in {
        "jazzy", "technical", "standard"}


def test_apply_guide_rules_calls_groove_engine() -> None:
    events = [
        {"start_tick": 0, "velocity": 90, "pitch": SNARE_PITCH, "channel": DRUM_CHANNEL},
        {"start_tick": 120, "velocity": 85, "pitch": HIHAT_PITCH, "channel": DRUM_CHANNEL},
    ]
    humanizer = DrumHumanizer()
    out = humanizer.apply_guide_rules(
        midi=events,
        notes=[DummyNote(onset_ticks=0, velocity=90, pitch=SNARE_PITCH)],
        seed=123,
    )
    assert isinstance(out, list)
    assert len(out) >= len(events)
    for ev in out:
        assert "velocity" in ev and "start_tick" in ev


class TestDrumAnalyzer:
    """Test the DrumAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DrumAnalyzer(ppq=480, bpm=120.0)
        self.mock_notes = [
            DummyNote(0, 100, KICK_PITCH),       # Kick
            DummyNote(240, 90, SNARE_PITCH),     # Snare
            DummyNote(480, 95, KICK_PITCH),      # Kick
            DummyNote(720, 85, SNARE_PITCH),     # Snare
            DummyNote(0, 60, HIHAT_PITCH),       # Hi-hat (every 8th note)
            DummyNote(120, 55, HIHAT_PITCH),
            DummyNote(240, 65, HIHAT_PITCH),
            DummyNote(360, 50, HIHAT_PITCH),
            DummyNote(480, 70, HIHAT_PITCH),
            DummyNote(600, 45, HIHAT_PITCH),
            DummyNote(720, 75, HIHAT_PITCH),
            DummyNote(840, 40, HIHAT_PITCH),
        ]

    def test_init(self):
        """Test initialization."""
        assert self.analyzer.ppq == 480
        assert self.analyzer.bpm == 120.0
        assert self.analyzer.flam_ticks == 28  # 30ms at 120 BPM with 480 PPQ
        assert self.analyzer.buzz_ticks == 48  # 50ms at 120 BPM with 480 PPQ

    def test_analyze_empty_notes(self):
        """Test analysis with no notes."""
        profile = self.analyzer.analyze([])
        assert isinstance(profile, DrumTechniqueProfile)
        assert profile.tightness == 0.0  # Default for empty datasets

    def test_analyze_basic_drum_pattern(self):
        """Test analysis of basic drum pattern."""
        profile = self.analyzer.analyze(self.mock_notes)

        assert isinstance(profile, DrumTechniqueProfile)
        assert isinstance(profile.snare, SnareBounceSignature)
        assert isinstance(profile.hihat, HiHatAlternation)

        # Check overall characteristics
        assert 0.0 <= profile.tightness <= 1.0
        assert 0.0 <= profile.dynamics_range <= 1.0
        assert 0.0 <= profile.ghost_note_density <= 1.0

    def test_calculate_tightness_perfect_grid(self):
        """Test tightness calculation for perfectly quantized notes."""
        # Create perfectly quantized 16th notes
        perfect_notes = []
        for i in range(16):
            tick = i * 120  # 16th notes at 120 BPM, 480 PPQ
            perfect_notes.append(DummyNote(
                onset_ticks=tick,
                pitch=HIHAT_PITCH,
                velocity=60,
                duration_ticks=60,
            ))

        tightness = self.analyzer._calculate_tightness(perfect_notes)
        assert tightness > 0.8  # Should be very tight

    def test_calculate_dynamics_range(self):
        """Test dynamics range calculation."""
        notes = [
            DummyNote(onset_ticks=i, velocity=vel, pitch=SNARE_PITCH)
            for i, vel in enumerate([127, 64, 1])
        ]
        range_val = self.analyzer._calculate_dynamics_range(notes)
        assert abs(range_val - 1.0) < 0.01  # Full range

    def test_calculate_ghost_density(self):
        """Test ghost note density calculation."""
        velocities = [30, 80, 25, 90]
        notes = [
            DummyNote(onset_ticks=i, velocity=vel, pitch=SNARE_PITCH)
            for i, vel in enumerate(velocities)
        ]
        density = self.analyzer._calculate_ghost_density(notes)
        assert density == 0.5  # 2 out of 4 are ghosts

    def test_analyze_snare_bounces_no_snares(self):
        """Test snare analysis with no snare notes."""
        kick_notes = [n for n in self.mock_notes if n.pitch == KICK_PITCH]
        sig = self.analyzer._analyze_snare_bounces(kick_notes)

        assert isinstance(sig, SnareBounceSignature)
        assert sig.flam_count == 0
        assert sig.total_bounces == 0

    def test_analyze_snare_bounces_with_flam(self):
        """Test snare analysis detecting a flam."""
        # Create notes with a flam (two snares very close together)
        snare_notes = [
            DummyNote(onset_ticks=240, pitch=SNARE_PITCH, velocity=90),
            DummyNote(onset_ticks=250, pitch=SNARE_PITCH, velocity=70),  # Flam
        ]

        sig = self.analyzer._analyze_snare_bounces(snare_notes)
        assert sig.flam_count >= 1
        assert sig.total_bounces >= 1

    def test_analyze_hihat_alternation_basic(self):
        """Test hi-hat alternation analysis."""
        hihat_notes = [n for n in self.mock_notes if n.pitch == HIHAT_PITCH]
        alt = self.analyzer._analyze_hihat_alternation(hihat_notes)

        assert isinstance(alt, HiHatAlternation)
        # With our mock data, should detect some alternation
        assert isinstance(alt.is_alternating, bool)
        assert 0.0 <= alt.confidence <= 1.0

    def test_analyze_hihat_alternation_few_notes(self):
        """Test hi-hat analysis with too few notes."""
        few_hihats = [DummyNote(onset_ticks=0, pitch=HIHAT_PITCH, velocity=60)]
        alt = self.analyzer._analyze_hihat_alternation(few_hihats)

        assert isinstance(alt, HiHatAlternation)
        assert alt.is_alternating is False  # Not enough data

    def test_ms_to_ticks_conversion(self):
        """Test millisecond to ticks conversion."""
        ticks = self.analyzer._ms_to_ticks(100.0)  # 100ms at 120 BPM
        expected = int(100.0 * 480 * 120 / 60000)  # PPQ * BPM / 60000 * ms
        assert ticks == expected

    def test_ticks_to_ms_conversion(self):
        """Test ticks to millisecond conversion."""
        ms = self.analyzer._ticks_to_ms(480)  # Quarter note at 120 BPM
        expected = 480 * 60000 / (480 * 120)  # ticks * 60000 / (PPQ * BPM)
        # Should be 500ms (quarter note is 500ms at 120 BPM)
        assert abs(ms - 500.0) < 0.1

    def test_std_calculation(self):
        """Test standard deviation calculation."""
        values = [1, 2, 3, 4, 5]
        std = self.analyzer._std(values)
        # For [1,2,3,4,5], mean=3, variance=2, std=sqrt(2)≈1.414
        assert abs(std - 1.414213562) < 0.01

    def test_std_empty_list(self):
        """Test standard deviation with empty list."""
        std = self.analyzer._std([])
        assert std == 0.0

    def test_std_single_value(self):
        """Test standard deviation with single value."""
        std = self.analyzer._std([5])
        assert std == 0.0


class TestAnalyzeDrumTechnique:
    """Test the convenience function."""

    def test_analyze_drum_technique_function(self):
        """Test the module-level convenience function."""
        # Empty notes
        profile = analyze_drum_technique([])
        assert isinstance(profile, DrumTechniqueProfile)

    def test_analyze_drum_technique_with_params(self):
        """Test convenience function with custom parameters."""
        profile = analyze_drum_technique([], ppq=960, bpm=100.0)
        assert isinstance(profile, DrumTechniqueProfile)


class TestDrumTechniqueProfile:
    """Test the DrumTechniqueProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a profile."""
        profile = DrumTechniqueProfile()
        assert isinstance(profile.snare, SnareBounceSignature)
        assert isinstance(profile.hihat, HiHatAlternation)
        assert profile.tightness == 0.0
        assert profile.dynamics_range == 0.0
        assert profile.ghost_note_density == 0.0

    def test_profile_with_custom_values(self):
        """Test profile with custom field values."""
        profile = DrumTechniqueProfile(
            tightness=0.8,
            dynamics_range=0.6,
            ghost_note_density=0.2
        )
        assert profile.tightness == 0.8
        assert profile.dynamics_range == 0.6
        assert profile.ghost_note_density == 0.2


class TestSnareBounceSignature:
    """Test the SnareBounceSignature dataclass."""

    def test_snare_signature_creation(self):
        """Test creating a snare signature."""
        sig = SnareBounceSignature()
        assert sig.flam_count == 0
        assert sig.drag_count == 0
        assert sig.total_bounces == 0
        assert sig.has_buzz_rolls is False
        assert sig.has_ghost_drags is False
        assert sig.primary_technique == "standard"

    def test_snare_signature_with_bounces(self):
        """Test signature with bounce data."""
        sig = SnareBounceSignature(
            flam_count=2,
            drag_count=1,
            total_bounces=3,
            has_buzz_rolls=True,
            primary_technique="jazzy"
        )
        assert sig.flam_count == 2
        assert sig.drag_count == 1
        assert sig.total_bounces == 3
        assert sig.has_buzz_rolls is True
        assert sig.primary_technique == "jazzy"


class TestHiHatAlternation:
    """Test the HiHatAlternation dataclass."""

    def test_hihat_alternation_creation(self):
        """Test creating hi-hat alternation."""
        alt = HiHatAlternation()
        assert alt.is_alternating is False
        assert alt.confidence == 0.0
        assert alt.dominant_hand == "unknown"
        assert alt.downbeat_avg_velocity == 0.0
        assert alt.upbeat_avg_velocity == 0.0

    def test_hihat_alternation_with_data(self):
        """Test alternation with analysis data."""
        alt = HiHatAlternation(
            is_alternating=True,
            confidence=0.85,
            dominant_hand="right",
            downbeat_avg_velocity=75.0,
            upbeat_avg_velocity=65.0,
            velocity_alternation_ratio=1.15
        )
        assert alt.is_alternating is True
        assert alt.confidence == 0.85
        assert alt.dominant_hand == "right"
        assert alt.downbeat_avg_velocity == 75.0
        assert alt.upbeat_avg_velocity == 65.0
        assert alt.velocity_alternation_ratio == 1.15
