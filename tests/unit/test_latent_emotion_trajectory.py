"""Tests for ``music_brain.latent.emotion_trajectory``.

Closes **Emotion trajectory planning** and **Latent emotional
steering**. The trajectory planner takes a sequence of (valence,
arousal) waypoints with bar positions and yields smooth interpolated
VA tuples per bar, suitable to feed back into IntentFrame emission
or to bias the LatentFrame.emotion_va tag.
"""

from __future__ import annotations

import pytest

from music_brain.latent.emotion_trajectory import (  # noqa: E402
    EmotionTrajectory,
    EmotionWaypoint,
)


def test_single_waypoint_yields_constant_va() -> None:
    traj = EmotionTrajectory(waypoints=[EmotionWaypoint(bar=0, valence=0.5, arousal=0.5)])
    out = list(traj.sample_bars(start_bar=0, end_bar=4))
    assert all(v == 0.5 and a == 0.5 for _, v, a in out)
    assert [b for b, _, _ in out] == [0, 1, 2, 3]


def test_two_waypoints_linearly_interpolate() -> None:
    traj = EmotionTrajectory(
        waypoints=[
            EmotionWaypoint(bar=0, valence=0.0, arousal=0.0),
            EmotionWaypoint(bar=4, valence=1.0, arousal=-1.0),
        ]
    )
    out = list(traj.sample_bars(start_bar=0, end_bar=5))
    expected = [
        (0, 0.0, 0.0),
        (1, 0.25, -0.25),
        (2, 0.5, -0.5),
        (3, 0.75, -0.75),
        (4, 1.0, -1.0),
    ]
    assert out == pytest.approx(expected, abs=1e-6)


def test_waypoint_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="valence"):
        EmotionWaypoint(bar=0, valence=2.0, arousal=0.0)
    with pytest.raises(ValueError, match="bar"):
        EmotionWaypoint(bar=-1, valence=0.0, arousal=0.0)


def test_sample_before_first_waypoint_holds_initial_value() -> None:
    traj = EmotionTrajectory(
        waypoints=[
            EmotionWaypoint(bar=4, valence=0.5, arousal=0.5),
            EmotionWaypoint(bar=8, valence=-0.5, arousal=-0.5),
        ]
    )
    out = list(traj.sample_bars(start_bar=0, end_bar=4))
    assert all((v == 0.5 and a == 0.5) for _, v, a in out)


def test_sample_after_last_waypoint_holds_final_value() -> None:
    traj = EmotionTrajectory(
        waypoints=[
            EmotionWaypoint(bar=0, valence=0.5, arousal=0.5),
            EmotionWaypoint(bar=4, valence=-0.5, arousal=-0.5),
        ]
    )
    out = list(traj.sample_bars(start_bar=4, end_bar=8))
    assert all(v == -0.5 and a == -0.5 for _, v, a in out)


def test_waypoints_must_be_strictly_monotonic_by_bar() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        EmotionTrajectory(
            waypoints=[
                EmotionWaypoint(bar=4, valence=0.0, arousal=0.0),
                EmotionWaypoint(bar=4, valence=0.5, arousal=0.5),
            ]
        )
    with pytest.raises(ValueError, match="monotonic"):
        EmotionTrajectory(
            waypoints=[
                EmotionWaypoint(bar=4, valence=0.0, arousal=0.0),
                EmotionWaypoint(bar=2, valence=0.5, arousal=0.5),
            ]
        )


def test_empty_trajectory_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        EmotionTrajectory(waypoints=[])


def test_sample_inclusive_exclusive() -> None:
    """sample_bars is start-inclusive, end-exclusive — mirrors GenerationScope."""
    traj = EmotionTrajectory(waypoints=[EmotionWaypoint(bar=0, valence=0.0, arousal=0.0)])
    out = list(traj.sample_bars(start_bar=2, end_bar=5))
    assert [b for b, _, _ in out] == [2, 3, 4]


def test_blend_with_user_bias_pulls_toward_bias() -> None:
    traj = EmotionTrajectory(waypoints=[EmotionWaypoint(bar=0, valence=0.0, arousal=0.0)])
    bias = (0.8, 0.4)
    out = list(traj.sample_bars(start_bar=0, end_bar=2, blend_with=bias, blend=0.5))
    # All bars produce (0.4, 0.2) — midpoint of (0,0) and bias.
    for _, v, a in out:
        assert v == pytest.approx(0.4)
        assert a == pytest.approx(0.2)


def test_blend_validates_range() -> None:
    traj = EmotionTrajectory(waypoints=[EmotionWaypoint(bar=0, valence=0.0, arousal=0.0)])
    with pytest.raises(ValueError, match="blend"):
        list(traj.sample_bars(start_bar=0, end_bar=1, blend_with=(0.0, 0.0), blend=-0.1))


def test_clamped_to_unit_box() -> None:
    """Even if interpolation overshoots numerically, output stays in [-1, 1]."""
    traj = EmotionTrajectory(
        waypoints=[
            EmotionWaypoint(bar=0, valence=-1.0, arousal=-1.0),
            EmotionWaypoint(bar=2, valence=1.0, arousal=1.0),
        ]
    )
    out = list(traj.sample_bars(start_bar=0, end_bar=3))
    for _, v, a in out:
        assert -1.0 <= v <= 1.0
        assert -1.0 <= a <= 1.0
