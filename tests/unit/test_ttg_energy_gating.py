"""TTG energy-curve-driven orchestration gating (PRD §8.2 active_threshold)."""

import pytest

from music_brain.api_schemas.ttg_adapter import (
    compute_section_energy,
    interpolate_energy_at_bar,
    orchestration_with_energy_gating,
    validate_ttg_expansion,
)
from music_brain.api_schemas.ttg_v1 import (
    EnergyCurvePointV1,
    EnergyCurveV1,
    OrchestrationRoleSpecV1,
    TTGOrchestrationV1,
)


# ---------------------------------------------------------------------------
# interpolate_energy_at_bar
# ---------------------------------------------------------------------------


def test_interpolate_exact_point_returns_its_value():
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.2),
            EnergyCurvePointV1(bar=8.0, value=0.9),
        ]
    )
    assert interpolate_energy_at_bar(curve, 0.0) == pytest.approx(0.2)
    assert interpolate_energy_at_bar(curve, 8.0) == pytest.approx(0.9)


def test_interpolate_linear_between_points():
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.2),
            EnergyCurvePointV1(bar=8.0, value=1.0),
        ]
    )
    # halfway: 0.2 + (1.0 - 0.2) * 0.5 = 0.6
    assert interpolate_energy_at_bar(curve, 4.0) == pytest.approx(0.6)
    # quarter: 0.2 + (1.0 - 0.2) * 0.25 = 0.4
    assert interpolate_energy_at_bar(curve, 2.0) == pytest.approx(0.4)


def test_interpolate_before_first_point_clamps_to_first_value():
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=4.0, value=0.5),
            EnergyCurvePointV1(bar=8.0, value=0.9),
        ]
    )
    assert interpolate_energy_at_bar(curve, 0.0) == pytest.approx(0.5)
    assert interpolate_energy_at_bar(curve, 2.0) == pytest.approx(0.5)


def test_interpolate_after_last_point_clamps_to_last_value():
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.2),
            EnergyCurvePointV1(bar=4.0, value=0.6),
        ]
    )
    assert interpolate_energy_at_bar(curve, 8.0) == pytest.approx(0.6)
    assert interpolate_energy_at_bar(curve, 100.0) == pytest.approx(0.6)


def test_interpolate_handles_multiple_segments():
    # 3-point curve: rise then fall
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.0),
            EnergyCurvePointV1(bar=4.0, value=1.0),
            EnergyCurvePointV1(bar=8.0, value=0.0),
        ]
    )
    # rising segment midpoint
    assert interpolate_energy_at_bar(curve, 2.0) == pytest.approx(0.5)
    # peak
    assert interpolate_energy_at_bar(curve, 4.0) == pytest.approx(1.0)
    # falling segment midpoint
    assert interpolate_energy_at_bar(curve, 6.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_section_energy
# ---------------------------------------------------------------------------


def test_compute_section_energy_returns_peak_and_mean():
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.2),
            EnergyCurvePointV1(bar=8.0, value=1.0),
        ]
    )
    # section spans bars 0..8
    result = compute_section_energy(curve, section_start_bar=0.0, section_bars=8)
    assert result["peak"] == pytest.approx(1.0)
    # mean of a linear ramp 0.2→1.0 is (0.2+1.0)/2 = 0.6
    assert result["mean"] == pytest.approx(0.6, abs=0.05)


def test_compute_section_energy_clamps_outside_curve_range():
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.5),
            EnergyCurvePointV1(bar=4.0, value=0.5),
        ]
    )
    # Section beyond curve range — should clamp to last value
    result = compute_section_energy(curve, section_start_bar=8.0, section_bars=4)
    assert result["peak"] == pytest.approx(0.5)
    assert result["mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# orchestration_with_energy_gating
# ---------------------------------------------------------------------------


def test_orchestration_without_energy_curve_matches_legacy_behavior():
    """No curve → behavior matches orchestration_to_instruments."""
    orch = TTGOrchestrationV1(
        roles={
            "bass": OrchestrationRoleSpecV1(patch="808_clean", active_threshold=0.5),
            "lead": OrchestrationRoleSpecV1(patch="square_lead", active_threshold=0.8),
        }
    )
    instruments = orchestration_with_energy_gating(orch, energy_curve=None, structure=[])
    assert instruments == [
        {"instrument": "808_clean", "techniques": ["role:bass"]},
        {"instrument": "square_lead", "techniques": ["role:lead"]},
    ]


def test_orchestration_with_curve_annotates_active_sections():
    """With energy curve, technique list gains `active_in:` markers for each section
    whose peak energy meets the role's threshold."""
    structure = [
        {"name": "intro", "bars": 4, "repetitions": 1},
        {"name": "verse", "bars": 4, "repetitions": 1},
        {"name": "chorus", "bars": 4, "repetitions": 1},
    ]
    # Energy curve: rises across the song. Section peaks (at endpoint breakpoints):
    #   intro [0..4]   → peak = max(0.1, 0.4)  = 0.4
    #   verse [4..8]   → peak = max(0.4, 0.7)  = 0.7
    #   chorus [8..12] → peak = max(0.7, 1.0)  = 1.0
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.1),
            EnergyCurvePointV1(bar=4.0, value=0.4),
            EnergyCurvePointV1(bar=8.0, value=0.7),
            EnergyCurvePointV1(bar=12.0, value=1.0),
        ]
    )
    orch = TTGOrchestrationV1(
        roles={
            "bass": OrchestrationRoleSpecV1(patch="808", active_threshold=0.5),  # verse+chorus
            "lead": OrchestrationRoleSpecV1(patch="lead", active_threshold=0.9),  # chorus only
            "pad": OrchestrationRoleSpecV1(patch="pad", active_threshold=0.0),  # always
        }
    )
    instruments = orchestration_with_energy_gating(orch, energy_curve=curve, structure=structure)

    # Roles are sorted alphabetically (same as legacy adapter).
    by_role = {ins["instrument"]: ins["techniques"] for ins in instruments}
    assert by_role["808"] == ["role:bass", "active_in:verse", "active_in:chorus"]
    assert by_role["lead"] == ["role:lead", "active_in:chorus"]
    assert by_role["pad"] == ["role:pad", "active_in:intro", "active_in:verse", "active_in:chorus"]


def test_orchestration_role_never_active_is_filtered_out():
    """A role whose threshold is never met by any section is dropped from output."""
    structure = [
        {"name": "intro", "bars": 4, "repetitions": 1},
        {"name": "verse", "bars": 4, "repetitions": 1},
    ]
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.1),
            EnergyCurvePointV1(bar=8.0, value=0.4),
        ]
    )
    orch = TTGOrchestrationV1(
        roles={
            "lead": OrchestrationRoleSpecV1(patch="square", active_threshold=0.9),  # never met
            "bass": OrchestrationRoleSpecV1(patch="808", active_threshold=0.2),  # met in verse
        }
    )
    instruments = orchestration_with_energy_gating(orch, energy_curve=curve, structure=structure)
    patches = [ins["instrument"] for ins in instruments]
    assert "square" not in patches
    assert "808" in patches


def test_orchestration_with_empty_structure_falls_back_to_legacy():
    """No structure to gate against → behave like legacy (no per-section markers)."""
    orch = TTGOrchestrationV1(
        roles={
            "bass": OrchestrationRoleSpecV1(patch="808", active_threshold=0.5),
        }
    )
    curve = EnergyCurveV1(
        points=[
            EnergyCurvePointV1(bar=0.0, value=0.6),
            EnergyCurvePointV1(bar=4.0, value=0.7),
        ]
    )
    instruments = orchestration_with_energy_gating(orch, energy_curve=curve, structure=[])
    assert instruments == [{"instrument": "808", "techniques": ["role:bass"]}]


# ---------------------------------------------------------------------------
# validate_ttg_expansion
# ---------------------------------------------------------------------------


def test_validate_ttg_expansion_flags_unreachable_thresholds():
    """Validator should report roles whose threshold exceeds the max energy sample."""
    normalized = {
        "structure": [{"name": "verse", "bars": 8, "repetitions": 1}],
        "instruments": [],  # not consulted
        "energy_curve": {
            "points": [{"bar": 0.0, "value": 0.1}, {"bar": 8.0, "value": 0.4}],
        },
        "orchestration": {
            "roles": {
                "bass": {"patch": "808", "active_threshold": 0.3},
                "lead": {"patch": "square", "active_threshold": 0.9},  # unreachable
            }
        },
    }
    issues = validate_ttg_expansion(normalized)
    assert any("lead" in issue and "0.9" in issue and "0.4" in issue for issue in issues), issues
    assert not any("bass" in issue for issue in issues)


def test_validate_ttg_expansion_no_issues_when_no_energy_curve():
    """Without energy_curve, no gating checks apply — return empty list."""
    normalized = {
        "structure": [{"name": "verse", "bars": 8, "repetitions": 1}],
        "instruments": [],
        "orchestration": {
            "roles": {"bass": {"patch": "808", "active_threshold": 0.5}},
        },
    }
    assert validate_ttg_expansion(normalized) == []


def test_validate_ttg_expansion_no_issues_when_no_orchestration():
    """Energy curve alone without orchestration → no issues."""
    normalized = {
        "structure": [{"name": "verse", "bars": 8, "repetitions": 1}],
        "instruments": [],
        "energy_curve": {
            "points": [{"bar": 0.0, "value": 0.2}, {"bar": 8.0, "value": 0.9}],
        },
    }
    assert validate_ttg_expansion(normalized) == []


# ---------------------------------------------------------------------------
# Pipeline integration — energy gating wired into IntentPipeline.normalize.
# ---------------------------------------------------------------------------


def test_pipeline_normalize_emits_energy_gated_instruments():
    """When energy_curve + orchestration are both present, normalize() should emit
    instruments with `active_in:` techniques rather than the legacy unconditional
    `role:` only output."""
    from music_brain.api_schemas.generate_v1 import (
        EmotionalIntent,
        GenerateRequest,
        TechnicalIntent,
    )
    from music_brain.api_schemas.ttg_v1 import TTGMovementV1, TTGPhraseV1
    from music_brain.pipeline import IntentPipeline

    tech = TechnicalIntent(
        key="C major",
        bpm=120,
        genre="electronic",
        structure=[{"name": "verse", "bars": 8}],
        instruments=[{"instrument": "ignored", "techniques": []}],
        timeline=TTGMovementV1(
            id="A",
            bars=12,
            children=[
                TTGPhraseV1(bars=4, section_role="intro"),
                TTGPhraseV1(bars=8, section_role="chorus"),
            ],
        ),
        orchestration=TTGOrchestrationV1(
            roles={
                "bass": OrchestrationRoleSpecV1(patch="808", active_threshold=0.5),
                "lead": OrchestrationRoleSpecV1(patch="square", active_threshold=0.95),
            }
        ),
        energy_curve={
            "points": [
                {"bar": 0.0, "value": 0.2},
                {"bar": 4.0, "value": 0.6},
                {"bar": 12.0, "value": 1.0},
            ]
        },
    )
    req = GenerateRequest(intent=EmotionalIntent(emotional_intent="tension", technical=tech))
    normalized = IntentPipeline().normalize(req)
    by_patch = {ins["instrument"]: ins["techniques"] for ins in normalized["instruments"]}
    # bass threshold 0.5: intro peak ≈ 0.6 → active; chorus peak 1.0 → active.
    assert "808" in by_patch
    assert "active_in:intro" in by_patch["808"]
    assert "active_in:chorus" in by_patch["808"]
    # lead threshold 0.95: intro peak 0.6 → no; chorus peak 1.0 → yes.
    assert "square" in by_patch
    assert "active_in:intro" not in by_patch["square"]
    assert "active_in:chorus" in by_patch["square"]
