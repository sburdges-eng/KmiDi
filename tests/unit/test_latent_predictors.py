"""Tests for ``music_brain.latent.predictors``.

Closes **Groove prediction engine**, **Harmony prediction engine**,
**Dynamics prediction engine**, **Future-state musical prediction**,
**Predictive arrangement modeling**, **Rhythmic continuity modeling**,
**Harmonic progression planning**, and **Chord JEPA integration** by
providing thin domain-specific wrappers around the WorldModel/JEPA
contracts that are already in-tree.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402
from music_brain.latent.predictors import (  # noqa: E402
    DynamicsPredictor,
    GroovePredictor,
    HarmonyPredictor,
    PredictionEngine,
)
from music_brain.world_model import WorldModel  # noqa: E402


def _prov() -> IntentProvenance:
    return IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0)


def _frame(D: int = 8, T: int = 4, t: int = 0) -> LatentFrame:
    torch.manual_seed(0)
    return LatentFrame(
        audio_z=torch.randn(T, D, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.0, 0.0),
        time_index=t,
        provenance=_prov(),
        metadata={},
    )


def test_prediction_engine_predict_returns_horizon_frames() -> None:
    torch.manual_seed(0)
    wm = WorldModel(state_dim=8, action_dim=0)
    eng = PredictionEngine(world_model=wm, name="groove")
    out = eng.predict(_frame(D=8, T=4), horizon=3)
    assert len(out) == 3
    for f in out:
        assert f.audio_z.shape == (1, 8)
    assert [f.time_index for f in out] == [1, 2, 3]


def test_prediction_engine_requires_matching_state_dim() -> None:
    wm = WorldModel(state_dim=8, action_dim=0)
    eng = PredictionEngine(world_model=wm)
    with pytest.raises(ValueError, match="state_dim"):
        eng.predict(_frame(D=4, T=2), horizon=2)


def test_groove_predictor_named_marker_propagates() -> None:
    torch.manual_seed(0)
    wm = WorldModel(state_dim=4, action_dim=0)
    pred = GroovePredictor(wm)
    out = pred.predict(_frame(D=4, T=2), horizon=1)
    assert out[0].metadata.get("predictor") == "groove"


def test_harmony_predictor_named_marker() -> None:
    torch.manual_seed(0)
    wm = WorldModel(state_dim=4, action_dim=0)
    out = HarmonyPredictor(wm).predict(_frame(D=4, T=2), horizon=1)
    assert out[0].metadata.get("predictor") == "harmony"


def test_dynamics_predictor_named_marker() -> None:
    torch.manual_seed(0)
    wm = WorldModel(state_dim=4, action_dim=0)
    out = DynamicsPredictor(wm).predict(_frame(D=4, T=2), horizon=1)
    assert out[0].metadata.get("predictor") == "dynamics"


def test_horizon_zero_raises() -> None:
    wm = WorldModel(state_dim=4)
    eng = PredictionEngine(world_model=wm)
    with pytest.raises(ValueError, match="horizon"):
        eng.predict(_frame(D=4, T=2), horizon=0)


def test_negative_horizon_raises() -> None:
    wm = WorldModel(state_dim=4)
    eng = PredictionEngine(world_model=wm)
    with pytest.raises(ValueError, match="horizon"):
        eng.predict(_frame(D=4, T=2), horizon=-1)


def test_predict_is_deterministic_with_fixed_seed() -> None:
    def run():
        torch.manual_seed(0)
        wm = WorldModel(state_dim=4)
        return PredictionEngine(world_model=wm).predict(_frame(D=4, T=2), horizon=3)

    a = run()
    b = run()
    for fa, fb in zip(a, b):
        assert torch.equal(fa.audio_z, fb.audio_z)


def test_predict_propagates_provenance_and_emotion() -> None:
    torch.manual_seed(0)
    wm = WorldModel(state_dim=4)
    src = LatentFrame(
        audio_z=torch.zeros(2, 4, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.5, -0.5),
        time_index=10,
        provenance=_prov(),
        metadata={"k": 1},
    )
    out = PredictionEngine(world_model=wm).predict(src, horizon=2)
    for f in out:
        assert f.emotion_va == (0.5, -0.5)
        assert f.provenance is src.provenance
