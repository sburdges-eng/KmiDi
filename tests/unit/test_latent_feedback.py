"""Tests for ``music_brain.latent.feedback``.

Closes **Feedback-loop learning**, **Longitudinal user modeling**, and
**Personalized latent calibration**. The feedback layer is what closes
the RAG loop: accepted generations get embedded back into the
LatentMemory with their satisfaction tag, and per-user EMA biases are
maintained so future generation can pull personalized parameters.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.feedback import (  # noqa: E402
    FeedbackEvent,
    UserModel,
)
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402
from music_brain.latent.retrieval import LatentMemory  # noqa: E402


def _prov() -> IntentProvenance:
    return IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0)


def _frame(audio: torch.Tensor, *, time_index: int = 0, emotion_va=(0.0, 0.0)) -> LatentFrame:
    return LatentFrame(
        audio_z=audio,
        chord_z=None,
        emotion_va=emotion_va,
        time_index=time_index,
        provenance=_prov(),
        metadata={},
    )


def test_user_model_starts_neutral() -> None:
    um = UserModel("u1")
    assert um.emotion_bias == (0.0, 0.0)
    assert um.average_satisfaction == 0.0
    assert um.event_count == 0


def test_observe_accumulates_emotion_bias() -> None:
    um = UserModel("u1", ema_alpha=0.5)
    um.observe(FeedbackEvent(emotion_va=(0.6, 0.4), satisfaction=0.9))
    um.observe(FeedbackEvent(emotion_va=(0.2, 0.0), satisfaction=0.7))
    # First observe sets the EMA to the observed value; second blends.
    v, a = um.emotion_bias
    assert 0.0 < v < 0.6
    assert 0.0 < a < 0.4
    assert um.event_count == 2


def test_observe_rejects_satisfaction_out_of_range() -> None:
    um = UserModel("u1")
    with pytest.raises(ValueError, match="satisfaction"):
        um.observe(FeedbackEvent(emotion_va=(0.0, 0.0), satisfaction=-0.1))
    with pytest.raises(ValueError, match="satisfaction"):
        um.observe(FeedbackEvent(emotion_va=(0.0, 0.0), satisfaction=1.1))


def test_observe_rejects_emotion_out_of_range() -> None:
    um = UserModel("u1")
    with pytest.raises(ValueError, match="emotion_va"):
        um.observe(FeedbackEvent(emotion_va=(1.5, 0.0), satisfaction=0.5))


def test_calibrate_intent_va_blends_with_bias() -> None:
    um = UserModel("u1", ema_alpha=1.0)
    um.observe(FeedbackEvent(emotion_va=(0.8, 0.6), satisfaction=1.0))
    out = um.calibrate_emotion_va((0.0, 0.0), blend=0.5)
    # With blend=0.5 and bias (0.8, 0.6), output is the midpoint.
    assert out == pytest.approx((0.4, 0.3), abs=1e-6)


def test_calibrate_blend_zero_returns_input_untouched() -> None:
    um = UserModel("u1")
    um.observe(FeedbackEvent(emotion_va=(0.8, 0.6), satisfaction=1.0))
    out = um.calibrate_emotion_va((0.1, -0.1), blend=0.0)
    assert out == pytest.approx((0.1, -0.1))


def test_low_satisfaction_dampens_bias_pull() -> None:
    """A negative-satisfaction event should not move the bias as much
    as a high-satisfaction event with the same VA."""
    um_high = UserModel("u1", ema_alpha=0.5)
    um_low = UserModel("u2", ema_alpha=0.5)
    um_high.observe(FeedbackEvent(emotion_va=(0.8, 0.4), satisfaction=0.9))
    um_low.observe(FeedbackEvent(emotion_va=(0.8, 0.4), satisfaction=0.1))
    # Both started at (0, 0); the high-satisfaction model moved further.
    assert um_high.emotion_bias[0] > um_low.emotion_bias[0]
    assert um_high.emotion_bias[1] > um_low.emotion_bias[1]


def test_observe_emits_to_memory_when_attached() -> None:
    mem = LatentMemory(dim=4)
    um = UserModel("u1", memory=mem)
    frame = _frame(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    um.observe(
        FeedbackEvent(emotion_va=(0.5, 0.5), satisfaction=0.9, frame=frame, frame_id="gen-1")
    )
    hits = mem.recall(frame, top_k=1, user_id="u1")
    assert hits and hits[0].id == "gen-1"
    assert hits[0].metadata["satisfaction"] == pytest.approx(0.9)
    assert hits[0].metadata["user_id"] == "u1"


def test_event_count_and_average_satisfaction_track_history() -> None:
    um = UserModel("u1")
    um.observe(FeedbackEvent(emotion_va=(0.0, 0.0), satisfaction=0.8))
    um.observe(FeedbackEvent(emotion_va=(0.0, 0.0), satisfaction=0.4))
    assert um.event_count == 2
    assert um.average_satisfaction == pytest.approx(0.6)


def test_invalid_ema_alpha_rejected() -> None:
    with pytest.raises(ValueError, match="ema_alpha"):
        UserModel("u1", ema_alpha=0.0)
    with pytest.raises(ValueError, match="ema_alpha"):
        UserModel("u1", ema_alpha=1.5)
