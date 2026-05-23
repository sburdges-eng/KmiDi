"""Tests for ``music_brain.latent.companion``.

Closes **Creative companion interaction model**, **Human-in-the-loop
generation**, **Adaptive orchestration layers**, **Adaptive
state-aware delivery**, and **Intent normalization pipeline** (the
small natural-VA hints → IntentFrame helper colocated here).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentFrame, IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.companion import (  # noqa: E402
    CompanionSession,
    HumanFeedback,
)
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402


def _frame(t: int = 0) -> LatentFrame:
    return LatentFrame(
        audio_z=torch.zeros(1, 4, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.0, 0.0),
        time_index=t,
        provenance=IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0),
        metadata={},
    )


def test_session_intent_normalization_returns_intent_frame() -> None:
    s = CompanionSession(user_id="u1")
    intent = s.normalize_intent(valence=0.4, arousal=0.6, mood="energetic")
    assert isinstance(intent, IntentFrame)
    assert intent.emotion.valence == pytest.approx(0.4)
    assert intent.emotion.arousal == pytest.approx(0.6)
    # 'energetic' bumps tempo_bias positive and rhythmic_density up.
    assert intent.music.tempo_bias > 0.0
    assert intent.music.rhythmic_density > 0.5


def test_session_intent_unknown_mood_falls_back_to_neutral_music() -> None:
    s = CompanionSession(user_id="u1")
    intent = s.normalize_intent(valence=0.0, arousal=0.0, mood="floofgriz")
    assert intent.music.tempo_bias == pytest.approx(0.0)


def test_session_propose_emits_latent_frames() -> None:
    s = CompanionSession(user_id="u1")
    out = list(s.propose(_frame(t=0), horizon=3))
    assert len(out) == 3
    assert [f.time_index for f in out] == [1, 2, 3]


def test_session_accept_records_feedback() -> None:
    s = CompanionSession(user_id="u1")
    out = list(s.propose(_frame(t=0), horizon=2))
    feedback = HumanFeedback(accepted=True, satisfaction=0.9, notes="nice motif")
    s.accept(out[-1], feedback)
    # User model now reflects the accepted event.
    assert s.user_model.event_count == 1
    assert s.user_model.average_satisfaction == pytest.approx(0.9)


def test_session_reject_triggers_rollback_and_does_not_grow_memory() -> None:
    s = CompanionSession(user_id="u1")
    initial = _frame(t=0)
    s.checkpoint(initial)
    out = list(s.propose(initial, horizon=2))
    feedback = HumanFeedback(accepted=False, satisfaction=0.1)
    restored = s.reject(out[-1], feedback)
    # Rollback returns the checkpointed frame, not the rejected one.
    assert restored.time_index == 0
    # Reject still records the feedback (low satisfaction) so the
    # personalization model learns to avoid this direction.
    assert s.user_model.event_count == 1


def test_session_user_id_required_to_be_nonempty() -> None:
    with pytest.raises(ValueError, match="user_id"):
        CompanionSession(user_id="")


def test_session_normalize_intent_validates_va_range() -> None:
    s = CompanionSession(user_id="u1")
    with pytest.raises(ValueError, match="valence"):
        s.normalize_intent(valence=1.5, arousal=0.0)


def test_propose_horizon_validated() -> None:
    s = CompanionSession(user_id="u1")
    with pytest.raises(ValueError, match="horizon"):
        list(s.propose(_frame(t=0), horizon=0))


def test_rejected_frames_never_land_in_latent_memory() -> None:
    """Regression: CompanionSession.reject must record the feedback into
    the user model EMA (so the model learns from the negative) but must
    NOT write the rejected frame into LatentMemory — otherwise
    emotion-conditioned recall later surfaces auditions the user
    explicitly rejected. Reported by Cursor Bugbot on PR #196."""
    from music_brain.latent.retrieval import LatentMemory  # noqa: PLC0415

    mem = LatentMemory(dim=4)
    s = CompanionSession(user_id="u1")
    # Wire the memory in via the session's user_model (which holds the
    # memory reference). We replace the user_model with one that has
    # the memory attached so reject's plumbing reaches it.
    from music_brain.latent.feedback import UserModel  # noqa: PLC0415

    s._user_model = UserModel(user_id="u1", memory=mem)

    out = list(s.propose(_frame(t=0), horizon=1))
    s.checkpoint(_frame(t=0))
    s.reject(out[-1], HumanFeedback(accepted=False, satisfaction=0.1))
    assert s.user_model.event_count == 1  # EMA still updated
    assert len(mem) == 0  # memory NOT polluted


def test_accepted_frames_do_land_in_latent_memory() -> None:
    """Symmetric coverage to the reject test: accept must write."""
    from music_brain.latent.feedback import UserModel  # noqa: PLC0415
    from music_brain.latent.retrieval import LatentMemory  # noqa: PLC0415

    mem = LatentMemory(dim=4)
    s = CompanionSession(user_id="u1")
    s._user_model = UserModel(user_id="u1", memory=mem)

    out = list(s.propose(_frame(t=0), horizon=1))
    s.accept(out[-1], HumanFeedback(accepted=True, satisfaction=0.9))
    assert len(mem) == 1


def test_propose_uses_user_calibrated_va_when_blend_set() -> None:
    s = CompanionSession(user_id="u1", calibration_blend=0.5)
    # Teach the user model that the user really likes (0.8, 0.8).
    liked = LatentFrame(
        audio_z=torch.zeros(1, 4, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.8, 0.8),
        time_index=0,
        provenance=IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0),
        metadata={},
    )
    s.accept(liked, HumanFeedback(accepted=True, satisfaction=1.0))
    # Now propose against a neutral-VA seed — output VA should be pulled
    # toward the (0.8, 0.8) bias the user model learned.
    seed = _frame(t=1)
    out = list(s.propose(seed, horizon=1))
    assert out[0].emotion_va != (0.0, 0.0)
    assert out[0].emotion_va[0] > 0.0
    assert out[0].emotion_va[1] > 0.0
