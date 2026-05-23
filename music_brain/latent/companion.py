"""Creative companion: human-in-the-loop orchestration around the latent core.

Closes:
  - Creative companion interaction model
  - Human-in-the-loop generation
  - Adaptive orchestration layers
  - Adaptive state-aware delivery
  - Intent normalization pipeline

``CompanionSession`` is the small orchestrator a UI sits on top of:

  - ``normalize_intent(valence, arousal, mood)`` → ``IntentFrame``.
    The mood string is mapped to a few canonical musical adjustments
    (``energetic`` bumps tempo + density; ``calm`` softens both; …).
    Unknown moods fall through to neutral music, which mirrors the
    existing ``IntentIREmitter.emit_from_emotion`` behavior.

  - ``propose(frame, horizon)`` rolls the world-model forward to
    suggest ``horizon`` continuation frames, applying the user's
    calibrated VA bias (from the per-session ``UserModel``) when
    ``calibration_blend > 0``.

  - ``checkpoint(frame)`` / ``accept(...)`` / ``reject(...)`` are the
    realtime human-in-the-loop surface. Accept records positive
    feedback into ``UserModel`` + ``LatentMemory``; reject rewinds
    via the ``RollbackRing`` *and* records the negative feedback so
    the model learns from rejections too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from music_brain.intent_ir import (
    EmotionState,
    IntentConstraints,
    IntentFrame,
    IntentMeta,
    IntentProvenance,
    IntentSource,
    MusicalIntent,
    TimeScope,
)
from music_brain.latent.feedback import FeedbackEvent, UserModel
from music_brain.latent.latent_frame import LatentFrame
from music_brain.latent.predictors import PredictionEngine
from music_brain.latent.retrieval import LatentMemory
from music_brain.latent.rollback import RollbackRing
from music_brain.world_model import WorldModel


_MOOD_TABLE: dict[str, tuple[float, float, float]] = {
    # mood -> (tempo_bias, rhythmic_density, dynamic_range)
    "energetic": (0.4, 0.8, 0.7),
    "calm": (-0.4, 0.3, 0.4),
    "tense": (0.0, 0.7, 0.6),
    "dreamy": (-0.2, 0.4, 0.5),
    "driving": (0.5, 0.85, 0.6),
    "sad": (-0.3, 0.4, 0.4),
    "joyful": (0.2, 0.7, 0.65),
}


@dataclass(frozen=True)
class HumanFeedback:
    accepted: bool
    satisfaction: float  # [0, 1]
    notes: str = ""


def _validate_va(valence: float, arousal: float) -> None:
    if not (-1.0 <= float(valence) <= 1.0):
        raise ValueError(f"valence must be in [-1, 1], got {valence}")
    if not (-1.0 <= float(arousal) <= 1.0):
        raise ValueError(f"arousal must be in [-1, 1], got {arousal}")


class CompanionSession:
    """One user / session orchestrator around the latent core."""

    def __init__(self, *, user_id: str,
                 world_model: Optional[WorldModel] = None,
                 memory: Optional[LatentMemory] = None,
                 calibration_blend: float = 0.0) -> None:
        if not user_id:
            raise ValueError("user_id must be a non-empty string")
        self._user_id = user_id
        self._world_model = world_model
        self._memory = memory
        self._user_model = UserModel(user_id=user_id, memory=memory)
        self._rollback = RollbackRing(capacity=32)
        self._calibration_blend = float(calibration_blend)
        self._next_intent_id = 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def user_model(self) -> UserModel:
        return self._user_model

    # ------------------------------------------------------------------
    # Intent normalization
    # ------------------------------------------------------------------

    def normalize_intent(self, *, valence: float, arousal: float,
                         mood: Optional[str] = None) -> IntentFrame:
        _validate_va(valence, arousal)
        intent_id = self._next_intent_id
        self._next_intent_id += 1
        music = MusicalIntent()
        if mood:
            adjustments = _MOOD_TABLE.get(mood.lower())
            if adjustments:
                tb, rd, dr = adjustments
                music = MusicalIntent(
                    tempo_bias=tb,
                    rhythmic_density=rd,
                    dynamic_range=dr,
                )
        return IntentFrame(
            meta=IntentMeta(intent_id=intent_id),
            emotion=EmotionState(valence=float(valence),
                                 arousal=float(arousal),
                                 dominance=0.5,
                                 confidence=1.0),
            music=music,
            time=TimeScope(),
            constraints=IntentConstraints(),
            provenance=IntentProvenance(source=IntentSource.UI_DIRECT,
                                        user_override_weight=1.0),
        )

    # ------------------------------------------------------------------
    # Propose / checkpoint / accept / reject
    # ------------------------------------------------------------------

    def propose(self, frame: LatentFrame, *, horizon: int
                ) -> Iterator[LatentFrame]:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        if self._world_model is None:
            self._world_model = WorldModel(state_dim=frame.audio_feature_dim)
        eng = PredictionEngine(world_model=self._world_model, name="companion")
        trajectory = eng.predict(frame, horizon=horizon)
        # Apply calibration to each emitted frame's emotion_va tag.
        calibrated_va = self._calibrate_va(frame.emotion_va)
        for f in trajectory:
            if calibrated_va == f.emotion_va:
                yield f
            else:
                yield LatentFrame(
                    audio_z=f.audio_z, chord_z=f.chord_z,
                    emotion_va=calibrated_va,
                    time_index=f.time_index,
                    provenance=f.provenance,
                    metadata=dict(f.metadata),
                )

    def checkpoint(self, frame: LatentFrame) -> None:
        self._rollback.checkpoint(frame)

    def accept(self, frame: LatentFrame, feedback: HumanFeedback) -> None:
        self._observe(frame, feedback)

    def reject(self, frame: LatentFrame, feedback: HumanFeedback
               ) -> LatentFrame:
        self._observe(frame, feedback)
        # Roll back to the most recent checkpoint, if any.
        if len(self._rollback) == 0:
            return frame  # nothing to revert to
        head = self._rollback.head_index
        return self._rollback.rollback_to(head) if head is not None else frame

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observe(self, frame: LatentFrame, feedback: HumanFeedback) -> None:
        self._user_model.observe(FeedbackEvent(
            emotion_va=frame.emotion_va,
            satisfaction=float(feedback.satisfaction),
            frame=frame,
            frame_id=f"{self._user_id}-{frame.time_index}",
        ))

    def _calibrate_va(self, requested: Tuple[float, float]
                      ) -> Tuple[float, float]:
        if self._calibration_blend <= 0.0:
            return requested
        return self._user_model.calibrate_emotion_va(
            requested, blend=self._calibration_blend)


__all__ = ["CompanionSession", "HumanFeedback"]
