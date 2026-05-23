"""Per-user feedback loop + longitudinal personalization model.

Closes **Feedback-loop learning**, **Longitudinal user modeling**, and
**Personalized latent calibration**. Together with
``music_brain.latent.retrieval`` this is what makes the RAG loop
closed: every accepted generation is re-embedded into ``LatentMemory``
with its satisfaction tag, *and* the per-user emotion bias is rolled
into an EMA that downstream emission can pull from when constructing
the next ``IntentFrame``.

The bias-pull is satisfaction-weighted: an event the user disliked
moves the bias less than an event they loved. This is intentionally
the minimum closed loop — RL/DPO refinement is downstream work; this
ships the data plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from music_brain.latent.latent_frame import LatentFrame
from music_brain.latent.retrieval import LatentMemory


@dataclass(frozen=True)
class FeedbackEvent:
    """One user-observed generation event.

    Args:
        emotion_va: the (valence, arousal) of the generation under
            review. Range [-1, 1] x [-1, 1].
        satisfaction: the user's [0, 1] rating. 0 = strongly disliked,
            1 = strongly liked.
        frame: optional ``LatentFrame`` to re-embed into memory.
        frame_id: when ``frame`` is set, the id under which it is
            stored. Required if ``frame`` is given.
        intent_id: optional cross-reference to the source IntentFrame.
    """

    emotion_va: Tuple[float, float]
    satisfaction: float
    frame: Optional[LatentFrame] = None
    frame_id: Optional[str] = None
    intent_id: Optional[int] = None


def _validate_event(event: FeedbackEvent) -> None:
    if not (0.0 <= event.satisfaction <= 1.0):
        raise ValueError(f"satisfaction must be in [0, 1], got {event.satisfaction}")
    v, a = event.emotion_va
    if not (-1.0 <= float(v) <= 1.0) or not (-1.0 <= float(a) <= 1.0):
        raise ValueError(f"emotion_va components must be in [-1, 1], got {event.emotion_va}")
    if event.frame is not None and event.frame_id is None:
        raise ValueError("frame_id is required when frame is provided")


class UserModel:
    """Longitudinal per-user latent calibration model.

    Args:
        user_id: stable user identifier.
        ema_alpha: emotion-bias EMA learning rate, in (0, 1]. Larger →
            faster adaptation; smaller → more stability.
        memory: optional ``LatentMemory`` to which accepted frames are
            re-embedded for retrieval-augmented generation.
    """

    def __init__(
        self, user_id: str, *, ema_alpha: float = 0.25, memory: Optional[LatentMemory] = None
    ) -> None:
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self.user_id = user_id
        self._alpha = float(ema_alpha)
        self._bias_v = 0.0
        self._bias_a = 0.0
        self._initialized = False
        self._event_count = 0
        self._satisfaction_sum = 0.0
        self._memory = memory

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, event: FeedbackEvent) -> None:
        _validate_event(event)
        # Satisfaction-weighted EMA: a low-satisfaction event still
        # contributes, but proportionally less. Pure (0, 0) bias on
        # zero-satisfaction would discard information we want to learn
        # from ("user didn't like this VA"), so we floor the weight to
        # alpha * 0.25 — disliked events nudge weakly in the OPPOSITE
        # direction… actually no, simpler: just scale alpha by sat.
        eff_alpha = self._alpha * max(0.05, event.satisfaction)
        v, a = float(event.emotion_va[0]), float(event.emotion_va[1])
        if not self._initialized:
            # Cold start: anchor the EMA on the first observation
            # (scaled by satisfaction so a "hated" first event doesn't
            # fully pin the bias).
            self._bias_v = eff_alpha * v
            self._bias_a = eff_alpha * a
            self._initialized = True
        else:
            self._bias_v = (1.0 - eff_alpha) * self._bias_v + eff_alpha * v
            self._bias_a = (1.0 - eff_alpha) * self._bias_a + eff_alpha * a
        self._event_count += 1
        self._satisfaction_sum += event.satisfaction

        if self._memory is not None and event.frame is not None and event.frame_id:
            self._memory.remember(
                event.frame_id,
                event.frame,
                metadata={
                    "user_id": self.user_id,
                    "satisfaction": float(event.satisfaction),
                    "intent_id": event.intent_id,
                },
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate_emotion_va(
        self, requested: Tuple[float, float], *, blend: float = 0.25
    ) -> Tuple[float, float]:
        """Mix the requested VA with the learned bias.

        Args:
            requested: the (valence, arousal) the caller wants right now.
            blend: in [0, 1]. 0.0 returns ``requested`` unchanged;
                1.0 returns the user bias.
        """
        if not (0.0 <= blend <= 1.0):
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        if not self._initialized:
            return (float(requested[0]), float(requested[1]))
        v = (1.0 - blend) * float(requested[0]) + blend * self._bias_v
        a = (1.0 - blend) * float(requested[1]) + blend * self._bias_a
        return (v, a)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def emotion_bias(self) -> Tuple[float, float]:
        return (self._bias_v, self._bias_a)

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def average_satisfaction(self) -> float:
        if self._event_count == 0:
            return 0.0
        return self._satisfaction_sum / self._event_count


__all__ = ["FeedbackEvent", "UserModel"]
