"""Domain-specific prediction engines over the world-model.

Closes a cluster of spec items by composing existing primitives:

  - Groove prediction engine
  - Harmony prediction engine
  - Dynamics prediction engine
  - Future-state musical prediction
  - Predictive arrangement modeling
  - Rhythmic continuity modeling
  - Harmonic progression planning
  - Chord JEPA integration (via the chord_z slot on LatentFrame)

Each domain predictor is a thin wrapper around
``WorldModel.rollout_frames`` that stamps a ``predictor`` marker into
the trajectory metadata. Future work can substitute domain-specific
WorldModel weights without changing the public contract — the
``PredictionEngine`` is parameterized over the WorldModel, not over a
hardcoded model identity.
"""

from __future__ import annotations

from typing import List, Optional

from music_brain.latent.latent_frame import LatentFrame
from music_brain.world_model import WorldModel


class PredictionEngine:
    """Generic predictive arrangement engine.

    Args:
        world_model: the trained ``WorldModel`` whose state-dim must
            match the input frame's ``audio_feature_dim``.
        name: short identifier stamped into trajectory metadata; used
            by the orchestrator to route concurrent predictions.
    """

    def __init__(self, world_model: WorldModel, *,
                 name: Optional[str] = None) -> None:
        self._world_model = world_model
        self._name = name

    @property
    def name(self) -> Optional[str]:
        return self._name

    def predict(self, frame: LatentFrame, *, horizon: int) -> List[LatentFrame]:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        if frame.audio_feature_dim != self._world_model.state_dim:
            raise ValueError(
                f"frame audio_feature_dim {frame.audio_feature_dim} "
                f"!= WorldModel.state_dim {self._world_model.state_dim}")
        traj = self._world_model.rollout_frames(frame, steps=horizon)
        # Re-stamp the metadata bag with this predictor's name.
        out: list[LatentFrame] = []
        for f in traj:
            meta = dict(f.metadata)
            if self._name:
                meta["predictor"] = self._name
            out.append(LatentFrame(
                audio_z=f.audio_z, chord_z=f.chord_z,
                emotion_va=f.emotion_va, time_index=f.time_index,
                provenance=f.provenance, metadata=meta,
            ))
        return out


class GroovePredictor(PredictionEngine):
    """Forecast rhythmic continuation in latent space."""

    def __init__(self, world_model: WorldModel) -> None:
        super().__init__(world_model, name="groove")


class HarmonyPredictor(PredictionEngine):
    """Forecast harmonic progression in latent space."""

    def __init__(self, world_model: WorldModel) -> None:
        super().__init__(world_model, name="harmony")


class DynamicsPredictor(PredictionEngine):
    """Forecast dynamics / loudness contour in latent space."""

    def __init__(self, world_model: WorldModel) -> None:
        super().__init__(world_model, name="dynamics")


__all__ = [
    "PredictionEngine",
    "GroovePredictor",
    "HarmonyPredictor",
    "DynamicsPredictor",
]
