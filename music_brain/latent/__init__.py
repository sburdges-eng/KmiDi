"""Latent control core: contract + decoding + RAG + composer surface.

The ``music_brain.latent`` package bundles the cross-encoder
``LatentFrame`` envelope plus everything built on top of it across
the four waves:

  Wave 1 — foundation: ``LatentFrame``, regularizers, conditioning
    bridge, ordered streaming hook.
  Wave 2 — decoding: KV-cache, constraint sampling, sub-16ms
    jitter-bounded scheduler, incremental decode loop.
  Wave 3 — closed-loop personalization: emotion-conditioned retrieval
    over the existing VectorStore, per-user EMA bias model, realtime
    rollback ring with KV-cache-aware truncation.
  Wave 4 — structure & wire format: online motif clusterer,
    hierarchical section/bar/beat planner, MIDI 2.0 UMP packer +
    validator, generation scope isolation.
  Wave 5 — composer surface: domain-specific prediction engines
    (groove/harmony/dynamics), multimodal fusion + stem bundles, smooth
    emotion-trajectory planner, ``CompanionSession`` human-in-the-loop
    orchestrator.
"""

from music_brain.latent.companion import CompanionSession, HumanFeedback
from music_brain.latent.conditioning_bridge import (
    INTENT_SLOTS,
    ConditioningProjection,
)
from music_brain.latent.decoding import (
    DecodeConfig,
    greedy_argmax,
    sample_with_constraints,
)
from music_brain.latent.emotion_trajectory import (
    EmotionTrajectory,
    EmotionWaypoint,
)
from music_brain.latent.feedback import FeedbackEvent, UserModel
from music_brain.latent.fusion import MultimodalFusion, StemBundle
from music_brain.latent.incremental_decode import (
    DecodeStep,
    incremental_decode,
)
from music_brain.latent.kv_cache import KVCache
from music_brain.latent.latent_frame import LatentFrame
from music_brain.latent.motif import MotifRecurrence, MotifTracker
from music_brain.latent.predictors import (
    DynamicsPredictor,
    GroovePredictor,
    HarmonyPredictor,
    PredictionEngine,
)
from music_brain.latent.regularization import L2NormProjection, VarianceFloor
from music_brain.latent.retrieval import LatentMemory, RecallHit, pool_audio_z
from music_brain.latent.rollback import RollbackRing
from music_brain.latent.scheduler import (
    JitterBoundedScheduler,
    StepDecision,
    StepTimer,
)
from music_brain.latent.scope import GenerationScope, ScopeViolation
from music_brain.latent.section_planner import (
    BarPosition,
    Section,
    SectionPlan,
    SectionPlanner,
    SectionRole,
)
from music_brain.latent.streaming import stream_decode
from music_brain.latent.ump import (
    UMPMessage,
    UMPMessageType,
    pack_midi1_note_on,
    pack_midi2_note_on,
    validate_ump,
)

__all__ = [
    # Wave 1
    "LatentFrame",
    "L2NormProjection",
    "VarianceFloor",
    "ConditioningProjection",
    "INTENT_SLOTS",
    "stream_decode",
    # Wave 2
    "KVCache",
    "DecodeConfig",
    "greedy_argmax",
    "sample_with_constraints",
    "JitterBoundedScheduler",
    "StepDecision",
    "StepTimer",
    "DecodeStep",
    "incremental_decode",
    # Wave 3
    "LatentMemory",
    "RecallHit",
    "pool_audio_z",
    "UserModel",
    "FeedbackEvent",
    "RollbackRing",
    # Wave 4
    "MotifTracker",
    "MotifRecurrence",
    "SectionPlanner",
    "SectionPlan",
    "Section",
    "SectionRole",
    "BarPosition",
    "UMPMessage",
    "UMPMessageType",
    "pack_midi1_note_on",
    "pack_midi2_note_on",
    "validate_ump",
    "GenerationScope",
    "ScopeViolation",
    # Wave 5
    "PredictionEngine",
    "GroovePredictor",
    "HarmonyPredictor",
    "DynamicsPredictor",
    "MultimodalFusion",
    "StemBundle",
    "EmotionTrajectory",
    "EmotionWaypoint",
    "CompanionSession",
    "HumanFeedback",
]
