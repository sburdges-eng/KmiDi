"""
Bridge between example-based melody learning and ML/rule melody generation.

Prefer learned patterns when a profile exists; otherwise delegate to
``MLMelodyGenerator`` (MelodyTransformer when available, else rule-based).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from music_brain.session.ml_melody_generator import (
    DEFAULT_TRAITS,
    EMOTION_MELODIC_TRAITS,
    GeneratedMelody,
    MLMelodyGenerator,
    MelodyConfig,
)
from music_brain.session.melody_generator import AdaptiveMelodyGenerator
from music_brain.learning.melody_learning import MelodyLearningManager


def create_adaptive_melody_stack(
    learning_storage: Optional[Path] = None,
    melody_config: Optional[MelodyConfig] = None,
) -> Tuple[AdaptiveMelodyGenerator, MLMelodyGenerator]:
    """Build adaptive (learned) + ML generators sharing the same learning storage."""
    mgr = MelodyLearningManager(learning_storage)
    adaptive = AdaptiveMelodyGenerator(learning_manager=mgr)
    cfg = melody_config or MelodyConfig()
    ml_gen = MLMelodyGenerator(cfg)
    return adaptive, ml_gen


def generate_melody_with_learning(
    emotion: str,
    *,
    learning_profile: Optional[str] = None,
    learning_storage: Optional[Path] = None,
    length: Optional[int] = None,
    key: str = "C",
    mode: str = "major",
    melody_config: Optional[MelodyConfig] = None,
) -> GeneratedMelody:
    """
    Generate a melody: learned profile first (if any), then ML, then rules.

    ``learning_profile`` selects a learned profile under ``learning_storage``
    (defaults to ``~/.parrot/music_learning/melodies`` when storage is None).
    """
    n = length or (melody_config.length_notes if melody_config else 8)
    cfg = melody_config or MelodyConfig(key=key, mode=mode, length_notes=n)
    adaptive, ml_gen = create_adaptive_melody_stack(learning_storage, cfg)

    ml_notes: Optional[List[int]] = None
    if ml_gen.is_ml_available():
        try:
            ml_out = ml_gen.generate(emotion, n, key, mode)
            ml_notes = ml_out.notes
        except Exception:
            ml_notes = None

    notes = adaptive.generate(
        emotion,
        length=n,
        profile_name=learning_profile,
        ml_melody=ml_notes,
    )

    traits = EMOTION_MELODIC_TRAITS.get(emotion.lower(), DEFAULT_TRAITS)
    durations = ml_gen._generate_durations(len(notes), traits)  # noqa: SLF001
    velocities = ml_gen._generate_velocities(len(notes), emotion)  # noqa: SLF001

    if learning_profile:
        method = "learned_plus_ml_fallback"
    elif ml_gen.is_ml_available():
        method = "ml_model"
    else:
        method = "rule_based"

    return GeneratedMelody(
        notes=notes,
        durations=durations,
        velocities=velocities,
        key=key,
        mode=mode,
        emotion=emotion,
        method=method,
        confidence=0.8 if learning_profile else 0.6,
    )
