"""
Harmony helpers with optional example-based learning.

Rule-based and ML-specific harmony live under ``session.intent_processor`` and
orchestrator processors; this module exposes a single entry point that prefers
learned progressions when a profile is configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def generate_chords_with_learning(
    emotion: str,
    *,
    key: str = "C",
    mode: str = "major",
    length: Optional[int] = None,
    learning_profile: Optional[str] = None,
    learning_storage: Optional[Path] = None,
) -> List[str]:
    """
    Return chord symbols for ``emotion``, using a learned profile when present.

    Falls back to a simple diatonic loop when no profile is loaded.
    """
    from music_brain.learning.harmony_learning import HarmonyLearningManager

    mgr = HarmonyLearningManager(learning_storage)
    if learning_profile:
        return mgr.generate(emotion, learning_profile, length, key, mode)
    return mgr.generate(emotion, None, length, key, mode)
