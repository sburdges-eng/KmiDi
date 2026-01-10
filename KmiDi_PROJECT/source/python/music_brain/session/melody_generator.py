"""
Adaptive Melody Generator - Uses learned patterns (example-based) with ML fallback.
"""

from typing import List, Optional
from music_brain.learning.melody_learning import (
    MelodyExample,
    MelodyLearningManager,
    MelodyProfile,
)


class AdaptiveMelodyGenerator:
    """
    Generates melodies using learned patterns; can fall back to ML outputs if provided.
    """

    def __init__(self, learning_manager: Optional[MelodyLearningManager] = None):
        self.learning_manager = learning_manager or MelodyLearningManager()

    def generate(
        self,
        emotion: str,
        length: Optional[int] = None,
        profile_name: Optional[str] = None,
        ml_melody: Optional[List[int]] = None,
    ) -> List[int]:
        # Try learned melody first
        learned = self.learning_manager.generate(emotion, profile_name, length)
        if learned:
            return learned
        # Fallback to ML melody if provided
        if ml_melody:
            return ml_melody
        # Final fallback: simple scale
        base_note = 60
        scale = [0, 2, 4, 5, 7, 9, 11, 12]
        length = length or 8
        return [base_note + scale[i % len(scale)] for i in range(length)]

    def add_example(self, example: MelodyExample, name: Optional[str] = None) -> str:
        return self.learning_manager.add_example(example, name)

    def learn_profile(self, name: str, ids: Optional[List[str]] = None) -> MelodyProfile:
        return self.learning_manager.learn_profile(name, ids)

    def load_profile(self, name: str) -> Optional[MelodyProfile]:
        return self.learning_manager.load_profile(name)
