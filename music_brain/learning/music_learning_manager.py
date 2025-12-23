"""
Music Learning Manager - Unified interface for all music learning systems.
"""

from pathlib import Path
from typing import Dict, List, Optional

from music_brain.learning.melody_learning import (
    MelodyExample,
    MelodyLearningManager,
    MelodyProfile,
)
from music_brain.learning.harmony_learning import (
    ChordExample,
    HarmonyLearningManager,
    HarmonyProfile,
)
from music_brain.learning.groove_learning import (
    GrooveExample,
    GrooveLearningManager,
    GrooveProfile,
)
from music_brain.learning.bass_learning import (
    BassExample,
    BassLearningManager,
    BassProfile,
)
from music_brain.learning.arrangement_learning import (
    ArrangementExample,
    ArrangementLearningManager,
    ArrangementProfile,
)
from music_brain.learning.expression_learning import (
    ExpressionExample,
    ExpressionLearningManager,
    ExpressionProfile,
)
from music_brain.learning.rulebreak_learning import (
    RuleBreakExample,
    RuleBreakLearningManager,
    RuleBreakProfile,
)


class MusicLearningManager:
    """
    Unified API to add examples, learn profiles, and generate using learned patterns
    across melody, harmony, groove, bass, arrangement, expression, and rule-breaking.
    """

    def __init__(self, storage_root: Optional[Path] = None):
        self.storage_root = storage_root
        self.melody = MelodyLearningManager(self._subdir("melodies"))
        self.harmony = HarmonyLearningManager(self._subdir("harmonies"))
        self.groove = GrooveLearningManager(self._subdir("grooves"))
        self.bass = BassLearningManager(self._subdir("bass"))
        self.arrangement = ArrangementLearningManager(self._subdir("arrangements"))
        self.expression = ExpressionLearningManager(self._subdir("expression"))
        self.rulebreak = RuleBreakLearningManager(self._subdir("rulebreaks"))

    def _subdir(self, name: str) -> Optional[Path]:
        if self.storage_root:
            path = Path(self.storage_root) / name
            path.mkdir(parents=True, exist_ok=True)
            return path
        return None

    # Melody
    def add_melody_example(self, example: MelodyExample, name: Optional[str] = None) -> str:
        return self.melody.add_example(example, name)

    def learn_melody_profile(self, name: str, ids: Optional[List[str]] = None) -> MelodyProfile:
        return self.melody.learn_profile(name, ids)

    def generate_melody(self, emotion: str, profile: Optional[str] = None, length: Optional[int] = None) -> List[int]:
        return self.melody.generate(emotion, profile, length)

    # Harmony
    def add_chord_example(self, example: ChordExample, name: Optional[str] = None) -> str:
        return self.harmony.add_example(example, name)

    def learn_harmony_profile(self, name: str, ids: Optional[List[str]] = None) -> HarmonyProfile:
        return self.harmony.learn_profile(name, ids)

    def generate_chords(
        self,
        emotion: str,
        profile: Optional[str] = None,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major"
    ) -> List[str]:
        return self.harmony.generate(emotion, profile, length, key, mode)

    # Groove
    def add_groove_example(self, example: GrooveExample, name: Optional[str] = None) -> str:
        return self.groove.add_example(example, name)

    def learn_groove_profile(self, name: str, ids: Optional[List[str]] = None) -> GrooveProfile:
        return self.groove.learn_profile(name, ids)

    def generate_groove(self, emotion: str, profile: Optional[str] = None, tempo: Optional[int] = None, genre: str = "straight") -> Dict:
        return self.groove.generate(emotion, profile, tempo, genre)

    # Bass
    def add_bass_example(self, example: BassExample, name: Optional[str] = None) -> str:
        return self.bass.add_example(example, name)

    def learn_bass_profile(self, name: str, ids: Optional[List[str]] = None) -> BassProfile:
        return self.bass.learn_profile(name, ids)

    def generate_bass(
        self,
        emotion: str,
        profile: Optional[str] = None,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major"
    ) -> List[int]:
        return self.bass.generate(emotion, profile, length, key, mode)

    # Arrangement
    def add_arrangement_example(self, example: ArrangementExample, name: Optional[str] = None) -> str:
        return self.arrangement.add_example(example, name)

    def learn_arrangement_profile(self, name: str, ids: Optional[List[str]] = None) -> ArrangementProfile:
        return self.arrangement.learn_profile(name, ids)

    def generate_arrangement(self, emotion: str, profile: Optional[str] = None, length: Optional[int] = None, genre: str = "general") -> Dict:
        return self.arrangement.generate(emotion, profile, length, genre)

    # Expression
    def add_expression_example(self, example: ExpressionExample, name: Optional[str] = None) -> str:
        return self.expression.add_example(example, name)

    def learn_expression_profile(self, name: str, ids: Optional[List[str]] = None) -> ExpressionProfile:
        return self.expression.learn_profile(name, ids)

    def generate_expression(self, emotion: str, profile: Optional[str] = None, length: Optional[int] = None, instrument: str = "general") -> Dict:
        return self.expression.generate(emotion, profile, length, instrument)

    # Rule breaking
    def add_rulebreak_example(self, example: RuleBreakExample, name: Optional[str] = None) -> str:
        return self.rulebreak.add_example(example, name)

    def learn_rulebreak_profile(self, name: str, ids: Optional[List[str]] = None) -> RuleBreakProfile:
        return self.rulebreak.learn_profile(name, ids)

    def choose_rulebreak(self, emotion: str, profile: Optional[str] = None) -> Optional[str]:
        return self.rulebreak.choose(emotion, profile)
