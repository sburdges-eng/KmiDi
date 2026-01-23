"""
Expression Learning - Learn dynamics/expression patterns from examples.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import json
import numpy as np

DEFAULT_STORAGE = Path.home() / ".parrot" / "music_learning" / "expression"
EXAMPLES_DIR = DEFAULT_STORAGE / "examples"
PROFILES_DIR = DEFAULT_STORAGE / "profiles"


@dataclass
class ExpressionExample:
    """Expression example capturing velocity/dynamics curves."""
    velocity_curve: List[int]  # e.g., per-16th velocities
    emotion: str = "neutral"
    instrument: str = "general"
    tempo_bpm: int = 120
    accepted: bool = True
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "velocity_curve": self.velocity_curve,
            "emotion": self.emotion,
            "instrument": self.instrument,
            "tempo_bpm": self.tempo_bpm,
            "accepted": self.accepted,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExpressionExample":
        return cls(**data)


@dataclass
class ExpressionProfile:
    name: str
    emotion_patterns: Dict[str, Dict]
    global_patterns: Dict
    example_count: int

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "emotion_patterns": self.emotion_patterns,
            "global_patterns": self.global_patterns,
            "example_count": self.example_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExpressionProfile":
        return cls(
            name=data.get("name", ""),
            emotion_patterns=data.get("emotion_patterns", {}),
            global_patterns=data.get("global_patterns", {}),
            example_count=data.get("example_count", 0),
        )


class ExpressionStore:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE
        self.examples_dir = self.storage_dir / "examples"
        self.profiles_dir = self.storage_dir / "profiles"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_example(self, example: ExpressionExample, name: Optional[str] = None) -> str:
        example_id = name or f"expression_{len(list(self.examples_dir.glob('*.json'))) + 1:04d}"
        path = self.examples_dir / f"{example_id}.json"
        with open(path, "w") as f:
            json.dump(example.to_dict(), f, indent=2)
        return example_id

    def load_example(self, example_id: str) -> Optional[ExpressionExample]:
        path = self.examples_dir / f"{example_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ExpressionExample.from_dict(data)

    def list_examples(self) -> List[str]:
        return sorted([p.stem for p in self.examples_dir.glob("*.json")])

    def save_profile(self, profile: ExpressionProfile) -> Path:
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return path

    def load_profile(self, name: str) -> Optional[ExpressionProfile]:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ExpressionProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])


class ExpressionLearner:
    def __init__(self):
        pass

    def learn_profile(self, examples: List[ExpressionExample], name: str = "default") -> ExpressionProfile:
        if not examples:
            raise ValueError("No expression examples provided")

        emotion_groups: Dict[str, List[ExpressionExample]] = defaultdict(list)
        for ex in examples:
            emotion_groups[ex.emotion.lower()].append(ex)

        emotion_patterns: Dict[str, Dict] = {}
        global_velocity = []

        for emotion, group in emotion_groups.items():
            velocity_vectors = [ex.velocity_curve for ex in group if ex.velocity_curve]
            tempos = [ex.tempo_bpm for ex in group]
            instruments = Counter([ex.instrument for ex in group])

            if velocity_vectors:
                avg_velocity = list(np.mean(np.array(velocity_vectors), axis=0))
            else:
                avg_velocity = []

            emotion_patterns[emotion] = {
                "avg_velocity": avg_velocity,
                "avg_tempo": int(np.mean(tempos)) if tempos else 120,
                "instrument_counts": dict(instruments),
            }

            if avg_velocity:
                global_velocity.append(avg_velocity)

        global_patterns = {
            "avg_velocity": list(np.mean(np.array(global_velocity), axis=0)) if global_velocity else [],
        }

        return ExpressionProfile(
            name=name,
            emotion_patterns=emotion_patterns,
            global_patterns=global_patterns,
            example_count=len(examples),
        )

    def generate_expression(
        self,
        emotion: str,
        profile: ExpressionProfile,
        length: Optional[int] = None,
        instrument: str = "general"
    ) -> Dict:
        emotion_key = emotion.lower()
        patterns = profile.emotion_patterns.get(emotion_key) or profile.emotion_patterns.get("neutral")
        if not patterns:
            patterns = profile.global_patterns

        velocity = patterns.get("avg_velocity") or profile.global_patterns.get("avg_velocity") or [80] * (length or 16)
        if length and len(velocity) < length:
            velocity = (velocity * (length // len(velocity) + 1))[:length]

        return {
            "velocity_curve": velocity,
            "instrument": instrument,
        }


class ExpressionLearningManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.store = ExpressionStore(storage_dir)
        self.learner = ExpressionLearner()

    def add_example(self, example: ExpressionExample, name: Optional[str] = None) -> str:
        return self.store.add_example(example, name)

    def learn_profile(self, name: str, example_ids: Optional[List[str]] = None) -> ExpressionProfile:
        if example_ids is None:
            example_ids = self.store.list_examples()
        examples = []
        for ex_id in example_ids:
            ex = self.store.load_example(ex_id)
            if ex:
                examples.append(ex)
        if not examples:
            raise ValueError("No expression examples found")
        profile = self.learner.learn_profile(examples, name)
        self.store.save_profile(profile)
        return profile

    def load_profile(self, name: str) -> Optional[ExpressionProfile]:
        return self.store.load_profile(name)

    def generate(
        self,
        emotion: str,
        profile_name: Optional[str] = None,
        length: Optional[int] = None,
        instrument: str = "general"
    ) -> Dict:
        profile: Optional[ExpressionProfile] = None
        if profile_name:
            profile = self.load_profile(profile_name)
        if not profile:
            profile = self.store.load_profile("default")
        if not profile:
            return {
                "velocity_curve": [80] * (length or 16),
                "instrument": instrument,
            }
        return self.learner.generate_expression(emotion, profile, length, instrument)
