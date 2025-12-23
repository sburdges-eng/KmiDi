"""
Groove Learning - Learn groove patterns (timing/velocity) from examples.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import json
import numpy as np

DEFAULT_STORAGE = Path.home() / ".parrot" / "music_learning" / "grooves"
EXAMPLES_DIR = DEFAULT_STORAGE / "examples"
PROFILES_DIR = DEFAULT_STORAGE / "profiles"


@dataclass
class GrooveExample:
    """Groove example with timing and velocity patterns."""
    pattern_name: str = "default"
    emotion: str = "neutral"
    genre: str = "straight"
    tempo_bpm: int = 120
    swing_factor: float = 0.0
    timing_offsets_16th: List[float] = field(default_factory=list)
    velocity_curve: List[int] = field(default_factory=list)
    rule_broken: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "pattern_name": self.pattern_name,
            "emotion": self.emotion,
            "genre": self.genre,
            "tempo_bpm": self.tempo_bpm,
            "swing_factor": self.swing_factor,
            "timing_offsets_16th": self.timing_offsets_16th,
            "velocity_curve": self.velocity_curve,
            "rule_broken": self.rule_broken,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GrooveExample":
        return cls(**data)


@dataclass
class GrooveProfile:
    """Learned groove profile."""
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
    def from_dict(cls, data: Dict) -> "GrooveProfile":
        return cls(
            name=data.get("name", ""),
            emotion_patterns=data.get("emotion_patterns", {}),
            global_patterns=data.get("global_patterns", {}),
            example_count=data.get("example_count", 0),
        )


class GrooveStore:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE
        self.examples_dir = self.storage_dir / "examples"
        self.profiles_dir = self.storage_dir / "profiles"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_example(self, example: GrooveExample, name: Optional[str] = None) -> str:
        example_id = name or f"groove_{len(list(self.examples_dir.glob('*.json'))) + 1:04d}"
        path = self.examples_dir / f"{example_id}.json"
        with open(path, "w") as f:
            json.dump(example.to_dict(), f, indent=2)
        return example_id

    def load_example(self, example_id: str) -> Optional[GrooveExample]:
        path = self.examples_dir / f"{example_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return GrooveExample.from_dict(data)

    def list_examples(self) -> List[str]:
        return sorted([p.stem for p in self.examples_dir.glob("*.json")])

    def save_profile(self, profile: GrooveProfile) -> Path:
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return path

    def load_profile(self, name: str) -> Optional[GrooveProfile]:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return GrooveProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])


class GrooveLearner:
    def __init__(self):
        pass

    def learn_profile(self, examples: List[GrooveExample], name: str = "default") -> GrooveProfile:
        if not examples:
            raise ValueError("No groove examples provided")

        emotion_groups: Dict[str, List[GrooveExample]] = defaultdict(list)
        for ex in examples:
            emotion_groups[ex.emotion.lower()].append(ex)

        emotion_patterns: Dict[str, Dict] = {}
        global_timing: List[List[float]] = []
        global_velocity: List[List[int]] = []

        for emotion, group in emotion_groups.items():
            timing_vectors = [ex.timing_offsets_16th for ex in group if ex.timing_offsets_16th]
            velocity_vectors = [ex.velocity_curve for ex in group if ex.velocity_curve]
            swing_factors = [ex.swing_factor for ex in group]
            tempos = [ex.tempo_bpm for ex in group]
            pattern_names = Counter([ex.pattern_name for ex in group])

            if timing_vectors:
                avg_timing = list(np.mean(np.array(timing_vectors), axis=0))
            else:
                avg_timing = []

            if velocity_vectors:
                avg_velocity = list(np.mean(np.array(velocity_vectors), axis=0))
            else:
                avg_velocity = []

            emotion_patterns[emotion] = {
                "avg_timing": avg_timing,
                "avg_velocity": avg_velocity,
                "avg_swing": float(np.mean(swing_factors)) if swing_factors else 0.0,
                "avg_tempo": int(np.mean(tempos)) if tempos else 120,
                "common_patterns": dict(pattern_names),
            }

            if avg_timing:
                global_timing.append(avg_timing)
            if avg_velocity:
                global_velocity.append(avg_velocity)

        global_patterns = {
            "avg_timing": list(np.mean(np.array(global_timing), axis=0)) if global_timing else [],
            "avg_velocity": list(np.mean(np.array(global_velocity), axis=0)) if global_velocity else [],
        }

        return GrooveProfile(
            name=name,
            emotion_patterns=emotion_patterns,
            global_patterns=global_patterns,
            example_count=len(examples),
        )

    def generate_groove(
        self,
        emotion: str,
        profile: GrooveProfile,
        tempo: Optional[int] = None,
        genre: str = "straight"
    ) -> Dict:
        emotion_key = emotion.lower()
        patterns = profile.emotion_patterns.get(emotion_key) or profile.emotion_patterns.get("neutral")
        if not patterns:
            patterns = profile.global_patterns

        timing = patterns.get("avg_timing") or profile.global_patterns.get("avg_timing") or [0.0] * 16
        velocity = patterns.get("avg_velocity") or profile.global_patterns.get("avg_velocity") or [80] * 16
        swing = patterns.get("avg_swing", 0.0)
        tempo = tempo or int(patterns.get("avg_tempo", 120))

        return {
            "pattern_name": genre,
            "tempo_bpm": tempo,
            "swing_factor": swing,
            "timing_offsets_16th": timing,
            "velocity_curve": velocity,
        }


class GrooveLearningManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.store = GrooveStore(storage_dir)
        self.learner = GrooveLearner()

    def add_example(self, example: GrooveExample, name: Optional[str] = None) -> str:
        return self.store.add_example(example, name)

    def learn_profile(self, name: str, example_ids: Optional[List[str]] = None) -> GrooveProfile:
        if example_ids is None:
            example_ids = self.store.list_examples()
        examples = []
        for ex_id in example_ids:
            ex = self.store.load_example(ex_id)
            if ex:
                examples.append(ex)
        if not examples:
            raise ValueError("No groove examples found")
        profile = self.learner.learn_profile(examples, name)
        self.store.save_profile(profile)
        return profile

    def load_profile(self, name: str) -> Optional[GrooveProfile]:
        return self.store.load_profile(name)

    def generate(
        self,
        emotion: str,
        profile_name: Optional[str] = None,
        tempo: Optional[int] = None,
        genre: str = "straight"
    ) -> Dict:
        profile: Optional[GrooveProfile] = None
        if profile_name:
            profile = self.load_profile(profile_name)
        if not profile:
            profile = self.store.load_profile("default")
        if not profile:
            # Default groove
            return {
                "pattern_name": genre,
                "tempo_bpm": tempo or 120,
                "swing_factor": 0.0,
                "timing_offsets_16th": [0.0] * 16,
                "velocity_curve": [80] * 16,
            }
        return self.learner.generate_groove(emotion, profile, tempo, genre)
