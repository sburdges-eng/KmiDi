"""
Bass Learning - Learn bass line patterns from examples.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import json
import numpy as np

DEFAULT_STORAGE = Path.home() / ".parrot" / "music_learning" / "bass"
EXAMPLES_DIR = DEFAULT_STORAGE / "examples"
PROFILES_DIR = DEFAULT_STORAGE / "profiles"


@dataclass
class BassExample:
    """Bass line example with context."""
    notes: List[int]  # MIDI notes
    pattern: str = "root_only"
    emotion: str = "neutral"
    genre: str = "general"
    key: str = "C"
    mode: str = "major"
    tempo_bpm: int = 120
    accepted: bool = True
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "notes": self.notes,
            "pattern": self.pattern,
            "emotion": self.emotion,
            "genre": self.genre,
            "key": self.key,
            "mode": self.mode,
            "tempo_bpm": self.tempo_bpm,
            "accepted": self.accepted,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BassExample":
        return cls(**data)


@dataclass
class BassProfile:
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
    def from_dict(cls, data: Dict) -> "BassProfile":
        return cls(
            name=data.get("name", ""),
            emotion_patterns=data.get("emotion_patterns", {}),
            global_patterns=data.get("global_patterns", {}),
            example_count=data.get("example_count", 0),
        )


class BassStore:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE
        self.examples_dir = self.storage_dir / "examples"
        self.profiles_dir = self.storage_dir / "profiles"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_example(self, example: BassExample, name: Optional[str] = None) -> str:
        example_id = name or f"bass_{len(list(self.examples_dir.glob('*.json'))) + 1:04d}"
        path = self.examples_dir / f"{example_id}.json"
        with open(path, "w") as f:
            json.dump(example.to_dict(), f, indent=2)
        return example_id

    def load_example(self, example_id: str) -> Optional[BassExample]:
        path = self.examples_dir / f"{example_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return BassExample.from_dict(data)

    def list_examples(self) -> List[str]:
        return sorted([p.stem for p in self.examples_dir.glob("*.json")])

    def save_profile(self, profile: BassProfile) -> Path:
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return path

    def load_profile(self, name: str) -> Optional[BassProfile]:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return BassProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])


class BassLearner:
    def __init__(self):
        pass

    def _interval_pattern(self, notes: List[int]) -> List[int]:
        if len(notes) < 2:
            return []
        return [notes[i + 1] - notes[i] for i in range(len(notes) - 1)]

    def learn_profile(self, examples: List[BassExample], name: str = "default") -> BassProfile:
        if not examples:
            raise ValueError("No bass examples provided")

        emotion_groups: Dict[str, List[BassExample]] = defaultdict(list)
        for ex in examples:
            emotion_groups[ex.emotion.lower()].append(ex)

        emotion_patterns: Dict[str, Dict] = {}
        global_intervals = Counter()
        global_notes = Counter()

        for emotion, group in emotion_groups.items():
            interval_counter = Counter()
            note_counter = Counter()
            patterns = Counter()
            lengths = []

            for ex in group:
                interval_pattern = tuple(self._interval_pattern(ex.notes))
                interval_counter[interval_pattern] += 1
                note_counter.update(ex.notes)
                patterns[ex.pattern] += 1
                lengths.append(len(ex.notes))

                global_intervals[interval_pattern] += 1
                global_notes.update(ex.notes)

            emotion_patterns[emotion] = {
                "interval_patterns": dict(interval_counter),
                "note_frequencies": dict(note_counter),
                "pattern_counts": dict(patterns),
                "avg_length": float(np.mean(lengths)) if lengths else 8.0,
            }

        global_patterns = {
            "interval_patterns": dict(global_intervals),
            "note_frequencies": dict(global_notes),
        }

        return BassProfile(
            name=name,
            emotion_patterns=emotion_patterns,
            global_patterns=global_patterns,
            example_count=len(examples),
        )

    def generate_bass(
        self,
        emotion: str,
        profile: BassProfile,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major",
        base_note: int = 36  # C2
    ) -> List[int]:
        emotion_key = emotion.lower()
        patterns = profile.emotion_patterns.get(emotion_key) or profile.emotion_patterns.get("neutral")
        if not patterns:
            patterns = profile.global_patterns

        avg_length = int(patterns.get("avg_length", 8))
        length = length or avg_length

        note_freqs = patterns.get("note_frequencies", {}) or profile.global_patterns.get("note_frequencies", {})
        interval_patterns = patterns.get("interval_patterns", {}) or profile.global_patterns.get("interval_patterns", {})

        notes = []
        if note_freqs:
            counter = Counter(note_freqs)
            most_common = counter.most_common(1)
            if most_common:
                start_note = int(most_common[0][0])
            else:
                start_note = base_note
        else:
            start_note = base_note
        notes.append(start_note)

        interval_list = []
        for pattern, count in interval_patterns.items():
            interval_list.extend([pattern] * max(1, int(count)))

        for i in range(length - 1):
            if interval_list:
                pattern = interval_list[np.random.randint(0, len(interval_list))]
                if pattern:
                    step = pattern[min(i, len(pattern) - 1)]
                else:
                    step = 0
            else:
                step = np.random.choice([-5, -2, 0, 2, 5])
            notes.append(int(notes[-1] + step))

        return notes


class BassLearningManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.store = BassStore(storage_dir)
        self.learner = BassLearner()

    def add_example(self, example: BassExample, name: Optional[str] = None) -> str:
        return self.store.add_example(example, name)

    def learn_profile(self, name: str, example_ids: Optional[List[str]] = None) -> BassProfile:
        if example_ids is None:
            example_ids = self.store.list_examples()
        examples = []
        for ex_id in example_ids:
            ex = self.store.load_example(ex_id)
            if ex:
                examples.append(ex)
        if not examples:
            raise ValueError("No bass examples found")
        profile = self.learner.learn_profile(examples, name)
        self.store.save_profile(profile)
        return profile

    def load_profile(self, name: str) -> Optional[BassProfile]:
        return self.store.load_profile(name)

    def generate(
        self,
        emotion: str,
        profile_name: Optional[str] = None,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major"
    ) -> List[int]:
        profile: Optional[BassProfile] = None
        if profile_name:
            profile = self.load_profile(profile_name)
        if not profile:
            profile = self.store.load_profile("default")
        if not profile:
            # Default simple pattern: root-only bass
            base_note = 36  # C2
            length = length or 8
            return [base_note] * length
        return self.learner.generate_bass(emotion, profile, length, key, mode)
