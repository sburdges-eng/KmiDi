"""
Melody Learning - Example-based melody pattern storage and adaptive generation.

Provides storage for melody examples, pattern extraction, simple statistical
learning, and adaptive melody generation that can be combined with existing
ML models (e.g., MelodyTransformer) as a fallback.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import json
import numpy as np

DEFAULT_STORAGE = Path.home() / ".parrot" / "music_learning" / "melodies"
EXAMPLES_DIR = DEFAULT_STORAGE / "examples"
PROFILES_DIR = DEFAULT_STORAGE / "profiles"


@dataclass
class MelodyExample:
    """Single melody example with context and feedback."""
    melody: List[int]  # MIDI notes
    emotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.5
    key: str = "C"
    mode: str = "major"
    tempo: int = 120
    accepted: bool = True
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "melody": self.melody,
            "emotion": self.emotion,
            "valence": self.valence,
            "arousal": self.arousal,
            "key": self.key,
            "mode": self.mode,
            "tempo": self.tempo,
            "accepted": self.accepted,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MelodyExample":
        return cls(**data)


@dataclass
class MelodyProfile:
    """Learned melody profile containing pattern statistics."""
    name: str
    emotion_patterns: Dict[str, Dict]  # emotion -> stats
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
    def from_dict(cls, data: Dict) -> "MelodyProfile":
        return cls(
            name=data.get("name", ""),
            emotion_patterns=data.get("emotion_patterns", {}),
            global_patterns=data.get("global_patterns", {}),
            example_count=data.get("example_count", 0),
        )


class MelodyStore:
    """Stores melody examples and profiles on disk."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE
        self.examples_dir = self.storage_dir / "examples"
        self.profiles_dir = self.storage_dir / "profiles"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_example(self, example: MelodyExample, name: Optional[str] = None) -> str:
        """Save a melody example to disk and return its id."""
        example_id = name or f"melody_{len(list(self.examples_dir.glob('*.json'))) + 1:04d}"
        path = self.examples_dir / f"{example_id}.json"
        with open(path, "w") as f:
            json.dump(example.to_dict(), f, indent=2)
        return example_id

    def load_example(self, example_id: str) -> Optional[MelodyExample]:
        path = self.examples_dir / f"{example_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return MelodyExample.from_dict(data)

    def list_examples(self) -> List[str]:
        return sorted([p.stem for p in self.examples_dir.glob("*.json")])

    def save_profile(self, profile: MelodyProfile) -> Path:
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return path

    def load_profile(self, name: str) -> Optional[MelodyProfile]:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return MelodyProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])


class MelodyLearner:
    """Learns melody patterns from examples."""

    def __init__(self):
        pass

    def _extract_interval_pattern(self, melody: List[int]) -> List[int]:
        if len(melody) < 2:
            return []
        return [melody[i + 1] - melody[i] for i in range(len(melody) - 1)]

    def _extract_contour(self, melody: List[int]) -> List[int]:
        # -1 down, 0 same, 1 up
        contour = []
        for i in range(1, len(melody)):
            diff = melody[i] - melody[i - 1]
            if diff > 0:
                contour.append(1)
            elif diff < 0:
                contour.append(-1)
            else:
                contour.append(0)
        return contour

    def learn_profile(self, examples: List[MelodyExample], name: str = "default") -> MelodyProfile:
        if not examples:
            raise ValueError("No melody examples provided")

        # Aggregate by emotion
        emotion_groups: Dict[str, List[MelodyExample]] = defaultdict(list)
        for ex in examples:
            emotion_groups[ex.emotion.lower()].append(ex)

        emotion_patterns: Dict[str, Dict] = {}
        global_intervals: Counter = Counter()
        global_contours: Counter = Counter()
        global_notes: Counter = Counter()

        for emotion, group in emotion_groups.items():
            interval_counter = Counter()
            contour_counter = Counter()
            note_counter = Counter()
            lengths = []
            tempos = []

            for ex in group:
                interval_pattern = tuple(self._extract_interval_pattern(ex.melody))
                contour_pattern = tuple(self._extract_contour(ex.melody))
                interval_counter[interval_pattern] += 1
                contour_counter[contour_pattern] += 1
                note_counter.update(ex.melody)
                lengths.append(len(ex.melody))
                tempos.append(ex.tempo)

                # Update global stats
                global_intervals[interval_pattern] += 1
                global_contours[contour_pattern] += 1
                global_notes.update(ex.melody)

            emotion_patterns[emotion] = {
                "interval_patterns": dict(interval_counter),
                "contour_patterns": dict(contour_counter),
                "note_frequencies": dict(note_counter),
                "avg_length": float(np.mean(lengths)) if lengths else 8.0,
                "avg_tempo": int(np.mean(tempos)) if tempos else 120,
            }

        global_patterns = {
            "interval_patterns": dict(global_intervals),
            "contour_patterns": dict(global_contours),
            "note_frequencies": dict(global_notes),
        }

        return MelodyProfile(
            name=name,
            emotion_patterns=emotion_patterns,
            global_patterns=global_patterns,
            example_count=len(examples),
        )

    def generate_melody(
        self,
        emotion: str,
        profile: MelodyProfile,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major"
    ) -> List[int]:
        """Generate a melody using learned patterns (basic statistical sampling)."""
        emotion_key = emotion.lower()
        patterns = profile.emotion_patterns.get(emotion_key) or profile.emotion_patterns.get("neutral")
        if not patterns:
            patterns = profile.global_patterns

        avg_length = int(patterns.get("avg_length", 8))
        length = length or avg_length

        note_freqs = patterns.get("note_frequencies", {})
        if not note_freqs and profile.global_patterns.get("note_frequencies"):
            note_freqs = profile.global_patterns["note_frequencies"]

        # Simple sampling: start from most common note, apply common interval patterns
        notes = []
        if note_freqs:
            counter = Counter(note_freqs)
            most_common = counter.most_common(1)
            if most_common:
                most_common_note = int(most_common[0][0])
            else:
                most_common_note = 60  # Default to C4
        else:
            # Default to C4
            most_common_note = 60
        notes.append(most_common_note)

        interval_patterns = patterns.get("interval_patterns", {})
        if not interval_patterns and profile.global_patterns.get("interval_patterns"):
            interval_patterns = profile.global_patterns["interval_patterns"]

        # Flatten interval patterns by frequency
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
                step = np.random.choice([-2, -1, 0, 1, 2])

            next_note = notes[-1] + step
            notes.append(int(next_note))

        return notes


class MelodyLearningManager:
    """High-level API for melody learning and generation."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.store = MelodyStore(storage_dir)
        self.learner = MelodyLearner()

    def add_example(self, example: MelodyExample, name: Optional[str] = None) -> str:
        return self.store.add_example(example, name)

    def learn_profile(self, name: str, example_ids: Optional[List[str]] = None) -> MelodyProfile:
        if example_ids is None:
            example_ids = self.store.list_examples()
        examples = []
        for ex_id in example_ids:
            ex = self.store.load_example(ex_id)
            if ex:
                examples.append(ex)
        if not examples:
            raise ValueError("No melody examples found")
        profile = self.learner.learn_profile(examples, name)
        self.store.save_profile(profile)
        return profile

    def load_profile(self, name: str) -> Optional[MelodyProfile]:
        return self.store.load_profile(name)

    def generate(
        self,
        emotion: str,
        profile_name: Optional[str] = None,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major"
    ) -> List[int]:
        profile: Optional[MelodyProfile] = None
        if profile_name:
            profile = self.load_profile(profile_name)
        if not profile:
            # Try default profile
            profile = self.store.load_profile("default")
        if not profile:
            # No profiles, return simple default scale pattern
            base_note = 60  # C4
            scale = [0, 2, 4, 5, 7, 9, 11, 12]
            length = length or 8
            return [base_note + scale[i % len(scale)] for i in range(length)]

        return self.learner.generate_melody(emotion, profile, length, key, mode)
