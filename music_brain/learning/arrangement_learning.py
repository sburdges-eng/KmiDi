"""
Arrangement Learning - Learn arrangement structures from examples.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import json
import numpy as np

DEFAULT_STORAGE = Path.home() / ".parrot" / "music_learning" / "arrangements"
EXAMPLES_DIR = DEFAULT_STORAGE / "examples"
PROFILES_DIR = DEFAULT_STORAGE / "profiles"


@dataclass
class ArrangementExample:
    """Arrangement example capturing sections and instrumentation."""
    sections: List[str]  # e.g., ["verse", "chorus", "verse", "bridge", "chorus"]
    energy_curve: List[float] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    emotion: str = "neutral"
    genre: str = "general"
    tempo_bpm: int = 120
    accepted: bool = True
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "sections": self.sections,
            "energy_curve": self.energy_curve,
            "instruments": self.instruments,
            "emotion": self.emotion,
            "genre": self.genre,
            "tempo_bpm": self.tempo_bpm,
            "accepted": self.accepted,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ArrangementExample":
        return cls(**data)


@dataclass
class ArrangementProfile:
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
    def from_dict(cls, data: Dict) -> "ArrangementProfile":
        return cls(
            name=data.get("name", ""),
            emotion_patterns=data.get("emotion_patterns", {}),
            global_patterns=data.get("global_patterns", {}),
            example_count=data.get("example_count", 0),
        )


class ArrangementStore:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE
        self.examples_dir = self.storage_dir / "examples"
        self.profiles_dir = self.storage_dir / "profiles"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_example(self, example: ArrangementExample, name: Optional[str] = None) -> str:
        example_id = name or f"arr_{len(list(self.examples_dir.glob('*.json'))) + 1:04d}"
        path = self.examples_dir / f"{example_id}.json"
        with open(path, "w") as f:
            json.dump(example.to_dict(), f, indent=2)
        return example_id

    def load_example(self, example_id: str) -> Optional[ArrangementExample]:
        path = self.examples_dir / f"{example_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ArrangementExample.from_dict(data)

    def list_examples(self) -> List[str]:
        return sorted([p.stem for p in self.examples_dir.glob("*.json")])

    def save_profile(self, profile: ArrangementProfile) -> Path:
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return path

    def load_profile(self, name: str) -> Optional[ArrangementProfile]:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ArrangementProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])


class ArrangementLearner:
    def __init__(self):
        pass

    def learn_profile(self, examples: List[ArrangementExample], name: str = "default") -> ArrangementProfile:
        if not examples:
            raise ValueError("No arrangement examples provided")

        emotion_groups: Dict[str, List[ArrangementExample]] = defaultdict(list)
        for ex in examples:
            emotion_groups[ex.emotion.lower()].append(ex)

        emotion_patterns: Dict[str, Dict] = {}
        global_sections = Counter()
        global_transitions = Counter()
        global_instruments = Counter()

        for emotion, group in emotion_groups.items():
            section_counter = Counter()
            transition_counter = Counter()
            instrument_counter = Counter()
            lengths = []

            for ex in group:
                section_counter.update(ex.sections)
                transitions = [(ex.sections[i], ex.sections[i + 1]) for i in range(len(ex.sections) - 1)]
                transition_counter.update(transitions)
                instrument_counter.update(ex.instruments)
                lengths.append(len(ex.sections))

                global_sections.update(ex.sections)
                global_transitions.update(transitions)
                global_instruments.update(ex.instruments)

            emotion_patterns[emotion] = {
                "section_counts": dict(section_counter),
                "transition_counts": dict(transition_counter),
                "instrument_counts": dict(instrument_counter),
                "avg_length": float(np.mean(lengths)) if lengths else 5.0,
            }

        global_patterns = {
            "section_counts": dict(global_sections),
            "transition_counts": dict(global_transitions),
            "instrument_counts": dict(global_instruments),
        }

        return ArrangementProfile(
            name=name,
            emotion_patterns=emotion_patterns,
            global_patterns=global_patterns,
            example_count=len(examples),
        )

    def generate_arrangement(
        self,
        emotion: str,
        profile: ArrangementProfile,
        length: Optional[int] = None,
        genre: str = "general"
    ) -> Dict:
        emotion_key = emotion.lower()
        patterns = profile.emotion_patterns.get(emotion_key) or profile.emotion_patterns.get("neutral")
        if not patterns:
            patterns = profile.global_patterns

        section_counts = patterns.get("section_counts") or profile.global_patterns.get("section_counts") or {}
        transition_counts = patterns.get("transition_counts") or profile.global_patterns.get("transition_counts") or {}
        instrument_counts = patterns.get("instrument_counts") or profile.global_patterns.get("instrument_counts") or {}

        length = length or int(patterns.get("avg_length", 5))

        # Choose sections based on frequencies
        if section_counts:
            sections = []
            choices = []
            for sec, cnt in section_counts.items():
                choices.extend([sec] * max(1, int(cnt)))
            for i in range(length):
                sections.append(np.random.choice(choices))
        else:
            sections = ["verse", "chorus", "verse", "bridge", "chorus"][:length]

        # Build transitions based on counts (simple Markov)
        for i in range(len(sections) - 1):
            current = sections[i]
            possible = {k[1]: v for k, v in transition_counts.items() if k[0] == current}
            if possible:
                options = []
                for nxt, cnt in possible.items():
                    options.extend([nxt] * max(1, int(cnt)))
                sections[i + 1] = np.random.choice(options)

        # Choose instruments by frequency
        instruments = []
        if instrument_counts:
            choices = []
            for inst, cnt in instrument_counts.items():
                choices.extend([inst] * max(1, int(cnt)))
            instruments = list(np.unique(choices))
        else:
            instruments = ["drums", "bass", "piano", "guitar"]

        return {
            "sections": sections,
            "instruments": instruments,
        }


class ArrangementLearningManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.store = ArrangementStore(storage_dir)
        self.learner = ArrangementLearner()

    def add_example(self, example: ArrangementExample, name: Optional[str] = None) -> str:
        return self.store.add_example(example, name)

    def learn_profile(self, name: str, example_ids: Optional[List[str]] = None) -> ArrangementProfile:
        if example_ids is None:
            example_ids = self.store.list_examples()
        examples = []
        for ex_id in example_ids:
            ex = self.store.load_example(ex_id)
            if ex:
                examples.append(ex)
        if not examples:
            raise ValueError("No arrangement examples found")
        profile = self.learner.learn_profile(examples, name)
        self.store.save_profile(profile)
        return profile

    def load_profile(self, name: str) -> Optional[ArrangementProfile]:
        return self.store.load_profile(name)

    def generate(
        self,
        emotion: str,
        profile_name: Optional[str] = None,
        length: Optional[int] = None,
        genre: str = "general"
    ) -> Dict:
        profile: Optional[ArrangementProfile] = None
        if profile_name:
            profile = self.load_profile(profile_name)
        if not profile:
            profile = self.store.load_profile("default")
        if not profile:
            return {
                "sections": ["verse", "chorus", "verse", "bridge", "chorus"][: length or 5],
                "instruments": ["drums", "bass", "piano", "guitar"],
            }
        return self.learner.generate_arrangement(emotion, profile, length, genre)
