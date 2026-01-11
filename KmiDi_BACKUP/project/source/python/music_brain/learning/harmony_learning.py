"""
Harmony Learning - Example-based chord progression storage and adaptive generation.

Stores chord progression examples with emotional context, learns common patterns
and rule-break effectiveness, and generates adaptive harmonic content.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import json
import numpy as np

DEFAULT_STORAGE = Path.home() / ".parrot" / "music_learning" / "harmonies"
EXAMPLES_DIR = DEFAULT_STORAGE / "examples"
PROFILES_DIR = DEFAULT_STORAGE / "profiles"


@dataclass
class ChordExample:
    """Single chord progression example with context."""
    progression: List[str]
    roman_numerals: List[str]
    emotion: str = "neutral"
    key: str = "C"
    mode: str = "major"
    rule_breaks: List[str] = field(default_factory=list)
    accepted: bool = True
    voice_leading: List[List[int]] = field(default_factory=list)  # List of MIDI note lists
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "progression": self.progression,
            "roman_numerals": self.roman_numerals,
            "emotion": self.emotion,
            "key": self.key,
            "mode": self.mode,
            "rule_breaks": self.rule_breaks,
            "accepted": self.accepted,
            "voice_leading": self.voice_leading,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChordExample":
        return cls(**data)


@dataclass
class HarmonyProfile:
    """Learned harmony profile with progression statistics."""
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
    def from_dict(cls, data: Dict) -> "HarmonyProfile":
        return cls(
            name=data.get("name", ""),
            emotion_patterns=data.get("emotion_patterns", {}),
            global_patterns=data.get("global_patterns", {}),
            example_count=data.get("example_count", 0),
        )


class HarmonyStore:
    """Stores chord progression examples and profiles."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE
        self.examples_dir = self.storage_dir / "examples"
        self.profiles_dir = self.storage_dir / "profiles"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_example(self, example: ChordExample, name: Optional[str] = None) -> str:
        example_id = name or f"harmony_{len(list(self.examples_dir.glob('*.json'))) + 1:04d}"
        path = self.examples_dir / f"{example_id}.json"
        with open(path, "w") as f:
            json.dump(example.to_dict(), f, indent=2)
        return example_id

    def load_example(self, example_id: str) -> Optional[ChordExample]:
        path = self.examples_dir / f"{example_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ChordExample.from_dict(data)

    def list_examples(self) -> List[str]:
        return sorted([p.stem for p in self.examples_dir.glob("*.json")])

    def save_profile(self, profile: HarmonyProfile) -> Path:
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return path

    def load_profile(self, name: str) -> Optional[HarmonyProfile]:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return HarmonyProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])


class HarmonyLearner:
    """Learns chord progression patterns from examples."""

    def __init__(self):
        pass

    def _progression_to_transitions(self, roman_numerals: List[str]) -> List[Tuple[str, str]]:
        transitions = []
        for i in range(len(roman_numerals) - 1):
            transitions.append((roman_numerals[i], roman_numerals[i + 1]))
        return transitions

    def learn_profile(self, examples: List[ChordExample], name: str = "default") -> HarmonyProfile:
        if not examples:
            raise ValueError("No chord progression examples provided")

        emotion_groups: Dict[str, List[ChordExample]] = defaultdict(list)
        for ex in examples:
            emotion_groups[ex.emotion.lower()].append(ex)

        emotion_patterns: Dict[str, Dict] = {}
        global_progressions = Counter()
        global_transitions = Counter()

        for emotion, group in emotion_groups.items():
            progression_counter = Counter()
            transition_counter = Counter()
            lengths = []

            for ex in group:
                progression = tuple(ex.roman_numerals) if ex.roman_numerals else tuple(ex.progression)
                progression_counter[progression] += 1
                transitions = self._progression_to_transitions(ex.roman_numerals or ex.progression)
                transition_counter.update(transitions)
                lengths.append(len(ex.progression))

                global_progressions[progression] += 1
                global_transitions.update(transitions)

            emotion_patterns[emotion] = {
                "progressions": dict(progression_counter),
                "transitions": dict(transition_counter),
                "avg_length": float(np.mean(lengths)) if lengths else 4.0,
            }

        global_patterns = {
            "progressions": dict(global_progressions),
            "transitions": dict(global_transitions),
        }

        return HarmonyProfile(
            name=name,
            emotion_patterns=emotion_patterns,
            global_patterns=global_patterns,
            example_count=len(examples),
        )

    def generate_progression(
        self,
        emotion: str,
        profile: HarmonyProfile,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major"
    ) -> List[str]:
        emotion_key = emotion.lower()
        patterns = profile.emotion_patterns.get(emotion_key) or profile.emotion_patterns.get("neutral")
        if not patterns:
            patterns = profile.global_patterns

        avg_length = int(patterns.get("avg_length", 4))
        length = length or avg_length

        progressions = patterns.get("progressions", {})
        if not progressions and profile.global_patterns.get("progressions"):
            progressions = profile.global_patterns["progressions"]

        if progressions:
            # Choose a progression weighted by frequency
            progression_choices = []
            for prog, count in progressions.items():
                progression_choices.extend([prog] * max(1, int(count)))
            chosen = progression_choices[np.random.randint(0, len(progression_choices))]
            chords = list(chosen)
        else:
            # Default I-V-vi-IV pattern
            chords = ["I", "V", "vi", "IV"]

        # Convert roman numerals to simple chords relative to key
        # Simple mapping for major/minor keys
        numeral_to_chord_major = {
            "I": "C",
            "ii": "Dm",
            "iii": "Em",
            "IV": "F",
            "V": "G",
            "vi": "Am",
            "vii°": "Bdim",
        }
        numeral_to_chord_minor = {
            "i": "Am",
            "ii°": "Bdim",
            "III": "C",
            "iv": "Dm",
            "v": "Em",
            "VI": "F",
            "VII": "G",
        }

        mapping = numeral_to_chord_major if mode.lower() == "major" else numeral_to_chord_minor
        result_chords = [mapping.get(ch, ch) for ch in chords]
        return result_chords


class HarmonyLearningManager:
    """High-level API for harmony learning and generation."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.store = HarmonyStore(storage_dir)
        self.learner = HarmonyLearner()

    def add_example(self, example: ChordExample, name: Optional[str] = None) -> str:
        return self.store.add_example(example, name)

    def learn_profile(self, name: str, example_ids: Optional[List[str]] = None) -> HarmonyProfile:
        if example_ids is None:
            example_ids = self.store.list_examples()
        examples = []
        for ex_id in example_ids:
            ex = self.store.load_example(ex_id)
            if ex:
                examples.append(ex)
        if not examples:
            raise ValueError("No harmony examples found")
        profile = self.learner.learn_profile(examples, name)
        self.store.save_profile(profile)
        return profile

    def load_profile(self, name: str) -> Optional[HarmonyProfile]:
        return self.store.load_profile(name)

    def generate(
        self,
        emotion: str,
        profile_name: Optional[str] = None,
        length: Optional[int] = None,
        key: str = "C",
        mode: str = "major"
    ) -> List[str]:
        profile: Optional[HarmonyProfile] = None
        if profile_name:
            profile = self.load_profile(profile_name)
        if not profile:
            profile = self.store.load_profile("default")
        if not profile:
            # Default fallback progression
            return ["C", "G", "Am", "F"]
        return self.learner.generate_progression(emotion, profile, length, key, mode)
