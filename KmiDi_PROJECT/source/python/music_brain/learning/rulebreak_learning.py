"""
Rule-Breaking Learning - Learn which rule breaks work for which emotions/contexts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import json

DEFAULT_STORAGE = Path.home() / ".parrot" / "music_learning" / "rulebreaks"
EXAMPLES_DIR = DEFAULT_STORAGE / "examples"
PROFILES_DIR = DEFAULT_STORAGE / "profiles"


@dataclass
class RuleBreakExample:
    rule_break: str
    emotion: str = "neutral"
    context: Dict = field(default_factory=dict)
    accepted: bool = True
    impact_notes: str = ""  # optional description of effect
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "rule_break": self.rule_break,
            "emotion": self.emotion,
            "context": self.context,
            "accepted": self.accepted,
            "impact_notes": self.impact_notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RuleBreakExample":
        return cls(**data)


@dataclass
class RuleBreakProfile:
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
    def from_dict(cls, data: Dict) -> "RuleBreakProfile":
        return cls(
            name=data.get("name", ""),
            emotion_patterns=data.get("emotion_patterns", {}),
            global_patterns=data.get("global_patterns", {}),
            example_count=data.get("example_count", 0),
        )


class RuleBreakStore:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE
        self.examples_dir = self.storage_dir / "examples"
        self.profiles_dir = self.storage_dir / "profiles"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_example(self, example: RuleBreakExample, name: Optional[str] = None) -> str:
        example_id = name or f"rule_{len(list(self.examples_dir.glob('*.json'))) + 1:04d}"
        path = self.examples_dir / f"{example_id}.json"
        with open(path, "w") as f:
            json.dump(example.to_dict(), f, indent=2)
        return example_id

    def load_example(self, example_id: str) -> Optional[RuleBreakExample]:
        path = self.examples_dir / f"{example_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return RuleBreakExample.from_dict(data)

    def list_examples(self) -> List[str]:
        return sorted([p.stem for p in self.examples_dir.glob("*.json")])

    def save_profile(self, profile: RuleBreakProfile) -> Path:
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return path

    def load_profile(self, name: str) -> Optional[RuleBreakProfile]:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return RuleBreakProfile.from_dict(data)

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])


class RuleBreakLearner:
    def __init__(self):
        pass

    def learn_profile(self, examples: List[RuleBreakExample], name: str = "default") -> RuleBreakProfile:
        if not examples:
            raise ValueError("No rule-break examples provided")

        emotion_groups: Dict[str, List[RuleBreakExample]] = defaultdict(list)
        for ex in examples:
            emotion_groups[ex.emotion.lower()].append(ex)

        emotion_patterns: Dict[str, Dict] = {}
        global_counts = Counter()

        for emotion, group in emotion_groups.items():
            rule_counts = Counter()
            accepted_counts = Counter()

            for ex in group:
                rule_counts[ex.rule_break] += 1
                if ex.accepted:
                    accepted_counts[ex.rule_break] += 1
                global_counts[ex.rule_break] += 1

            emotion_patterns[emotion] = {
                "rule_counts": dict(rule_counts),
                "acceptance": {k: accepted_counts[k] / rule_counts[k] for k in rule_counts},
            }

        global_patterns = {
            "rule_counts": dict(global_counts),
        }

        return RuleBreakProfile(
            name=name,
            emotion_patterns=emotion_patterns,
            global_patterns=global_patterns,
            example_count=len(examples),
        )

    def choose_rule_break(
        self,
        emotion: str,
        profile: RuleBreakProfile,
    ) -> Optional[str]:
        emotion_key = emotion.lower()
        patterns = profile.emotion_patterns.get(emotion_key) or profile.emotion_patterns.get("neutral")
        if not patterns:
            patterns = profile.global_patterns

        rule_counts = patterns.get("rule_counts") or profile.global_patterns.get("rule_counts") or {}
        if not rule_counts:
            return None

        # Choose rule weighted by frequency and acceptance (if available)
        choices = []
        acceptance = patterns.get("acceptance", {})
        for rule, count in rule_counts.items():
            weight = count
            if rule in acceptance:
                weight *= max(0.1, acceptance[rule])
            choices.extend([rule] * max(1, int(weight)))
        if not choices:
            return None
        return choices[np.random.randint(0, len(choices))]


class RuleBreakLearningManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.store = RuleBreakStore(storage_dir)
        self.learner = RuleBreakLearner()

    def add_example(self, example: RuleBreakExample, name: Optional[str] = None) -> str:
        return self.store.add_example(example, name)

    def learn_profile(self, name: str, example_ids: Optional[List[str]] = None) -> RuleBreakProfile:
        if example_ids is None:
            example_ids = self.store.list_examples()
        examples = []
        for ex_id in example_ids:
            ex = self.store.load_example(ex_id)
            if ex:
                examples.append(ex)
        if not examples:
            raise ValueError("No rule-break examples found")
        profile = self.learner.learn_profile(examples, name)
        self.store.save_profile(profile)
        return profile

    def load_profile(self, name: str) -> Optional[RuleBreakProfile]:
        return self.store.load_profile(name)

    def choose(
        self,
        emotion: str,
        profile_name: Optional[str] = None,
    ) -> Optional[str]:
        profile: Optional[RuleBreakProfile] = None
        if profile_name:
            profile = self.load_profile(profile_name)
        if not profile:
            profile = self.store.load_profile("default")
        if not profile:
            return None
        return self.learner.choose_rule_break(emotion, profile)
