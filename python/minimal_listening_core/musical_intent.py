"""MusicalIntent: creation, update/merge, serialization, comparison (distance)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class IntentAspect(Enum):
    HARMONIC = "harmonic"
    RHYTHMIC = "rhythmic"
    DYNAMIC = "dynamic"
    TEMPO = "tempo"


@dataclass
class IntentComponent:
    aspect: IntentAspect
    magnitude: float
    reason: str = ""


@dataclass
class MusicalIntent:
    """Opaque intent representation. No generation logic."""

    _components: Dict[IntentAspect, IntentComponent] = field(default_factory=dict)

    @classmethod
    def create(cls) -> MusicalIntent:
        return cls()

    @classmethod
    def create_with_components(
        cls, components: List[IntentComponent]
    ) -> MusicalIntent:
        intent = cls()
        for c in components:
            intent._components[c.aspect] = c
        return intent

    def update(self, other: MusicalIntent) -> None:
        for aspect, comp in other._components.items():
            self._components[aspect] = comp

    def merge(self, src: MusicalIntent) -> None:
        self.update(src)

    def serialize(self) -> str:
        obj = {
            aspect.value: comp.magnitude
            for aspect, comp in self._components.items()
        }
        return json.dumps(obj)

    @classmethod
    def deserialize(cls, json_str: str) -> MusicalIntent:
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return cls()
        intent = cls()
        valid_aspects = {e.value for e in IntentAspect}
        for k, v in obj.items():
            if isinstance(v, (int, float)) and k in valid_aspects:
                aspect = IntentAspect(k)
                intent._components[aspect] = IntentComponent(
                    aspect=aspect, magnitude=float(v), reason="deserialized"
                )
        return intent

    def distance(self, other: MusicalIntent) -> float:
        dist = 0.0
        for aspect, comp in self._components.items():
            other_mag = (
                other._components[aspect].magnitude
                if aspect in other._components
                else 0.0
            )
            dist += abs(comp.magnitude - other_mag)
        return dist

    def has_aspect(self, aspect: IntentAspect) -> bool:
        return aspect in self._components

    def get_magnitude(self, aspect: IntentAspect) -> float:
        if aspect in self._components:
            return self._components[aspect].magnitude
        return 0.0
