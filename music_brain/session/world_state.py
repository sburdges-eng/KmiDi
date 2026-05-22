"""WorldState — persistent musical session state with history.

A long-lived state object for a creative session: current key, tempo,
section, emotion + an append-only history log of every transition. Used
by orchestrators that need to know "what was just happening" to bias the
next decision toward continuity, and by retrieval code that wants to
key into the user's recent journey.

History entries are stored as ``{"kind": ..., "from": ..., "to": ...}``
dicts so they round-trip cleanly through JSON without custom decoders.

Stdlib only.

Goal-list ties:
  - Persistent latent world-state modeling (direct)
  - Longitudinal user modeling (state-tracking prerequisite)
  - Arrangement continuity preservation (section history)
  - Creative companion interaction model (session memory the companion
    can refer back to)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class WorldState:
    """Persistent session state. Updates record to ``history``."""

    key: str = ""
    tempo: int = 0
    current_section: Optional[str] = None
    current_emotion: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ----------------------------------------------------------------- mutate
    def update_section(self, section: Optional[str]) -> None:
        """Set current section; appends to history unless unchanged."""
        if section == self.current_section:
            return
        self.history.append(
            {"kind": "section", "from": self.current_section, "to": section}
        )
        self.current_section = section

    def update_emotion(self, emotion: Optional[str]) -> None:
        """Set current emotion; appends to history unless unchanged."""
        if emotion == self.current_emotion:
            return
        self.history.append(
            {"kind": "emotion", "from": self.current_emotion, "to": emotion}
        )
        self.current_emotion = emotion

    def reset_history(self) -> None:
        """Clear the history log but keep the current state values."""
        self.history.clear()

    # ----------------------------------------------------------------- query
    def section_transition_count(self) -> int:
        return sum(1 for e in self.history if e["kind"] == "section")

    def recent_sections(self, n: int) -> List[str]:
        """Last ``n`` section ``to`` values in chronological order."""
        sections = [e["to"] for e in self.history if e["kind"] == "section"]
        return sections[-n:]

    # --------------------------------------------------------- serialisation
    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "tempo": self.tempo,
                "current_section": self.current_section,
                "current_emotion": self.current_emotion,
                "history": self.history,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "WorldState":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from e
        return cls(
            key=data.get("key", ""),
            tempo=int(data.get("tempo", 0)),
            current_section=data.get("current_section"),
            current_emotion=data.get("current_emotion"),
            history=list(data.get("history", [])),
        )

    def save(self, path: Union[Path, str]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[Path, str]) -> "WorldState":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
