"""Preset model for capturing and recalling application state.

Presets capture intent and configuration, not raw audio or MIDI data.
Presets are immutable once saved.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.integrations.dynamics_integration import EmotionState


@dataclass
class Preset:
    """Immutable preset capturing application state.

    Presets include:
    - EmotionState snapshot
    - Intent Schema (Phase 0-2)
    - ML enable/disable states
    - ML visualization toggles
    - Teaching mode state
    - Trust settings (optional)

    Presets do NOT include:
    - Audio buffers
    - Live engine state
    - UI layout (handled separately)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    # Captured state blocks
    emotion_state: Optional[EmotionState] = None
    intent_schema: Optional[CompleteSongIntent] = None
    # model enable/disable
    ml_settings: Dict[str, bool] = field(default_factory=dict)
    # visualization toggles
    ml_visualization: Dict[str, bool] = field(default_factory=dict)
    teaching_mode: bool = False
    # optional trust levels
    trust_settings: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "emotion_state": None,
            "intent_schema": None,
            "ml_settings": self.ml_settings,
            "ml_visualization": self.ml_visualization,
            "teaching_mode": self.teaching_mode,
            "trust_settings": self.trust_settings,
        }

        if self.emotion_state:
            result["emotion_state"] = {
                "valence": self.emotion_state.valence,
                "arousal": self.emotion_state.arousal,
                "dominance": self.emotion_state.dominance,
                "intensity": self.emotion_state.intensity,
            }

        if self.intent_schema:
            result["intent_schema"] = self.intent_schema.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        """Create preset from dictionary."""
        timestamp = (
            datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        )
        preset = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            timestamp=timestamp,
            version=data.get("version", "1.0.0"),
            ml_settings=data.get("ml_settings", {}),
            ml_visualization=data.get("ml_visualization", {}),
            teaching_mode=data.get("teaching_mode", False),
            trust_settings=data.get("trust_settings"),
        )

        # Deserialize emotion state
        if "emotion_state" in data and data["emotion_state"]:
            emo_data = data["emotion_state"]
            preset.emotion_state = EmotionState(
                valence=emo_data.get("valence", 0.0),
                arousal=emo_data.get("arousal", 0.5),
                dominance=emo_data.get("dominance", 0.5),
                intensity=emo_data.get("intensity", 0.5),
            )

        # Deserialize intent schema
        if "intent_schema" in data and data["intent_schema"]:
            preset.intent_schema = CompleteSongIntent.from_dict(data["intent_schema"])

        return preset

    def validate(self) -> list[str]:
        """Validate preset data. Returns list of issues (empty if valid)."""
        issues = []

        if not self.name:
            issues.append("Preset name is required")

        if self.version != "1.0.0":
            issues.append(f"Unknown preset version: {self.version}")

        # Safe to ignore unknown fields - they will be preserved but not used

        return issues
