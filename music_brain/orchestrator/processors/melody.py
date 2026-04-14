"""
Melody processor for learning-aware pipelines.

Reads ``learning_profiles`` / ``learning_storage_root`` from ``ExecutionContext``
shared data (see ``music_brain.orchestrator.pipeline``) or from input ``params``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from music_brain.orchestrator.processors.base import BaseProcessor
from music_brain.orchestrator.interfaces import (
    ProcessorConfig,
    ProcessorResult,
    ExecutionContext,
)
from music_brain.integrations.ml_music_brain_bridge import generate_melody_with_learning


@dataclass
class MelodyInput:
    """Input for melody generation."""
    emotion: str = "neutral"
    key: str = "C"
    mode: str = "major"
    length: int = 8
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MelodyOutput:
    """Serializable melody stage output."""
    notes: List[int]
    durations: List[float]
    velocities: List[int]
    key: str
    mode: str
    emotion: str
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notes": self.notes,
            "durations": self.durations,
            "velocities": self.velocities,
            "key": self.key,
            "mode": self.mode,
            "emotion": self.emotion,
            "method": self.method,
        }


class MelodyProcessor(BaseProcessor):
    """Generate melodies with optional example-based learning."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__(config or ProcessorConfig(name="melody"))

    async def _validate_impl(self, input_data: Any) -> bool:
        if isinstance(input_data, MelodyInput):
            return bool(input_data.emotion)
        if isinstance(input_data, dict):
            return bool(input_data.get("emotion"))
        return hasattr(input_data, "intent")

    async def _process_impl(
        self,
        input_data: Any,
        context: ExecutionContext,
    ) -> ProcessorResult:
        try:
            if isinstance(input_data, dict):
                inp = MelodyInput(
                    emotion=input_data.get("emotion", "neutral"),
                    key=input_data.get("key", "C"),
                    mode=input_data.get("mode", "major"),
                    length=int(input_data.get("length", 8)),
                    params=input_data.get("params", {}),
                )
            elif isinstance(input_data, MelodyInput):
                inp = input_data
            elif hasattr(input_data, "intent"):
                inp = MelodyInput(
                    emotion=context.get_shared("emotion", "neutral"),
                    key=context.get_shared("key", "C"),
                    mode=context.get_shared("mode", "major"),
                    length=int(context.get_shared("melody_length", 8)),
                )
            else:
                return ProcessorResult(
                    success=False,
                    error=f"Invalid input type: {type(input_data).__name__}",
                )

            profiles = context.get_shared("learning_profiles") or {}
            storage = context.get_shared("learning_storage_root")
            profile = (
                inp.params.get("learning_melody_profile")
                or profiles.get("melody")
            )
            path_storage = None
            if storage is not None:
                from pathlib import Path
                path_storage = Path(storage) / "melodies"

            out = generate_melody_with_learning(
                inp.emotion,
                learning_profile=profile,
                learning_storage=path_storage,
                length=inp.length,
                key=inp.key,
                mode=inp.mode,
            )

            melody_output = MelodyOutput(
                notes=out.notes,
                durations=out.durations,
                velocities=out.velocities,
                key=out.key,
                mode=out.mode,
                emotion=out.emotion,
                method=out.method,
            )
            context.set_shared("melody", melody_output.to_dict())
            return ProcessorResult(
                success=True,
                data=melody_output,
                metadata={"method": out.method, "profile": profile},
            )
        except Exception as e:
            self._logger.error("Melody generation failed: %s", e)
            return ProcessorResult(success=False, error=str(e))
