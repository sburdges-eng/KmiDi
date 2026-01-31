"""Debug / logging helpers — stubs for orchestrator."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class DebugCategory(Enum):
    AI_COMMUNICATION = "ai_communication"
    PERFORMANCE = "performance"
    ERROR = "error"


@dataclass
class DebugEvent:
    timestamp: str = ""
    message: str = ""
    category: str = ""


class _Debug:
    def __init__(self) -> None:
        self.events: List[Any] = []

    def info(self, category: DebugCategory, message: str) -> None:
        self.events.append(DebugEvent(message=message, category=category.value))

    def get_errors(self, limit: int = 25) -> List[DebugEvent]:
        return []

    def get_performance_report(self) -> Dict[str, Any]:
        return {}


_debug_instance: _Debug | None = None


def get_debug() -> _Debug:
    global _debug_instance
    if _debug_instance is None:
        _debug_instance = _Debug()
    return _debug_instance
