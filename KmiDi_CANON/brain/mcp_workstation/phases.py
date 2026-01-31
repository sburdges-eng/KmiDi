"""Phases — stubs for orchestrator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import AIAgent, PhaseStatus


@dataclass
class _Task:
    assigned_to: Optional[AIAgent] = None


@dataclass
class _Phase:
    tasks: List[_Task] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {}


class PhaseManager:
    def __init__(self) -> None:
        self.phases: List[_Phase] = []

    def get_phase_summary(self) -> Dict[str, Any]:
        return {}

    def get_current_phase(self) -> _Phase:
        return _Phase()

    def update_task_status(
        self,
        phase_id: int,
        task_id: str,
        status: PhaseStatus,
        progress: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        pass

    def assign_task(
        self,
        phase_id: int,
        task_id: str,
        agent: AIAgent,
    ) -> None:
        pass


def format_phase_progress(phases: PhaseManager) -> str:
    return ""


def get_next_actions(phases: PhaseManager) -> List[Any]:
    return []
