"""Phases and task tracking for orchestrator dashboard."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import AIAgent, PhaseStatus


@dataclass
class _Task:
    task_id: str = ""
    assigned_to: Optional[AIAgent] = None
    status: PhaseStatus = PhaseStatus.PENDING
    progress: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "assigned_to": self.assigned_to.value if self.assigned_to else None,
            "status": self.status.value,
            "progress": self.progress,
            "notes": self.notes,
        }


@dataclass
class _Phase:
    phase_id: int = 0
    name: str = ""
    tasks: List[_Task] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks],
        }


class PhaseManager:
    """In-memory phase and task tracking for orchestrator dashboard."""

    def __init__(self) -> None:
        self.phases: List[_Phase] = []
        self._next_phase_id = 0

    def get_phase_summary(self) -> Dict[str, Any]:
        total_tasks = sum(len(p.tasks) for p in self.phases)
        current_id = self.phases[0].phase_id if self.phases else None
        return {
            "phases": [p.to_dict() for p in self.phases],
            "current_phase_id": current_id,
            "total_phases": len(self.phases),
            "total_tasks": total_tasks,
        }

    def get_current_phase(self) -> _Phase:
        if not self.phases:
            return _Phase()
        return self.phases[0]

    def update_task_status(
        self,
        phase_id: int,
        task_id: str,
        status: PhaseStatus,
        progress: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        for phase in self.phases:
            if phase.phase_id != phase_id:
                continue
            for task in phase.tasks:
                if task.task_id == task_id:
                    task.status = status
                    if progress is not None:
                        task.progress = progress
                    if notes is not None:
                        task.notes = notes
                    return
            return
        return

    def assign_task(
        self,
        phase_id: int,
        task_id: str,
        agent: AIAgent,
    ) -> None:
        for phase in self.phases:
            if phase.phase_id != phase_id:
                continue
            for task in phase.tasks:
                if task.task_id == task_id:
                    task.assigned_to = agent
                    return
            return
        return

    def add_phase(self, name: str, task_ids: Optional[List[str]] = None) -> _Phase:
        self._next_phase_id += 1
        phase = _Phase(
            phase_id=self._next_phase_id,
            name=name,
            tasks=[_Task(task_id=tid) for tid in (task_ids or [])],
        )
        self.phases.append(phase)
        return phase


def format_phase_progress(phases: PhaseManager) -> str:
    if not phases.phases:
        return "No phases."
    lines = []
    for p in phases.phases:
        total = len(p.tasks)
        done = sum(1 for t in p.tasks if t.status == PhaseStatus.COMPLETED)
        lines.append(f"Phase {p.phase_id} ({p.name}): {done}/{total} tasks")
    return "\n".join(lines)


def get_next_actions(phases: PhaseManager) -> List[Any]:
    actions = []
    for phase in phases.phases:
        for task in phase.tasks:
            if task.status == PhaseStatus.PENDING or task.status == PhaseStatus.IN_PROGRESS:
                actions.append({
                    "phase_id": phase.phase_id,
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "assigned_to": task.assigned_to.value if task.assigned_to else None,
                })
    return actions
