"""AI specializations — stubs for orchestrator."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from .models import AIAgent


class TaskType(Enum):
    LLM = "llm"
    MIDI = "midi"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class AgentCapabilities:
    display_name: str = ""
    description: str = ""
    strengths: Dict[Any, str] = None
    proposal_categories: List[Any] = None
    best_languages: List[str] = None
    special_abilities: List[str] = None
    limitations: List[str] = None
    recommended_for: List[str] = None

    def __post_init__(self) -> None:
        if self.strengths is None:
            self.strengths = {}
        if self.proposal_categories is None:
            self.proposal_categories = []
        if self.best_languages is None:
            self.best_languages = []
        if self.special_abilities is None:
            self.special_abilities = []
        if self.limitations is None:
            self.limitations = []
        if self.recommended_for is None:
            self.recommended_for = []


def get_capabilities(agent: AIAgent) -> AgentCapabilities:
    return AgentCapabilities(display_name=agent.value, description="")


def suggest_task_assignment(tasks: List[Tuple[str, TaskType]]) -> Dict[str, AIAgent]:
    return {}
