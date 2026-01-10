"""
Lightweight model and status definitions used by the music_brain CLI.

These are minimal stand-ins to satisfy CLI imports and tests. They capture the
enums and data structures the CLI expects without pulling in heavier deps.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ProposalCategory(Enum):
    ARCHITECTURE = "architecture"
    DATA = "data"
    TRAINING = "training"
    PRODUCT = "product"


class ProposalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PhaseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class AIAgent(Enum):
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    GITHUB_COPILOT = "github_copilot"


@dataclass
class Proposal:
    id: str
    title: str
    description: str
    category: ProposalCategory
    status: ProposalStatus = ProposalStatus.PENDING
    votes_for: int = 0
    votes_against: int = 0
    priority: int = 5
    phase: int = 1


def get_default_agents() -> List[AIAgent]:
    return [AIAgent.CLAUDE, AIAgent.CHATGPT, AIAgent.GEMINI, AIAgent.GITHUB_COPILOT]


__all__ = [
    "AIAgent",
    "ProposalCategory",
    "ProposalStatus",
    "PhaseStatus",
    "Proposal",
    "get_default_agents",
]
