"""MCP workstation domain models — stubs for orchestrator."""

from enum import Enum
from typing import Any, Dict, Optional


class AIAgent(Enum):
    """Agent type identifiers."""
    LLM = "llm"
    MIDI = "midi"
    IMAGE = "image"
    AUDIO = "audio"


class PhaseStatus(Enum):
    """Task/phase status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
