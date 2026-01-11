"""
Lightweight AI specialization metadata for CLI display.

This stub provides the TaskType enum and helper used by music_brain.cli without
requiring the full original implementation.
"""

from enum import Enum
from typing import List


class TaskType(Enum):
    ARCHITECTURE = "architecture"
    DATA = "data"
    TRAINING = "training"
    PRODUCT = "product"


def print_ai_summary(tasks: List["TaskType"] = None):
    """Simple summary printer placeholder."""
    tasks = tasks or list(TaskType)
    for task in tasks:
        print(f"- {task.value}")


__all__ = ["TaskType", "print_ai_summary"]
