"""
Adaptive module for music brain.

Provides adaptive generation that learns from user feedback.
"""

from .adaptive_generator import (
    AdaptiveGenerator,
    GenerationAttempt,
)

from .feedback_processor import FeedbackProcessor

__all__ = [
    "AdaptiveGenerator",
    "GenerationAttempt",
    "FeedbackProcessor",
]
