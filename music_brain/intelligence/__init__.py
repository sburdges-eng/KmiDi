"""
Intelligence module for music brain.

Provides intelligent suggestions and context-aware analysis.
"""

from .suggestion_engine import (
    SuggestionEngine,
    Suggestion,
    SuggestionType,
    SuggestionConfidence,
)

from .context_analyzer import (
    ContextAnalyzer,
    MusicalContext,
)
from .onnx_llm import (
    OnnxGenAILLM,
    OnnxLLMConfig,
)

__all__ = [
    "SuggestionEngine",
    "Suggestion",
    "SuggestionType",
    "SuggestionConfidence",
    "ContextAnalyzer",
    "MusicalContext",
    "OnnxGenAILLM",
    "OnnxLLMConfig",
]
