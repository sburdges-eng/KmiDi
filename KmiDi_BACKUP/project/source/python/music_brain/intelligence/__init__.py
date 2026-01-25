"""
Intelligence module for music brain.

Provides intelligent suggestions and context-aware analysis.
Includes NLM (Neural Language Model) integrations for:
- Lyrics generation
- Intent parsing
- Creative suggestions
"""

from .context_analyzer import (
    ContextAnalyzer,
    MusicalContext,
)
from .ollama_bridge import (
    OllamaBridge,
    OllamaConfig,
    create_ollama_bridge,
    generate_lyrics,
    parse_intent,
)
from .onnx_llm import (
    OnnxGenAILLM,
    OnnxLLMConfig,
)
from .suggestion_engine import (
    Suggestion,
    SuggestionConfidence,
    SuggestionEngine,
    SuggestionType,
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
    "OllamaBridge",
    "OllamaConfig",
    "create_ollama_bridge",
    "generate_lyrics",
    "parse_intent",
]
