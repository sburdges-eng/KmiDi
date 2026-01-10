"""
Intelligence module for music brain.

Provides intelligent suggestions and context-aware analysis.
Includes NLM (Neural Language Model) integrations for:
- Lyrics generation
- Intent parsing
- Creative suggestions
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
from .ollama_bridge import (
    OllamaBridge,
    OllamaConfig,
    create_ollama_bridge,
    generate_lyrics,
    parse_intent,
)
from .brain_controller import (
    BrainController,
    BrainConfig,
    create_brain_controller,
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
    "BrainController",
    "BrainConfig",
    "create_brain_controller",
]
