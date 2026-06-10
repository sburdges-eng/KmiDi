"""
Deprecated pipeline module. Retained for import-compatibility;
prefer ``music_brain.orchestrator`` (AIOrchestrator + Pipeline) for new code.
"""

from music_brain.pipeline.intent_pipeline import IntentPipeline

__all__ = ["IntentPipeline"]
