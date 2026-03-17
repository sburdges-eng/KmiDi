"""
Deterministic 3-stage intent pipeline: Normalize → Validate → Expand.

Single canonical path for converting UI/API request payloads into
CompleteSongIntent. Restores imagery_texture and emotion_map in a
structured way; validated schema values always override inferred.
"""

from music_brain.pipeline.intent_pipeline import IntentPipeline

__all__ = ["IntentPipeline"]
