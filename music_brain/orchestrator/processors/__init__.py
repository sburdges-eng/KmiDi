"""
Processing Modules for AI Orchestrator.

This package contains processor implementations for the AI pipeline:
- BaseProcessor: Abstract base class for all processors
- HarmonyProcessor: Harmony generation processor
- GrooveProcessor: Groove application processor
- IntentProcessor: Intent processing and validation
- MelodyProcessor: Melody generation with optional example-based learning

Usage:
    from music_brain.orchestrator.processors import (
        HarmonyProcessor,
        GrooveProcessor,
        IntentProcessor,
        MelodyProcessor,
    )

    # Create processor instances
    harmony_proc = HarmonyProcessor()
    groove_proc = GrooveProcessor()
    intent_proc = IntentProcessor()
    melody_proc = MelodyProcessor()

    # Use in pipeline
    pipeline.add_stage("harmony", harmony_proc)
"""

from music_brain.orchestrator.processors.base import (
    BaseProcessor,
    PassthroughProcessor,
)
from music_brain.orchestrator.processors.harmony import HarmonyProcessor
from music_brain.orchestrator.processors.groove import GrooveProcessor
from music_brain.orchestrator.processors.intent import IntentProcessor
from music_brain.orchestrator.processors.melody import MelodyProcessor

__all__ = [
    "BaseProcessor",
    "PassthroughProcessor",
    "HarmonyProcessor",
    "GrooveProcessor",
    "IntentProcessor",
    "MelodyProcessor",
]
