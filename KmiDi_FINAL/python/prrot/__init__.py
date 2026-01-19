"""
PRROT/PARROT Voice-Instrument Compiler - Tier B (Python ML Worker)

Offline analysis and reasoning layer for deep phoneme alignment,
speaker-specific articulation analysis, timbre embeddings, and prosody analysis.

16GB Mac Safety:
- Single worker process enforcement
- Memory monitoring (8GB worker limit, 10GB system reserve)
- Q4 quantization required
- Automatic cleanup and memory reclamation
"""

__version__ = "0.1.0"
__all__ = [
    "PRROTWorker",
    "VoiceProfileExtractionJob",
    "ControlDataGenerationJob",
    "ArticulationAnalysisJob",
]
