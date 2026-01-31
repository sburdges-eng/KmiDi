"""
Kelly Project - External Framework Integrations

This package provides integration layers for external music generation frameworks:
- Google Magenta (MusicVAE, GrooVAE, Performance RNN)
- Stem-JEPA (emotion-conditioned stem retrieval)

See individual modules for detailed documentation.
"""

from .magenta_integration import (
    VADState,
    NoteSequenceConverter,
    EmotionAttributeVectors,
    VADLatentMapper,
    HumanizationParams,
    EmotionHumanizer,
    RealtimeMIDIProcessor,
    create_kelly_magenta_pipeline,
)

from .stem_jepa_integration import (
    StemCategory,
    EmotionStemAffinity,
    FiLMConditioningVector,
    FiLMConditioningGenerator,
    StemCompatibilityScorer,
    StemRecommendation,
    EmotionStemPipeline,
)

__all__ = [
    # Magenta integration
    'VADState',
    'NoteSequenceConverter',
    'EmotionAttributeVectors',
    'VADLatentMapper',
    'HumanizationParams',
    'EmotionHumanizer',
    'RealtimeMIDIProcessor',
    'create_kelly_magenta_pipeline',
    # Stem-JEPA integration
    'StemCategory',
    'EmotionStemAffinity',
    'FiLMConditioningVector',
    'FiLMConditioningGenerator',
    'StemCompatibilityScorer',
    'StemRecommendation',
    'EmotionStemPipeline',
]
