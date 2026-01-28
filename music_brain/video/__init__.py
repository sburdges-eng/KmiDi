"""
music_brain.video - Music Video Generation Module

Provides stubs for future music video generation using Unreal Engine and Jespa.
This module maps emotional intent and musical parameters to visual parameters
for automated music video creation.

Core components:
- VideoGenerator: Main orchestrator for video generation
- UnrealBridge: Integration with Unreal Engine for rendering
- JespaConnector: Integration with Jespa for video processing
- EmotionVisualMapper: Maps emotions to visual parameters
- SceneComposer: Composes video scenes from musical structure

Usage:
    from music_brain.video import VideoGenerator
    
    gen = VideoGenerator()
    video = gen.generate_from_emotion(emotion="grief", music_path="song.wav")
"""

__version__ = "0.1.0"

from .video_generator import (
    VideoGenerator,
    VideoConfig,
    VideoGenerationResult,
    VideoFormat,
    VideoQuality,
)
from .unreal_bridge import (
    UnrealBridge,
    UnrealConfig,
    UnrealSceneParams,
    UnrealRenderMode,
    CameraMovement,
)
from .jespa_connector import (
    JespaConnector,
    JespaConfig,
    JespaEffect,
    JespaEffectType,
    TransitionType,
)
from .emotion_visual_mapper import (
    EmotionVisualMapper,
    VisualParams,
    VisualStyle,
)
from .scene_composer import (
    SceneComposer,
    SceneDefinition,
    SceneTransition,
    SceneType,
    TransitionStyle,
)
from .embedding_regularization import (
    EmbeddingRegularizer,
    FastEmbeddingPredictor,
    EmbeddingConfig,
    RegularizationConfig,
    RegularizationType,
    create_regularized_mapper,
)

__all__ = [
    # Video Generator
    "VideoGenerator",
    "VideoConfig",
    "VideoGenerationResult",
    "VideoFormat",
    "VideoQuality",
    # Unreal Bridge
    "UnrealBridge",
    "UnrealConfig",
    "UnrealSceneParams",
    "UnrealRenderMode",
    "CameraMovement",
    # Jespa Connector
    "JespaConnector",
    "JespaConfig",
    "JespaEffect",
    "JespaEffectType",
    "TransitionType",
    # Emotion Visual Mapper
    "EmotionVisualMapper",
    "VisualParams",
    "VisualStyle",
    # Scene Composer
    "SceneComposer",
    "SceneDefinition",
    "SceneTransition",
    "SceneType",
    "TransitionStyle",
    # Embedding Regularization
    "EmbeddingRegularizer",
    "FastEmbeddingPredictor",
    "EmbeddingConfig",
    "RegularizationConfig",
    "RegularizationType",
    "create_regularized_mapper",
]
