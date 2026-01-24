"""
Interactive module for music brain.

Provides real-time parameter adjustment and gesture controls.
"""

from .gesture_controls import (
    EmotionWheelGestureHandler,
    Gesture,
    GestureMapper,
    GestureType,
    create_emotion_wheel_drag_handler,
)
from .realtime_adjustment import (
    InterpolationType,
    MultiParameterMorpher,
    ParameterMorphEngine,
    ParameterSet,
    ParameterState,
)

__all__ = [
    "ParameterMorphEngine",
    "MultiParameterMorpher",
    "ParameterSet",
    "ParameterState",
    "InterpolationType",
    "GestureMapper",
    "Gesture",
    "GestureType",
    "EmotionWheelGestureHandler",
    "create_emotion_wheel_drag_handler",
]
