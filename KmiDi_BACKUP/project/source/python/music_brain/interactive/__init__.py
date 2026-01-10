"""
Interactive module for music brain.

Provides real-time parameter adjustment and gesture controls.
"""

from .realtime_adjustment import (
    ParameterMorphEngine,
    MultiParameterMorpher,
    ParameterSet,
    ParameterState,
    InterpolationType,
)

from .gesture_controls import (
    GestureMapper,
    Gesture,
    GestureType,
    EmotionWheelGestureHandler,
    create_emotion_wheel_drag_handler,
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
