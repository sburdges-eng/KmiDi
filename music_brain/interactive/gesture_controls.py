"""
Gesture Controls

Handles gesture-based parameter control (drag on emotion wheel, multi-touch, etc.).
Part of Phase 2 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class GestureType(Enum):
    """Types of gestures."""
    DRAG = "drag"
    PINCH = "pinch"
    ROTATE = "rotate"
    SWIPE = "swipe"
    TAP = "tap"
    LONG_PRESS = "long_press"


@dataclass
class Gesture:
    """Represents a gesture."""
    gesture_type: GestureType
    start_position: Tuple[float, float]
    current_position: Tuple[float, float]
    end_position: Optional[Tuple[float, float]] = None
    start_time: float = 0.0
    current_time: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_delta(self) -> Tuple[float, float]:
        """Get position delta from start to current."""
        return (
            self.current_position[0] - self.start_position[0],
            self.current_position[1] - self.start_position[1]
        )

    def get_distance(self) -> float:
        """Get distance from start to current position."""
        dx, dy = self.get_delta()
        return (dx * dx + dy * dy) ** 0.5

    def get_angle(self) -> float:
        """Get angle from start to current position (in radians)."""
        dx, dy = self.get_delta()
        import math
        return math.atan2(dy, dx)


class GestureMapper:
    """
    Maps gestures to parameter changes.

    Usage:
        mapper = GestureMapper()
        mapper.register_drag_handler("emotion_wheel", lambda gesture: ...)
        changes = mapper.process_gesture(gesture)
    """

    def __init__(self):
        """Initialize gesture mapper."""
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)

    def register_handler(
        self,
        target: str,
        gesture_type: GestureType,
        handler: Callable[[Gesture], Dict[str, float]]
    ):
        """
        Register a handler for a gesture on a target.

        Args:
            target: Target UI element (e.g., "emotion_wheel", "parameter_slider")
            gesture_type: Type of gesture
            handler: Function that takes Gesture and returns parameter changes dict
        """
        key = f"{target}:{gesture_type.value}"
        self.handlers[key].append(handler)

    def process_gesture(
        self,
        target: str,
        gesture: Gesture
    ) -> Dict[str, float]:
        """
        Process a gesture and return parameter changes.

        Args:
            target: Target UI element
            gesture: Gesture to process

        Returns:
            Dictionary of parameter changes
        """
        key = f"{target}:{gesture.gesture_type.value}"
        handlers = self.handlers.get(key, [])

        # Combine results from all handlers
        result = {}
        for handler in handlers:
            changes = handler(gesture)
            result.update(changes)

        return result


class EmotionWheelGestureHandler:
    """
    Handles gestures on the emotion wheel to morph between emotions.
    """

    def __init__(self, wheel_radius: float = 1.0):
        """Initialize handler."""
        self.wheel_radius = wheel_radius

    def handle_drag(self, gesture: Gesture) -> Dict[str, float]:
        """
        Handle drag gesture on emotion wheel.
        Maps position to VAD (Valence, Arousal) space.

        Args:
            gesture: Drag gesture

        Returns:
            Dictionary with valence and arousal changes
        """
        # Normalize position to -1 to 1 range
        # Assuming center is at (0, 0) and wheel spans from -radius to +radius
        norm_x = gesture.current_position[0] / self.wheel_radius
        norm_y = gesture.current_position[1] / self.wheel_radius

        # Clamp to unit circle
        import math
        distance = (norm_x * norm_x + norm_y * norm_y) ** 0.5
        if distance > 1.0:
            norm_x /= distance
            norm_y /= distance
            distance = 1.0

        # Map to VAD space:
        # X-axis: Valence (-1 to 1, left to right)
        # Y-axis: Arousal (0 to 1, bottom to top, but inverted: top = high arousal)
        # Distance from center: Intensity (0 to 1)

        valence = norm_x  # -1 (left) to 1 (right)
        arousal = (1.0 - norm_y) / 2.0  # 0 (top) to 1 (bottom), but we want 0 (bottom) to 1 (top)
        intensity = distance

        return {
            "valence": valence,
            "arousal": max(0.0, min(1.0, arousal)),
            "intensity": max(0.0, min(1.0, intensity))
        }


def create_emotion_wheel_drag_handler(radius: float = 1.0) -> Callable:
    """
    Create a drag handler for emotion wheel.

    Args:
        radius: Radius of emotion wheel

    Returns:
        Handler function
    """
    handler = EmotionWheelGestureHandler(radius)
    return lambda gesture: handler.handle_drag(gesture)


def main():
    """Example usage."""
    mapper = GestureMapper()

    # Register emotion wheel drag handler
    wheel_handler = create_emotion_wheel_drag_handler(radius=100.0)
    mapper.register_handler("emotion_wheel", GestureType.DRAG, wheel_handler)

    # Create a drag gesture
    gesture = Gesture(
        gesture_type=GestureType.DRAG,
        start_position=(0.0, 0.0),
        current_position=(50.0, -30.0)  # Right and up
    )

    # Process gesture
    changes = mapper.process_gesture("emotion_wheel", gesture)
    print(f"Parameter changes: {changes}")


if __name__ == "__main__":
    main()
