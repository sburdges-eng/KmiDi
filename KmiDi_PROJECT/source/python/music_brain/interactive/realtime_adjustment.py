"""
Real-Time Parameter Adjustment

Provides smooth interpolation and parameter morphing for real-time adjustments.
Part of Phase 2 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math


class InterpolationType(Enum):
    """Types of parameter interpolation."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    SMOOTH = "smooth"  # Cubic smoothstep


@dataclass
class ParameterState:
    """State of a single parameter."""
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 1.0

    def clamp(self) -> float:
        """Clamp value to valid range."""
        return max(self.min_value, min(self.max_value, self.value))


@dataclass
class ParameterSet:
    """Set of parameters representing a state."""
    parameters: Dict[str, float]
    timestamp: float = 0.0  # Time in seconds

    def get(self, name: str, default: float = 0.0) -> float:
        """Get parameter value."""
        return self.parameters.get(name, default)

    def set(self, name: str, value: float):
        """Set parameter value."""
        self.parameters[name] = value


class ParameterMorphEngine:
    """
    Engine for smoothly interpolating between parameter states.

    Usage:
        engine = ParameterMorphEngine()
        start_state = ParameterSet({"valence": 0.5, "arousal": 0.6})
        end_state = ParameterSet({"valence": 0.8, "arousal": 0.4})
        engine.setup_morph(start_state, end_state, duration=1.0)

        # In animation loop:
        current = engine.interpolate(elapsed_time)
    """

    def __init__(self, interpolation_type: InterpolationType = InterpolationType.SMOOTH):
        """Initialize morph engine."""
        self.interpolation_type = interpolation_type
        self.start_state: Optional[ParameterSet] = None
        self.end_state: Optional[ParameterSet] = None
        self.duration: float = 0.0
        self.start_time: float = 0.0

    def setup_morph(
        self,
        start_state: ParameterSet,
        end_state: ParameterSet,
        duration: float,
        start_time: float = 0.0
    ):
        """Setup a morph between two parameter states."""
        self.start_state = start_state
        self.end_state = end_state
        self.duration = duration
        self.start_time = start_time

    def interpolate(self, current_time: float) -> ParameterSet:
        """
        Interpolate between start and end states at given time.

        Args:
            current_time: Current time (should be >= start_time)

        Returns:
            Interpolated parameter set
        """
        if self.start_state is None or self.end_state is None:
            # Return start state if no morph set up
            return self.start_state or ParameterSet({})

        elapsed = current_time - self.start_time

        if elapsed <= 0.0:
            return ParameterSet(self.start_state.parameters.copy())

        if elapsed >= self.duration:
            return ParameterSet(self.end_state.parameters.copy())

        # Calculate interpolation factor (0.0 to 1.0)
        t = elapsed / self.duration
        t = self._apply_interpolation(t)

        # Interpolate all parameters
        result_params = {}
        all_param_names = set(self.start_state.parameters.keys()) | set(self.end_state.parameters.keys())

        for param_name in all_param_names:
            start_val = self.start_state.get(param_name, 0.0)
            end_val = self.end_state.get(param_name, 0.0)
            result_params[param_name] = start_val + (end_val - start_val) * t

        return ParameterSet(result_params, current_time)

    def _apply_interpolation(self, t: float) -> float:
        """Apply interpolation curve to t (0.0 to 1.0)."""
        t = max(0.0, min(1.0, t))  # Clamp

        if self.interpolation_type == InterpolationType.LINEAR:
            return t
        elif self.interpolation_type == InterpolationType.EASE_IN:
            return t * t
        elif self.interpolation_type == InterpolationType.EASE_OUT:
            return 1.0 - (1.0 - t) * (1.0 - t)
        elif self.interpolation_type == InterpolationType.EASE_IN_OUT:
            if t < 0.5:
                return 2.0 * t * t
            else:
                return 1.0 - 2.0 * (1.0 - t) * (1.0 - t)
        elif self.interpolation_type == InterpolationType.SMOOTH:
            # Smoothstep: 3t^2 - 2t^3
            return t * t * (3.0 - 2.0 * t)
        else:
            return t

    def is_complete(self, current_time: float) -> bool:
        """Check if morph is complete."""
        if self.start_state is None or self.end_state is None:
            return True
        elapsed = current_time - self.start_time
        return elapsed >= self.duration


class MultiParameterMorpher:
    """
    Handles concurrent parameter changes with different interpolation speeds.

    Usage:
        morpher = MultiParameterMorpher()
        morpher.set_target("valence", 0.8, duration=1.0)
        morpher.set_target("arousal", 0.4, duration=0.5)  # Faster
        current_params = morpher.update(elapsed_time)
    """

    def __init__(self):
        """Initialize multi-parameter morpher."""
        self.active_morphs: Dict[str, Tuple[ParameterMorphEngine, float, float]] = {}
        self.current_state: ParameterSet = ParameterSet({})

    def set_target(
        self,
        parameter_name: str,
        target_value: float,
        duration: float = 0.5,
        current_time: float = 0.0
    ):
        """Set target value for a parameter with specified duration."""
        current_value = self.current_state.get(parameter_name)

        start_state = ParameterSet({parameter_name: current_value})
        end_state = ParameterSet({parameter_name: target_value})

        engine = ParameterMorphEngine()
        engine.setup_morph(start_state, end_state, duration, current_time)

        self.active_morphs[parameter_name] = (engine, current_time, duration)

    def update(self, current_time: float) -> ParameterSet:
        """
        Update all active morphs and return current state.

        Args:
            current_time: Current time

        Returns:
            Current parameter set with all interpolations applied
        """
        # Remove completed morphs
        completed = [
            name for name, (engine, start_time, duration) in self.active_morphs.items()
            if current_time >= start_time + duration
        ]
        for name in completed:
            del self.active_morphs[name]

        # Update all active morphs
        result_params = self.current_state.parameters.copy()

        for param_name, (engine, start_time, duration) in self.active_morphs.items():
            interpolated = engine.interpolate(current_time)
            result_params[param_name] = interpolated.get(param_name, result_params.get(param_name, 0.0))

        self.current_state = ParameterSet(result_params, current_time)
        return self.current_state

    def set_parameter(self, parameter_name: str, value: float):
        """Set parameter immediately (no morphing)."""
        self.current_state.set(parameter_name, value)
        # Cancel any active morph for this parameter
        if parameter_name in self.active_morphs:
            del self.active_morphs[parameter_name]

    def get_parameter(self, parameter_name: str, default: float = 0.0) -> float:
        """Get current parameter value."""
        return self.current_state.get(parameter_name, default)

    def has_active_morphs(self) -> bool:
        """Check if there are any active morphs."""
        return len(self.active_morphs) > 0


def main():
    """Example usage."""
    morpher = MultiParameterMorpher()

    # Set initial state
    morpher.set_parameter("valence", 0.5)
    morpher.set_parameter("arousal", 0.6)

    # Start morphing to new values with different speeds
    morpher.set_target("valence", 0.8, duration=1.0, current_time=0.0)
    morpher.set_target("arousal", 0.4, duration=0.5, current_time=0.0)

    # Simulate updates
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        state = morpher.update(t)
        print(f"t={t:.2f}: valence={state.get('valence'):.2f}, arousal={state.get('arousal'):.2f}")


if __name__ == "__main__":
    main()
