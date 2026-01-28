"""
Dynamics Integration - Cross-Module Integration for Dynamics System

Bridges the gap between:
- C++ Penta-Core engines (real-time processing)
- Python ML training infrastructure
- Music Brain orchestration and user preferences

This module provides a unified interface for:
1. Section-aware dynamics processing
2. Emotion-to-dynamics mapping
3. User preference integration
4. ML model inference coordination

Usage:
    from music_brain.integrations.dynamics_integration import DynamicsIntegration

    dynamics = DynamicsIntegration()

    # Set emotion trajectory
    dynamics.set_emotion("melancholy", valence=-0.6, arousal=0.3)

    # Get dynamics parameters for current section
    params = dynamics.get_dynamics_for_section("verse", bar=16)

    # Apply user preferences
    params = dynamics.apply_user_preferences(params, user_id="default")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED TYPES (Mirror C++ types for Python use)
# =============================================================================

class SectionType(Enum):
    """Song section types - mirrors penta::dynamics::SectionType."""
    UNKNOWN = "unknown"
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    BREAKDOWN = "breakdown"
    DROP = "drop"
    OUTRO = "outro"
    SOLO = "solo"
    INSTRUMENTAL = "instrumental"
    BUILD_UP = "build_up"
    RELEASE = "release"
    TRANSITION = "transition"


@dataclass
class EmotionState:
    """Emotion state using PAD model - mirrors penta::dynamics::EmotionState."""
    valence: float = 0.0       # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.5       # 0.0 (calm) to 1.0 (excited)
    dominance: float = 0.5     # 0.0 (submissive) to 1.0 (dominant)
    intensity: float = 0.5     # 0.0 (subtle) to 1.0 (intense)

    def lerp(self, other: "EmotionState", t: float) -> "EmotionState":
        """Interpolate between two emotion states."""
        return EmotionState(
            valence=self.valence + (other.valence - self.valence) * t,
            arousal=self.arousal + (other.arousal - self.arousal) * t,
            dominance=self.dominance + (other.dominance - self.dominance) * t,
            intensity=self.intensity + (other.intensity - self.intensity) * t,
        )


@dataclass
class DynamicsParameters:
    """Dynamics parameters for a section or moment."""
    # Loudness targets
    target_lufs: float = -14.0
    target_crest_factor: float = 12.0
    target_dynamic_range: float = 8.0

    # Velocity targets
    velocity_mean: int = 80
    velocity_min: int = 40
    velocity_max: int = 120
    velocity_variance: float = 0.2

    # Density
    note_density: float = 2.0
    voice_count: int = 4

    # Emotion source
    emotion: EmotionState = field(default_factory=EmotionState)
    section_type: SectionType = SectionType.UNKNOWN


@dataclass
class SectionContext:
    """Context for a song section - bridges arrangement to dynamics."""
    section_id: str = ""
    section_type: SectionType = SectionType.UNKNOWN
    start_bar: int = 0
    end_bar: int = 0
    emotion: EmotionState = field(default_factory=EmotionState)
    dynamics: DynamicsParameters = field(default_factory=DynamicsParameters)


# =============================================================================
# EMOTION TO DYNAMICS MAPPER
# =============================================================================

class EmotionToDynamicsMapper:
    """
    Maps emotion states to dynamics parameters.

    Based on music psychology research linking emotional expression
    to musical parameters.
    """

    @staticmethod
    def map_emotion_to_dynamics(emotion: EmotionState) -> DynamicsParameters:
        """
        Convert emotion state to dynamics parameters.

        Args:
            emotion: PAD emotion state

        Returns:
            Corresponding dynamics parameters
        """
        params = DynamicsParameters(emotion=emotion)

        # Energy factor combines arousal and intensity
        energy_factor = emotion.arousal * 0.6 + emotion.intensity * 0.4

        # Velocity mapping
        # High arousal + high intensity = louder
        params.velocity_mean = int(50 + energy_factor * 60)
        params.velocity_mean = max(30, min(120, params.velocity_mean))

        # Velocity range based on dominance
        range_width = 20 + emotion.dominance * 40
        params.velocity_min = int(max(1, params.velocity_mean - range_width))
        params.velocity_max = int(min(127, params.velocity_mean + range_width))

        # LUFS: soft to loud based on energy
        # -20 (soft) to -8 (loud)
        params.target_lufs = -20.0 + energy_factor * 12.0

        # Crest factor: positive valence = more compressed (pop sound)
        # Negative valence = more dynamic range (classical/acoustic)
        compression_factor = 0.5 - emotion.valence * 0.3
        params.target_crest_factor = 6.0 + compression_factor * 12.0

        # Note density based on arousal
        params.note_density = 1.0 + emotion.arousal * 6.0

        # Dynamic range based on valence and intensity
        # Sad music often has subtle dynamics, angry has extremes
        if emotion.valence < 0:
            params.target_dynamic_range = 6.0 + emotion.intensity * 8.0
        else:
            params.target_dynamic_range = 4.0 + emotion.intensity * 6.0

        return params

    @staticmethod
    def map_section_to_emotion_bias(section_type: SectionType) -> EmotionState:
        """
        Get default emotion bias for a section type.

        Args:
            section_type: Type of section

        Returns:
            Default emotion state for the section
        """
        biases = {
            SectionType.INTRO: EmotionState(0.0, 0.3, 0.4, 0.4),
            SectionType.VERSE: EmotionState(0.0, 0.4, 0.5, 0.5),
            SectionType.PRE_CHORUS: EmotionState(0.2, 0.6, 0.5, 0.6),
            SectionType.CHORUS: EmotionState(0.4, 0.7, 0.7, 0.8),
            SectionType.BRIDGE: EmotionState(-0.1, 0.5, 0.4, 0.6),
            SectionType.BREAKDOWN: EmotionState(-0.2, 0.3, 0.3, 0.4),
            SectionType.DROP: EmotionState(0.3, 0.9, 0.8, 0.9),
            SectionType.BUILD_UP: EmotionState(0.1, 0.7, 0.6, 0.7),
            SectionType.OUTRO: EmotionState(0.0, 0.2, 0.4, 0.3),
            SectionType.SOLO: EmotionState(0.2, 0.6, 0.6, 0.7),
        }
        return biases.get(section_type, EmotionState())


# =============================================================================
# DYNAMICS INTEGRATION
# =============================================================================

class DynamicsIntegration:
    """
    Main integration class bridging all dynamics-related systems.

    Coordinates between:
    - Emotion trajectory (from user or AI)
    - Section context (from arrangement)
    - User preferences (from learning system)
    - ML models (for prediction)
    """

    def __init__(self):
        self.sections: List[SectionContext] = []
        self.current_bar: int = 0
        self.current_beat: float = 0.0
        self.base_tempo: float = 120.0

        self._user_preferences = None
        self._ml_interface = None

        # Callbacks for C++ integration
        self._on_dynamics_change: Optional[Callable[[DynamicsParameters], None]] = None

    # =========================================================================
    # Section Management
    # =========================================================================

    def add_section(
        self,
        section_type: SectionType,
        start_bar: int,
        end_bar: int,
        emotion: Optional[EmotionState] = None,
    ) -> SectionContext:
        """
        Add a section to the arrangement.

        Args:
            section_type: Type of section
            start_bar: Starting bar
            end_bar: Ending bar
            emotion: Optional emotion state (defaults to section bias)

        Returns:
            Created section context
        """
        if emotion is None:
            emotion = EmotionToDynamicsMapper.map_section_to_emotion_bias(section_type)

        dynamics = EmotionToDynamicsMapper.map_emotion_to_dynamics(emotion)
        dynamics.section_type = section_type

        section = SectionContext(
            section_id=f"{section_type.value}_{len(self.sections)}",
            section_type=section_type,
            start_bar=start_bar,
            end_bar=end_bar,
            emotion=emotion,
            dynamics=dynamics,
        )

        self.sections.append(section)
        self.sections.sort(key=lambda s: s.start_bar)

        return section

    def get_current_section(self) -> Optional[SectionContext]:
        """Get the section at the current position."""
        for section in self.sections:
            if section.start_bar <= self.current_bar < section.end_bar:
                return section
        return self.sections[-1] if self.sections else None

    def set_position(self, bar: int, beat: float = 0.0):
        """Set current playback position."""
        self.current_bar = bar
        self.current_beat = beat

    # =========================================================================
    # Dynamics Calculation
    # =========================================================================

    def get_dynamics_for_section(
        self,
        section_type: str,
        bar: Optional[int] = None,
    ) -> DynamicsParameters:
        """
        Get dynamics parameters for a section.

        Args:
            section_type: Section type name
            bar: Optional bar number for position-based lookup

        Returns:
            Dynamics parameters
        """
        try:
            stype = SectionType(section_type.lower())
        except ValueError:
            stype = SectionType.UNKNOWN

        if bar is not None:
            self.current_bar = bar

        # Find matching section
        for section in self.sections:
            if section.section_type == stype:
                return section.dynamics

        # Generate default dynamics for section type
        emotion = EmotionToDynamicsMapper.map_section_to_emotion_bias(stype)
        return EmotionToDynamicsMapper.map_emotion_to_dynamics(emotion)

    def get_dynamics_at_position(
        self,
        bar: int,
        beat: float = 0.0,
    ) -> DynamicsParameters:
        """
        Get interpolated dynamics at a specific position.

        Handles transitions between sections smoothly.

        Args:
            bar: Bar number
            beat: Beat within bar

        Returns:
            Dynamics parameters (potentially interpolated)
        """
        self.set_position(bar, beat)
        current = self.get_current_section()

        if not current:
            return DynamicsParameters()

        # Check for section transition
        if self._is_near_section_end(current, bar, beat):
            next_section = self._get_next_section(current)
            if next_section:
                t = self._calculate_transition_factor(current, bar, beat)
                return self._interpolate_dynamics(
                    current.dynamics,
                    next_section.dynamics,
                    t
                )

        return current.dynamics

    def _is_near_section_end(
        self,
        section: SectionContext,
        bar: int,
        beat: float
    ) -> bool:
        """Check if position is near section end (within 2 bars)."""
        return bar >= section.end_bar - 2

    def _get_next_section(self, current: SectionContext) -> Optional[SectionContext]:
        """Get the section following the current one."""
        idx = self.sections.index(current)
        if idx + 1 < len(self.sections):
            return self.sections[idx + 1]
        return None

    def _calculate_transition_factor(
        self,
        section: SectionContext,
        bar: int,
        beat: float
    ) -> float:
        """Calculate transition factor (0-1) for section boundary."""
        bars_remaining = section.end_bar - bar
        # 2-bar transition window
        return 1.0 - (bars_remaining / 2.0)

    def _interpolate_dynamics(
        self,
        a: DynamicsParameters,
        b: DynamicsParameters,
        t: float
    ) -> DynamicsParameters:
        """Interpolate between two dynamics states."""
        t = max(0.0, min(1.0, t))

        return DynamicsParameters(
            target_lufs=a.target_lufs + (b.target_lufs - a.target_lufs) * t,
            target_crest_factor=a.target_crest_factor + (b.target_crest_factor - a.target_crest_factor) * t,
            target_dynamic_range=a.target_dynamic_range + (b.target_dynamic_range - a.target_dynamic_range) * t,
            velocity_mean=int(a.velocity_mean + (b.velocity_mean - a.velocity_mean) * t),
            velocity_min=int(a.velocity_min + (b.velocity_min - a.velocity_min) * t),
            velocity_max=int(a.velocity_max + (b.velocity_max - a.velocity_max) * t),
            note_density=a.note_density + (b.note_density - a.note_density) * t,
            voice_count=int(a.voice_count + (b.voice_count - a.voice_count) * t),
            emotion=a.emotion.lerp(b.emotion, t),
        )

    # =========================================================================
    # User Preference Integration
    # =========================================================================

    def set_user_preferences(self, preferences):
        """Set user preference model for personalization."""
        self._user_preferences = preferences

    def apply_user_preferences(
        self,
        params: DynamicsParameters,
        user_id: str = "default",
    ) -> DynamicsParameters:
        """
        Adjust dynamics parameters based on user preferences.

        Args:
            params: Base dynamics parameters
            user_id: User ID for preference lookup

        Returns:
            Adjusted parameters
        """
        if self._user_preferences is None:
            return params

        try:
            # Get user's preferred parameter ranges
            preferred_ranges = self._user_preferences.get_preferred_parameter_ranges()

            # Adjust velocity based on user preference
            if "velocity" in preferred_ranges:
                vel_min, vel_max = preferred_ranges["velocity"]
                user_mean = (vel_min + vel_max) / 2
                # Blend 70/30 with user preference
                params.velocity_mean = int(params.velocity_mean * 0.7 + user_mean * 0.3)

            # Get prediction based on current emotion
            if params.emotion:
                emotion_name = self._emotion_to_name(params.emotion)
                predicted = self._user_preferences.predict_parameters(
                    emotion=emotion_name
                )

                # Apply predicted intensity
                if "intensity" in predicted:
                    intensity_factor = predicted["intensity"]
                    params.velocity_mean = int(
                        params.velocity_mean * (0.8 + intensity_factor * 0.4)
                    )

        except Exception as e:
            logger.warning(f"Error applying user preferences: {e}")

        return params

    def _emotion_to_name(self, emotion: EmotionState) -> str:
        """Convert emotion state to emotion name."""
        # Simple quadrant mapping
        if emotion.valence > 0:
            if emotion.arousal > 0.5:
                return "excited" if emotion.intensity > 0.6 else "happy"
            else:
                return "content" if emotion.intensity > 0.6 else "relaxed"
        else:
            if emotion.arousal > 0.5:
                return "angry" if emotion.intensity > 0.6 else "anxious"
            else:
                return "sad" if emotion.intensity > 0.6 else "melancholy"

    # =========================================================================
    # ML Integration
    # =========================================================================

    def set_ml_interface(self, ml_interface):
        """Set ML interface for model predictions."""
        self._ml_interface = ml_interface

    def predict_dynamics_ml(
        self,
        emotion: EmotionState,
        section_type: SectionType,
    ) -> Optional[DynamicsParameters]:
        """
        Use ML model to predict dynamics.

        Args:
            emotion: Current emotion state
            section_type: Section type

        Returns:
            Predicted dynamics, or None if ML unavailable
        """
        if self._ml_interface is None:
            return None

        try:
            # Create inference request
            request = self._create_dynamics_request(emotion, section_type)

            # Submit and wait for result
            if self._ml_interface.submitRequest(request):
                result = self._ml_interface.pollResult()
                if result and result.success:
                    return self._parse_dynamics_result(result)

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")

        return None

    def _create_dynamics_request(
        self,
        emotion: EmotionState,
        section_type: SectionType
    ) -> Dict[str, Any]:
        """Create ML inference request."""
        return {
            "model_type": "dynamics_engine",
            "input": {
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "dominance": emotion.dominance,
                "intensity": emotion.intensity,
                "section_type": section_type.value,
            }
        }

    def _parse_dynamics_result(self, result) -> DynamicsParameters:
        """Parse ML result into dynamics parameters."""
        output = result.output_data

        return DynamicsParameters(
            target_lufs=output.get("target_lufs", -14.0),
            velocity_mean=int(output.get("velocity_mean", 80)),
            velocity_min=int(output.get("velocity_min", 40)),
            velocity_max=int(output.get("velocity_max", 120)),
            note_density=output.get("note_density", 2.0),
        )

    # =========================================================================
    # C++ Integration Callbacks
    # =========================================================================

    def set_on_dynamics_change(self, callback: Callable[[DynamicsParameters], None]):
        """Set callback for dynamics changes (for C++ bridge)."""
        self._on_dynamics_change = callback

    def notify_dynamics_change(self, params: DynamicsParameters):
        """Notify listeners of dynamics change."""
        if self._on_dynamics_change:
            self._on_dynamics_change(params)

    # =========================================================================
    # Export for Training
    # =========================================================================

    def export_training_data(self) -> List[Dict[str, Any]]:
        """
        Export section/dynamics data for ML training.

        Returns:
            List of training samples
        """
        samples = []

        for section in self.sections:
            sample = {
                "section_type": section.section_type.value,
                "bar_start": section.start_bar,
                "bar_end": section.end_bar,
                "emotion": {
                    "valence": section.emotion.valence,
                    "arousal": section.emotion.arousal,
                    "dominance": section.emotion.dominance,
                    "intensity": section.emotion.intensity,
                },
                "dynamics": {
                    "target_lufs": section.dynamics.target_lufs,
                    "velocity_mean": section.dynamics.velocity_mean,
                    "velocity_range": (
                        section.dynamics.velocity_min,
                        section.dynamics.velocity_max
                    ),
                    "note_density": section.dynamics.note_density,
                    "crest_factor": section.dynamics.target_crest_factor,
                },
            }
            samples.append(sample)

        return samples


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_section_dynamics(
    sections: List[Tuple[str, int, int]],
    base_emotion: Optional[EmotionState] = None,
) -> DynamicsIntegration:
    """
    Create dynamics integration with section definitions.

    Args:
        sections: List of (section_type, start_bar, end_bar) tuples
        base_emotion: Optional base emotion to use

    Returns:
        Configured DynamicsIntegration instance

    Example:
        dynamics = create_section_dynamics([
            ("intro", 0, 8),
            ("verse", 8, 24),
            ("chorus", 24, 40),
            ("outro", 40, 48),
        ])
    """
    integration = DynamicsIntegration()

    for section_type, start_bar, end_bar in sections:
        try:
            stype = SectionType(section_type.lower())
        except ValueError:
            stype = SectionType.UNKNOWN

        integration.add_section(stype, start_bar, end_bar, base_emotion)

    return integration


def get_dynamics_for_emotion(
    emotion_name: str,
    section_type: str = "verse",
) -> DynamicsParameters:
    """
    Get dynamics parameters for an emotion.

    Args:
        emotion_name: Emotion name (e.g., "melancholy", "joyful")
        section_type: Section type for context

    Returns:
        Dynamics parameters
    """
    # Map emotion names to PAD values
    emotion_map = {
        "melancholy": EmotionState(-0.6, 0.3, 0.3, 0.6),
        "sad": EmotionState(-0.7, 0.2, 0.2, 0.5),
        "grief": EmotionState(-0.8, 0.4, 0.2, 0.8),
        "angry": EmotionState(-0.5, 0.8, 0.8, 0.9),
        "anxious": EmotionState(-0.4, 0.7, 0.3, 0.7),
        "happy": EmotionState(0.7, 0.6, 0.6, 0.6),
        "joyful": EmotionState(0.8, 0.7, 0.7, 0.8),
        "excited": EmotionState(0.6, 0.9, 0.7, 0.9),
        "peaceful": EmotionState(0.5, 0.2, 0.5, 0.4),
        "content": EmotionState(0.4, 0.3, 0.5, 0.5),
        "nostalgic": EmotionState(0.1, 0.3, 0.4, 0.6),
        "hopeful": EmotionState(0.5, 0.5, 0.5, 0.6),
    }

    emotion = emotion_map.get(
        emotion_name.lower(),
        EmotionState()
    )

    params = EmotionToDynamicsMapper.map_emotion_to_dynamics(emotion)

    try:
        params.section_type = SectionType(section_type.lower())
    except ValueError:
        pass

    return params
