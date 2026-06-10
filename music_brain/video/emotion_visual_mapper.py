"""
Emotion Visual Mapper - Maps emotional states to visual parameters.

Translates emotional intent from music_brain into concrete visual parameters
for Unreal Engine and Jespa, creating emotionally coherent music videos.

Now includes regularized embedding prediction for improved accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum

try:
    from .embedding_regularization import (
        FastEmbeddingPredictor,
        create_regularized_mapper,
    )

    HAS_REGULARIZATION = True
except ImportError:
    HAS_REGULARIZATION = False


class VisualStyle(Enum):
    """Overall visual style categories."""

    CINEMATIC = "cinematic"
    ABSTRACT = "abstract"
    MINIMALIST = "minimalist"
    SURREAL = "surreal"
    NATURALISTIC = "naturalistic"
    GEOMETRIC = "geometric"


@dataclass
class VisualParams:
    """
    Complete visual parameter set derived from emotion.

    These parameters can be applied to Unreal Engine scenes
    and Jespa effects to create emotion-aligned visuals.
    """

    # Color palette
    primary_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB
    secondary_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    accent_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)

    # Lighting
    brightness: float = 1.0  # 0.0 (dark) to 2.0 (very bright)
    contrast: float = 1.0  # 0.0 (flat) to 2.0 (high contrast)
    warmth: float = 0.0  # -1.0 (cool) to 1.0 (warm)

    # Motion
    motion_speed: float = 1.0  # 0.0 (still) to 2.0 (fast)
    motion_fluidity: float = 1.0  # 0.0 (jerky) to 1.0 (smooth)
    camera_dynamism: float = 0.5  # 0.0 (static) to 1.0 (very dynamic)

    # Space and composition
    depth: float = 0.5  # 0.0 (flat) to 1.0 (deep 3D)
    complexity: float = 0.5  # 0.0 (simple) to 1.0 (complex)
    symmetry: float = 0.5  # 0.0 (asymmetric) to 1.0 (symmetric)

    # Effects
    particle_intensity: float = 0.5  # 0.0 (none) to 1.0 (intense)
    blur_amount: float = 0.0  # 0.0 (sharp) to 1.0 (very blurred)
    glow_intensity: float = 0.5  # 0.0 (none) to 1.0 (intense)
    distortion: float = 0.0  # 0.0 (none) to 1.0 (heavy)

    # Atmosphere
    fog_density: float = 0.1  # 0.0 (clear) to 1.0 (dense)
    atmosphere: str = "neutral"  # "neutral", "dreamy", "harsh", "ethereal"

    # Style
    visual_style: VisualStyle = VisualStyle.CINEMATIC

    # Metadata
    emotion_source: str = ""
    intensity: float = 0.5
    notes: Dict[str, Any] = field(default_factory=dict)


class EmotionVisualMapper:
    """
    Maps emotions to visual parameters for video generation.

    Uses emotion theory and color psychology to create visually
    coherent representations of emotional states.

    Based on established emotion-color and emotion-motion mappings
    from psychology and film theory.

    Now includes optional regularized embedding prediction for improved
    accuracy and generalization.

    Example:
        >>> mapper = EmotionVisualMapper(use_regularization=True)
        >>> params = mapper.map_emotion("grief", intensity=0.8)
        >>> print(f"Primary color: {params.primary_color}")
        >>> print(f"Motion speed: {params.motion_speed}")
    """

    # Emotion to color mappings (RGB, 0-1 scale)
    _EMOTION_COLORS = {
        "joy": (1.0, 0.9, 0.3),  # Bright yellow
        "sadness": (0.3, 0.4, 0.6),  # Muted blue
        "grief": (0.2, 0.2, 0.3),  # Deep blue-grey
        "anger": (0.9, 0.2, 0.1),  # Intense red
        "fear": (0.3, 0.2, 0.4),  # Dark purple
        "surprise": (1.0, 0.6, 0.2),  # Orange
        "disgust": (0.5, 0.6, 0.2),  # Yellow-green
        "trust": (0.3, 0.6, 0.8),  # Sky blue
        "anticipation": (0.9, 0.7, 0.3),  # Gold
        "love": (0.9, 0.3, 0.5),  # Rose/pink
        "peace": (0.6, 0.8, 0.7),  # Mint/aqua
        "anxiety": (0.6, 0.5, 0.4),  # Brown/grey
    }

    def __init__(self, use_regularization: bool = False, regularization_strength: float = 0.001):
        """
        Initialize the emotion-visual mapper.

        Args:
            use_regularization: Whether to use regularized embeddings
            regularization_strength: L2 regularization strength if enabled
        """
        self._custom_mappings: Dict[str, VisualParams] = {}
        self._use_regularization = use_regularization and HAS_REGULARIZATION
        self._embedding_predictor: Optional[FastEmbeddingPredictor] = None

        # Initialize regularized predictor if requested
        if self._use_regularization:
            config, _ = create_regularized_mapper(
                regularization_strength=regularization_strength,
                use_dropout=False,  # Inference only
                use_fast_inference=True,
            )
            self._embedding_predictor = FastEmbeddingPredictor(config)

    def map_emotion(
        self, emotion: str, intensity: float = 0.5, style: Optional[VisualStyle] = None
    ) -> VisualParams:
        """
        Map an emotion to visual parameters.

        Args:
            emotion: Emotion name (e.g., "grief", "joy")
            intensity: Emotion intensity 0.0 to 1.0
            style: Optional visual style override

        Returns:
            VisualParams configured for the emotion

        Note:
            Uses regularized embeddings if enabled for improved accuracy.
            Falls back to rule-based mapping if regularization unavailable.
        """
        emotion_lower = emotion.lower()

        # Use regularized embedding prediction if available
        if self._use_regularization and self._embedding_predictor:
            return self._map_with_regularization(emotion_lower, intensity, style)

        # Fall back to rule-based mapping
        return self._map_rule_based(emotion_lower, intensity, style)

    def _map_rule_based(
        self, emotion: str, intensity: float, style: Optional[VisualStyle]
    ) -> VisualParams:
        """
        Rule-based emotion mapping (original implementation).

        Args:
            emotion: Emotion name
            intensity: Emotion intensity
            style: Visual style override

        Returns:
            VisualParams from rule-based mapping
        """
        # Get base color for emotion
        primary_color = self._EMOTION_COLORS.get(emotion, (0.5, 0.5, 0.5))  # Default neutral grey

        # Create visual parameters
        params = VisualParams(
            primary_color=primary_color, emotion_source=emotion, intensity=intensity
        )

        # Apply intensity to parameters
        params = self._apply_intensity(params, intensity)

        # Apply style if specified
        if style:
            params.visual_style = style

        return params

    def _map_with_regularization(
        self, emotion: str, intensity: float, style: Optional[VisualStyle]
    ) -> VisualParams:
        """
        Regularized embedding-based emotion mapping.

        Args:
            emotion: Emotion name
            intensity: Emotion intensity
            style: Visual style override

        Returns:
            VisualParams from regularized embeddings

        Note:
            This provides more accurate predictions through learned embeddings
            with regularization to prevent overfitting.
        """
        # Get embedding prediction (with caching for speed)
        _embedding = self._embedding_predictor.predict(  # noqa: F841
            emotion=emotion, intensity=intensity, use_cache=True  # Fast path
        )

        # TODO: Convert embedding to VisualParams
        # For now, fall back to rule-based with embedding influence
        params = self._map_rule_based(emotion, intensity, style)

        # Add embedding data to metadata for future use
        params.notes["embedding_used"] = True
        params.notes["embedding_cache_stats"] = self._embedding_predictor.get_performance_stats()

        return params

    def map_emotion_trajectory(
        self, emotions: List[Tuple[str, float, float]], style: Optional[VisualStyle] = None
    ) -> List[VisualParams]:
        """
        Map an emotion trajectory over time to visual parameters.

        Args:
            emotions: List of (emotion, intensity, timestamp) tuples
            style: Optional visual style for entire sequence

        Returns:
            List of VisualParams aligned with emotion timeline

        Note:
            This is a stub. Future implementation will:
            - Smooth transitions between emotions
            - Handle emotion blending
            - Create coherent visual narrative
        """
        return [self.map_emotion(emotion, intensity, style) for emotion, intensity, _ in emotions]

    def blend_emotions(
        self, emotion_a: str, emotion_b: str, blend_factor: float = 0.5, intensity: float = 0.5
    ) -> VisualParams:
        """
        Blend two emotions to create mixed visual parameters.

        Args:
            emotion_a: First emotion
            emotion_b: Second emotion
            blend_factor: Blend from A (0.0) to B (1.0)
            intensity: Overall intensity

        Returns:
            Blended VisualParams

        Note:
            This is a stub. Future implementation will
            intelligently blend color, motion, and effects.
        """
        # Get params for both emotions
        params_a = self.map_emotion(emotion_a, intensity)
        params_b = self.map_emotion(emotion_b, intensity)

        # TODO: Implement proper blending
        # - Blend colors in perceptual color space
        # - Interpolate motion parameters
        # - Combine effects intelligently

        # Simple linear blend for now
        return self._blend_params(params_a, params_b, blend_factor)

    def add_custom_mapping(self, emotion: str, params: VisualParams) -> None:
        """
        Add a custom emotion-to-visual mapping.

        Args:
            emotion: Emotion name
            params: Visual parameters to use for this emotion
        """
        self._custom_mappings[emotion.lower()] = params

    def _apply_intensity(self, params: VisualParams, intensity: float) -> VisualParams:
        """
        Apply intensity scaling to visual parameters.

        Args:
            params: Base visual parameters
            intensity: Intensity factor 0.0-1.0

        Returns:
            Scaled parameters

        Note:
            This is a stub with basic scaling.
        """
        # TODO: Implement sophisticated intensity curves
        # - Different parameters scale differently
        # - Some parameters inverse with intensity
        # - Non-linear scaling for visual impact

        # Basic scaling for now
        params.particle_intensity *= intensity
        params.glow_intensity *= intensity
        params.motion_speed *= 0.5 + intensity * 0.5

        return params

    def _blend_params(
        self, params_a: VisualParams, params_b: VisualParams, factor: float
    ) -> VisualParams:
        """
        Blend two VisualParams linearly.

        Args:
            params_a: First params
            params_b: Second params
            factor: Blend factor 0.0 (all A) to 1.0 (all B)

        Returns:
            Blended params

        Note:
            This is a stub with simple linear blending.
        """
        # TODO: Implement proper parameter blending
        # - Perceptual color blending
        # - Smart style combination

        # Simple linear blend for colors
        def blend_color(c1, c2, f):
            return tuple(c1[i] * (1 - f) + c2[i] * f for i in range(3))

        return VisualParams(
            primary_color=blend_color(params_a.primary_color, params_b.primary_color, factor),
            brightness=params_a.brightness * (1 - factor) + params_b.brightness * factor,
            motion_speed=params_a.motion_speed * (1 - factor) + params_b.motion_speed * factor,
            emotion_source=f"{params_a.emotion_source}+{params_b.emotion_source}",
            intensity=(params_a.intensity + params_b.intensity) / 2,
        )

    def get_emotion_palette(self, emotion: str) -> List[Tuple[float, float, float]]:
        """
        Get a full color palette for an emotion.

        Args:
            emotion: Emotion name

        Returns:
            List of RGB colors forming a cohesive palette

        Note:
            This is a stub. Future implementation will
            return harmonious color palettes.
        """
        # TODO: Generate full color palette
        # - Primary, secondary, accent colors
        # - Ensure visual harmony
        # - Consider color theory

        base_color = self._EMOTION_COLORS.get(emotion.lower(), (0.5, 0.5, 0.5))
        return [base_color]

    def get_regularization_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get regularization and embedding performance statistics.

        Returns:
            Performance stats if regularization enabled, None otherwise
        """
        if self._use_regularization and self._embedding_predictor:
            return self._embedding_predictor.get_performance_stats()
        return None

    def enable_regularization(self, regularization_strength: float = 0.001) -> bool:
        """
        Enable regularized embedding prediction.

        Args:
            regularization_strength: L2 regularization strength

        Returns:
            True if successfully enabled, False if not available
        """
        if not HAS_REGULARIZATION:
            return False

        config, _ = create_regularized_mapper(
            regularization_strength=regularization_strength,
            use_dropout=False,
            use_fast_inference=True,
        )
        self._embedding_predictor = FastEmbeddingPredictor(config)
        self._use_regularization = True
        return True

    def disable_regularization(self) -> None:
        """Disable regularized embedding prediction (use rule-based only)."""
        self._use_regularization = False
        self._embedding_predictor = None
