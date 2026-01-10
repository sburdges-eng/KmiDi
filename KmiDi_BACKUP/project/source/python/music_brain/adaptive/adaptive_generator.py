"""
Adaptive Generation Engine

Wraps IntentPipeline to adapt generation based on user feedback and preferences.
Part of Phase 4 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GenerationAttempt:
    """Record of a generation attempt."""
    parameters: Dict[str, float]
    emotion: Optional[str] = None
    accepted: bool = False
    modifications: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AdaptiveGenerator:
    """
    Adapts music generation based on user feedback and learned preferences.

    Wraps the IntentPipeline to personalize generation parameters.

    Usage:
        generator = AdaptiveGenerator(intent_pipeline, preference_model)
        result = generator.generate_adaptive(wound, learned_params=True)
    """

    def __init__(self, intent_pipeline=None, preference_model=None):
        """
        Initialize adaptive generator.

        Args:
            intent_pipeline: IntentPipeline instance (optional, can be set later)
            preference_model: UserPreferenceModel instance (optional)
        """
        self.intent_pipeline = intent_pipeline
        self.preference_model = preference_model
        self.generation_history: List[GenerationAttempt] = []

        # Learned adjustments
        self.parameter_biases: Dict[str, float] = {}  # Bias to add to parameters
        self.emotion_adjustments: Dict[str, Dict[str, float]] = {}  # Emotion-specific adjustments

    def set_intent_pipeline(self, intent_pipeline):
        """Set the intent pipeline to wrap."""
        self.intent_pipeline = intent_pipeline

    def generate_with_adaptation(
        self,
        wound,
        use_learned_preferences: bool = True
    ):
        """
        Generate music with adaptive parameters based on learned preferences.

        Args:
            wound: Wound object to process
            use_learned_preferences: Whether to apply learned preferences

        Returns:
            IntentResult with adapted parameters
        """
        if not self.intent_pipeline:
            raise ValueError("IntentPipeline not set. Call set_intent_pipeline() first.")

        # Process wound normally first
        result = self.intent_pipeline.process(wound)

        if use_learned_preferences and self.preference_model:
            # Adapt parameters based on learned preferences
            result = self._apply_learned_adjustments(result, wound)

        return result

    def _apply_learned_adjustments(self, result, wound) -> Any:
        """
        Apply learned parameter adjustments to result.

        Args:
            result: IntentResult from pipeline
            wound: Original wound

        Returns:
            Modified result with learned adjustments applied
        """
        # Get user's preferred parameter values
        if not self.preference_model:
            return result

        preferences = self.preference_model.get_parameter_preferences()

        # Apply parameter biases
        for param_name, pref in preferences.items():
            if hasattr(result, param_name):
                current_value = getattr(result, param_name, None)
                if current_value is not None and isinstance(current_value, (int, float)):
                    # Blend current value with preferred value (70% preferred, 30% current)
                    preferred_mean = pref.get("mean", current_value)
                    adjusted_value = current_value * 0.3 + preferred_mean * 0.7
                    setattr(result, param_name, adjusted_value)

        # Apply emotion-specific adjustments
        emotion = self._extract_emotion_from_wound(wound)
        if emotion and emotion in self.emotion_adjustments:
            adjustments = self.emotion_adjustments[emotion]
            for param_name, adjustment in adjustments.items():
                if hasattr(result, param_name):
                    current_value = getattr(result, param_name, None)
                    if current_value is not None and isinstance(current_value, (int, float)):
                        # Add adjustment (can be positive or negative)
                        new_value = current_value + adjustment
                        setattr(result, param_name, new_value)

        return result

    def _extract_emotion_from_wound(self, wound) -> Optional[str]:
        """Extract emotion name from wound (helper method)."""
        # Try to extract emotion from wound description or attributes
        if hasattr(wound, 'description'):
            description = str(wound.description).lower()
            # Simple keyword matching (in production, use emotion thesaurus)
            emotion_keywords = {
                "grief": ["grief", "sad", "sorrow", "mourning"],
                "anger": ["anger", "angry", "rage", "furious"],
                "hope": ["hope", "hopeful", "optimistic"],
                "joy": ["joy", "happy", "joyful", "elated"],
                "fear": ["fear", "afraid", "scared", "anxious"],
            }
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in description for keyword in keywords):
                    return emotion
        return None

    def record_generation_feedback(
        self,
        parameters: Dict[str, float],
        emotion: Optional[str] = None,
        accepted: bool = True,
        modifications: Optional[Dict[str, Any]] = None
    ):
        """Record feedback on a generation attempt."""
        attempt = GenerationAttempt(
            parameters=parameters.copy(),
            emotion=emotion,
            accepted=accepted,
            modifications=modifications or {}
        )
        self.generation_history.append(attempt)

        if self.preference_model:
            self.preference_model.record_generation(
                parameters=parameters,
                emotion=emotion,
                accepted=accepted,
                modifications_made=modifications
            )

        # Learn from feedback
        if accepted and modifications:
            self._learn_from_modifications(parameters, modifications)

    def _learn_from_modifications(
        self,
        original_parameters: Dict[str, float],
        modifications: Dict[str, Any]
    ):
        """
        Learn parameter adjustments from user modifications.

        Args:
            original_parameters: Original generation parameters
            modifications: User modifications
        """
        # Track parameter changes
        for param_name, new_value in modifications.get("parameters", {}).items():
            old_value = original_parameters.get(param_name)
            if old_value is not None and isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                # Calculate adjustment needed
                adjustment = new_value - old_value

                # Update bias (exponential moving average)
                if param_name not in self.parameter_biases:
                    self.parameter_biases[param_name] = 0.0

                # Learning rate: 0.1 (slow adaptation)
                self.parameter_biases[param_name] = (
                    self.parameter_biases[param_name] * 0.9 + adjustment * 0.1
                )

    def learn_from_parameter_changes(
        self,
        parameter_changes: Dict[str, Tuple[float, float]]  # param_name -> (old, new)
    ):
        """
        Learn from parameter adjustments user made after generation.

        Args:
            parameter_changes: Dictionary of parameter changes
        """
        for param_name, (old_value, new_value) in parameter_changes.items():
            adjustment = new_value - old_value

            # Update bias
            if param_name not in self.parameter_biases:
                self.parameter_biases[param_name] = 0.0

            # Learning rate: 0.1
            self.parameter_biases[param_name] = (
                self.parameter_biases[param_name] * 0.9 + adjustment * 0.1
            )

    def personalize_emotion_mapping(
        self,
        emotion: str,
        adjustments: Dict[str, float]
    ):
        """
        Store personalized adjustments for a specific emotion.

        Args:
            emotion: Emotion name
            adjustments: Parameter adjustments for this emotion
        """
        if emotion not in self.emotion_adjustments:
            self.emotion_adjustments[emotion] = {}

        # Merge adjustments (average if already exists)
        for param_name, adjustment in adjustments.items():
            if param_name in self.emotion_adjustments[emotion]:
                # Average with existing
                self.emotion_adjustments[emotion][param_name] = (
                    self.emotion_adjustments[emotion][param_name] * 0.5 + adjustment * 0.5
                )
            else:
                self.emotion_adjustments[emotion][param_name] = adjustment

    def get_acceptance_rate(self) -> float:
        """Get overall acceptance rate of generations."""
        if not self.generation_history:
            return 0.0

        accepted = sum(1 for attempt in self.generation_history if attempt.accepted)
        return accepted / len(self.generation_history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generation and learning."""
        return {
            "total_generations": len(self.generation_history),
            "acceptance_rate": self.get_acceptance_rate(),
            "parameter_biases": self.parameter_biases.copy(),
            "emotion_adjustments": {k: v.copy() for k, v in self.emotion_adjustments.items()},
        }


class FeedbackProcessor:
    """
    Processes user feedback to improve generation.
    """

    def __init__(self, adaptive_generator: AdaptiveGenerator):
        """Initialize feedback processor."""
        self.generator = adaptive_generator

    def process_explicit_feedback(
        self,
        parameters: Dict[str, float],
        emotion: Optional[str],
        thumbs_up: bool
    ):
        """Process explicit thumbs up/down feedback."""
        self.generator.record_generation_feedback(
            parameters=parameters,
            emotion=emotion,
            accepted=thumbs_up
        )

    def process_implicit_feedback(
        self,
        original_parameters: Dict[str, float],
        modified_parameters: Dict[str, float],
        emotion: Optional[str] = None
    ):
        """Process implicit feedback (parameter adjustments after generation)."""
        # Calculate changes
        changes = {
            param_name: (original_parameters.get(param_name, 0.0), new_value)
            for param_name, new_value in modified_parameters.items()
            if param_name in original_parameters and original_parameters[param_name] != new_value
        }

        if changes:
            self.generator.learn_from_parameter_changes(changes)

            # Record as accepted (user kept it, just modified)
            self.generator.record_generation_feedback(
                parameters=modified_parameters,
                emotion=emotion,
                accepted=True,
                modifications=changes
            )

    def detect_pattern_based_feedback(
        self,
        generation_history: List[GenerationAttempt]
    ) -> Dict[str, Any]:
        """
        Detect patterns in feedback to infer preferences.

        Args:
            generation_history: History of generation attempts

        Returns:
            Dictionary with detected patterns
        """
        patterns = {
            "always_changed_parameters": [],
            "never_accepted_with": [],
            "frequently_accepted_combinations": [],
        }

        # Find parameters that are always changed
        param_change_frequency = defaultdict(int)
        total_with_param = defaultdict(int)

        for attempt in generation_history:
            if attempt.modifications and "parameters" in attempt.modifications:
                for param_name in attempt.modifications["parameters"]:
                    param_change_frequency[param_name] += 1
                for param_name in attempt.parameters:
                    total_with_param[param_name] += 1

        # Parameters changed >80% of the time
        for param_name, changes in param_change_frequency.items():
            total = total_with_param.get(param_name, 1)
            if changes / total > 0.8:
                patterns["always_changed_parameters"].append(param_name)

        return patterns


def main():
    """Example usage."""
    generator = AdaptiveGenerator()

    # Simulate some feedback
    generator.record_generation_feedback(
        parameters={"valence": 0.5, "arousal": 0.6},
        emotion="grief",
        accepted=True,
        modifications={"parameters": {"valence": 0.7}}  # User increased valence
    )

    generator.learn_from_parameter_changes({"valence": (0.5, 0.7)})

    stats = generator.get_statistics()
    print("Adaptive Generator Statistics:")
    print(f"  Acceptance Rate: {stats['acceptance_rate']:.1%}")
    print(f"  Parameter Biases: {stats['parameter_biases']}")


if __name__ == "__main__":
    main()
