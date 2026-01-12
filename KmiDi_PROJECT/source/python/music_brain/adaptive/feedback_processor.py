"""
Feedback Processor

Processes user feedback (explicit and implicit) to improve generation.
Part of Phase 4 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from .adaptive_generator import AdaptiveGenerator, GenerationAttempt


class FeedbackProcessor:
    """
    Processes various forms of user feedback to improve generation.
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
        generation_history: Optional[List[GenerationAttempt]] = None
    ) -> Dict[str, Any]:
        """
        Detect patterns in feedback to infer preferences.

        Args:
            generation_history: History of generation attempts (uses generator's if None)

        Returns:
            Dictionary with detected patterns
        """
        if generation_history is None:
            generation_history = self.generator.generation_history

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
            if total > 0 and changes / total > 0.8:
                patterns["always_changed_parameters"].append(param_name)

        # Find parameter combinations that are never accepted
        rejected_combinations = defaultdict(int)
        accepted_combinations = defaultdict(int)

        for attempt in generation_history:
            # Create combination key from parameter values (rounded)
            param_key = tuple(
                (name, round(value, 1))
                for name, value in sorted(attempt.parameters.items())
            )

            if attempt.accepted:
                accepted_combinations[param_key] += 1
            else:
                rejected_combinations[param_key] += 1

        # Combinations that are always rejected
        for combo in rejected_combinations:
            if combo not in accepted_combinations:
                patterns["never_accepted_with"].append(dict(combo))

        return patterns


def main():
    """Example usage."""
    from .adaptive_generator import AdaptiveGenerator

    generator = AdaptiveGenerator()
    processor = FeedbackProcessor(generator)

    # Process some feedback
    processor.process_explicit_feedback(
        parameters={"valence": 0.5, "arousal": 0.6},
        emotion="grief",
        thumbs_up=True
    )

    processor.process_implicit_feedback(
        original_parameters={"valence": 0.5, "arousal": 0.6},
        modified_parameters={"valence": 0.7, "arousal": 0.6},
        emotion="grief"
    )

    # Detect patterns
    patterns = processor.detect_pattern_based_feedback()
    print("Detected Patterns:")
    print(f"  Always Changed Parameters: {patterns['always_changed_parameters']}")


if __name__ == "__main__":
    main()
