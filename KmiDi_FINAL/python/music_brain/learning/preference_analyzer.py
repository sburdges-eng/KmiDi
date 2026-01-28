"""
Preference Analyzer - Statistical analysis of user preferences.

Provides detailed analysis of user behavior patterns, preference evolution,
and insights for the suggestion engine.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

from music_brain.learning.user_preferences import (
    UserPreferenceModel,
    UserPreferenceProfile,
    ParameterAdjustment,
    EmotionSelection,
    MidiGenerationEvent,
)


class PreferenceAnalyzer:
    """
    Analyzes user preferences to extract patterns and insights.

    Provides:
    - Most-used emotion ranges
    - Preferred tempo/valence/arousal zones
    - Common parameter combinations
    - Evolution of preferences over time
    - Parameter correlation analysis
    """

    def __init__(self, preference_model: UserPreferenceModel):
        """Initialize analyzer with a preference model."""
        self.model = preference_model
        self.profile = preference_model.get_profile()

    def get_most_used_emotion_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get most-used ranges for valence and arousal.

        Returns:
            Dictionary with "valence" and "arousal" keys, each mapping to (min, max) range
        """
        if not self.profile.emotion_selections:
            return {}

        valence_values = [sel.valence for sel in self.profile.emotion_selections]
        arousal_values = [sel.arousal for sel in self.profile.emotion_selections]

        return {
            "valence": (min(valence_values), max(valence_values)),
            "arousal": (min(arousal_values), max(arousal_values)),
        }

    def get_preferred_tempo_range(self) -> Optional[Tuple[int, int]]:
        """
        Get preferred tempo range in BPM.

        Looks for tempo in parameter adjustments and generation events.
        """
        tempo_values = []

        # Check parameter adjustments for tempo
        for adj in self.profile.parameter_adjustments:
            if "tempo" in adj.parameter_name.lower():
                tempo_values.append(adj.new_value)

        # Check generation events for tempo parameter
        for event in self.profile.midi_generations:
            if "tempo" in event.parameters:
                tempo_values.append(event.parameters["tempo"])

        if not tempo_values:
            return None

        return (int(min(tempo_values)), int(max(tempo_values)))

    def get_common_parameter_combinations(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find most common parameter combinations.

        Returns:
            List of dictionaries with parameter combinations and their frequency
        """
        # Group generations by parameter "fingerprint" (rounded values)
        combinations = defaultdict(int)
        combination_params = {}

        for event in self.profile.midi_generations:
            if not event.parameters:
                continue

            # Create a fingerprint by rounding parameters to 0.1
            fingerprint_parts = []
            for key in sorted(event.parameters.keys()):
                val = round(event.parameters[key] * 10) / 10
                fingerprint_parts.append(f"{key}:{val}")

            fingerprint = "|".join(fingerprint_parts)
            combinations[fingerprint] += 1
            combination_params[fingerprint] = event.parameters.copy()

        # Sort by frequency and return top N
        sorted_combos = sorted(combinations.items(), key=lambda x: x[1], reverse=True)

        result = []
        for fingerprint, count in sorted_combos[:top_n]:
            result.append({
                "parameters": combination_params[fingerprint],
                "frequency": count,
                "percentage": count / len(self.profile.midi_generations) * 100 if self.profile.midi_generations else 0
            })

        return result

    def get_preference_evolution(self, days: int = 30) -> Dict[str, List[Tuple[str, float]]]:
        """
        Track how preferences have evolved over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary mapping parameter names to list of (date, average_value) tuples
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Group adjustments by date and parameter
        by_date_param = defaultdict(lambda: defaultdict(list))

        for adj in self.profile.parameter_adjustments:
            try:
                adj_date = datetime.fromisoformat(adj.timestamp)
                if adj_date < cutoff_date:
                    continue

                date_key = adj_date.date().isoformat()
                by_date_param[adj.parameter_name][date_key].append(adj.new_value)
            except (ValueError, TypeError):
                continue

        evolution = {}
        for param_name, date_values in by_date_param.items():
            daily_averages = []
            for date_key in sorted(date_values.keys()):
                values = date_values[date_key]
                avg = statistics.mean(values)
                daily_averages.append((date_key, avg))
            evolution[param_name] = daily_averages

        return evolution

    def get_parameter_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Find correlations between parameter adjustments.

        Returns:
            Dictionary mapping parameter name to dictionary of correlated parameters
            with correlation scores (-1.0 to 1.0)
        """
        # Collect all parameter values from generations
        param_matrix = defaultdict(list)

        for event in self.profile.midi_generations:
            for param_name, value in event.parameters.items():
                param_matrix[param_name].append(value)

        # Calculate correlations
        correlations = {}
        param_names = list(param_matrix.keys())

        for i, param1 in enumerate(param_names):
            if len(param_matrix[param1]) < 2:
                continue

            correlations[param1] = {}
            for param2 in param_names[i+1:]:
                if len(param_matrix[param2]) < 2:
                    continue

                # Simple correlation: check if they move together
                # (This is a simplified correlation - could use Pearson's r for more accuracy)
                values1 = param_matrix[param1]
                values2 = param_matrix[param2]

                if len(values1) != len(values2):
                    # Pad shorter list
                    min_len = min(len(values1), len(values2))
                    values1 = values1[:min_len]
                    values2 = values2[:min_len]

                if len(values1) < 2:
                    continue

                # Calculate Pearson correlation coefficient
                try:
                    corr = self._pearson_correlation(values1, values2)
                    if abs(corr) > 0.3:  # Only report meaningful correlations
                        correlations[param1][param2] = corr
                except (ValueError, ZeroDivisionError):
                    pass

        return correlations

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        return numerator / (denominator_x * denominator_y) ** 0.5

    def get_rule_break_preferences(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze which rule-breaks user prefers.

        Returns:
            Dictionary with "kept" and "removed" keys, each mapping to
            rule-break name -> count
        """
        kept = defaultdict(int)
        removed = defaultdict(int)

        for mod in self.profile.rule_break_modifications:
            if mod.action == "added":
                kept[mod.rule_break] += 1
            elif mod.action == "removed":
                removed[mod.rule_break] += 1

        return {
            "kept": dict(kept),
            "removed": dict(removed),
        }

    def get_style_preferences(self) -> Dict[str, float]:
        """
        Infer style preferences from parameter patterns.

        Returns:
            Dictionary mapping style/genre names to preference scores (0.0 to 1.0)
        """
        # This is a simplified version - could be enhanced with ML
        # For now, infer from parameter combinations

        style_scores = defaultdict(float)

        # Analyze accepted generations for style patterns
        for event in self.profile.midi_generations:
            if event.accepted is not True:
                continue

            params = event.parameters

            # Simple heuristics for style detection
            if params.get("valence", 0) > 0.5 and params.get("arousal", 0) > 0.5:
                style_scores["upbeat"] += 1
            if params.get("valence", 0) < -0.5:
                style_scores["melancholic"] += 1
            if params.get("complexity", 0) > 0.7:
                style_scores["complex"] += 1
            if params.get("humanize", 0) > 0.7:
                style_scores["organic"] += 1

        # Normalize scores
        total = sum(style_scores.values())
        if total > 0:
            return {style: score / total for style, score in style_scores.items()}

        return {}

    def get_emotion_transitions(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze emotion transition patterns from user history.

        Tracks sequences: emotion A → emotion B in user selections.
        Returns frequency map: {emotion: {next_emotion: count}}

        Returns:
            Dictionary mapping current emotion to dictionary of next emotions and their frequencies
        """
        transitions = defaultdict(lambda: defaultdict(int))

        # Analyze emotion selection sequence
        selections = sorted(
            self.profile.emotion_selections,
            key=lambda s: s.timestamp
        )

        # Track transitions between consecutive selections
        for i in range(len(selections) - 1):
            current_emotion = selections[i].emotion_name
            next_emotion = selections[i + 1].emotion_name
            transitions[current_emotion][next_emotion] += 1

        # Also analyze transitions within generation events
        # (when user changes emotion between generations)
        for i in range(len(self.profile.midi_generations) - 1):
            current_event = self.profile.midi_generations[i]
            next_event = self.profile.midi_generations[i + 1]

            if current_event.emotion and next_event.emotion:
                transitions[current_event.emotion][next_event.emotion] += 1

        return {emotion: dict(next_emotions) for emotion, next_emotions in transitions.items()}

    def get_parameter_preferences(self) -> Dict[str, Dict[str, Any]]:
        """
        Get user's preferred parameter values with statistics.

        Returns:
            Dictionary mapping parameter name to stats dict with:
            - mean: Average value
            - std_dev: Standard deviation
            - preferred_range: (min, max) tuple of preferred range
            - median: Median value
            - adjustment_count: Number of times this parameter was adjusted
        """
        # Use the existing parameter statistics from the model
        stats = self.model.get_parameter_statistics()

        # Enhance with preferred ranges
        preferred_ranges = self.model.get_preferred_parameter_ranges()

        # Build comprehensive preferences dict
        preferences = {}
        for param_name, param_stats in stats.items():
            preferences[param_name] = {
                "mean": param_stats.get("average_value", 0.0),
                "median": param_stats.get("median_value", 0.0),
                "std_dev": param_stats.get("std_dev", 0.0),
                "min": param_stats.get("min_value", 0.0),
                "max": param_stats.get("max_value", 0.0),
                "adjustment_count": param_stats.get("adjustment_count", 0),
                "preferred_range": preferred_ranges.get(param_name, (0.0, 1.0)),
                "most_used_range": param_stats.get("most_used_range", (0.0, 0.1)),
            }

        return preferences

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of user preferences.

        Returns:
            Dictionary with all key statistics
        """
        return {
            "total_adjustments": len(self.profile.parameter_adjustments),
            "total_emotions": len(self.profile.emotion_selections),
            "total_generations": len(self.profile.midi_generations),
            "acceptance_rate": self.model.get_acceptance_rate(),
            "most_used_emotions": self.model.get_emotion_preferences(),
            "parameter_statistics": self.model.get_parameter_statistics(),
            "preferred_ranges": self.model.get_preferred_parameter_ranges(),
            "emotion_ranges": self.get_most_used_emotion_ranges(),
            "tempo_range": self.get_preferred_tempo_range(),
            "common_combinations": self.get_common_parameter_combinations(),
            "rule_break_preferences": self.get_rule_break_preferences(),
            "style_preferences": self.get_style_preferences(),
        }
