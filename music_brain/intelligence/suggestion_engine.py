"""
Intelligent Suggestion Engine

Provides context-aware suggestions for parameter adjustments, emotions, and rule-breaks.
Part of Phase 3 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random
from pathlib import Path


class SuggestionType(Enum):
    """Types of suggestions."""
    PARAMETER = "parameter"
    EMOTION = "emotion"
    RULE_BREAK = "rule_break"
    STYLE = "style"
    TRANSITION = "transition"


class SuggestionConfidence(Enum):
    """Confidence levels for suggestions."""
    LOW = "low"  # 0.0-0.4
    MEDIUM = "medium"  # 0.4-0.7
    HIGH = "high"  # 0.7-1.0


@dataclass
class Suggestion:
    """A single suggestion."""
    suggestion_type: SuggestionType
    title: str
    description: str
    action: Dict[str, Any]  # What to do (parameter changes, emotion, etc.)
    confidence: float  # 0.0 to 1.0
    explanation: str  # Why this suggestion is made
    source: str  # "user_history", "musical_theory", "pattern_detection", etc.

    def get_confidence_level(self) -> SuggestionConfidence:
        """Get confidence level enum."""
        if self.confidence >= 0.7:
            return SuggestionConfidence.HIGH
        elif self.confidence >= 0.4:
            return SuggestionConfidence.MEDIUM
        else:
            return SuggestionConfidence.LOW


class SuggestionEngine:
    """
    Generates intelligent suggestions based on context and user preferences.

    Usage:
        engine = SuggestionEngine(preference_model)
        suggestions = engine.generate_suggestions(current_state)
    """

    def __init__(self, preference_model=None, context_analyzer=None):
        """
        Initialize suggestion engine.

        Args:
            preference_model: UserPreferenceModel instance
            context_analyzer: ContextAnalyzer instance (optional)
        """
        self.preference_model = preference_model
        self.context_analyzer = context_analyzer

        # Optional LLaMA ONNX generator (lazy, best-effort)
        self._llama_generator = self._init_llama_generator()

        # Create analyzer if preference model is available
        self.preference_analyzer = None
        if self.preference_model:
            from music_brain.learning.preference_analyzer import PreferenceAnalyzer
            self.preference_analyzer = PreferenceAnalyzer(self.preference_model)

        # Musical knowledge
        self.emotion_transitions = self._build_emotion_transition_map()
        self.parameter_correlations = self._build_parameter_correlations()

    def generate_suggestions(
        self,
        current_state: Dict[str, Any],
        max_suggestions: int = 5
    ) -> List[Suggestion]:
        """
        Generate suggestions based on current musical state.

        Args:
            current_state: Current state dict with parameters, emotion, etc.
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggestions, sorted by confidence
        """
        suggestions = []

        # Optional LLaMA-backed hint (off unless explicitly requested)
        llama_hint = self._maybe_llama_suggestion(current_state)
        if llama_hint:
            suggestions.append(llama_hint)

        # Generate different types of suggestions
        suggestions.extend(self._generate_parameter_suggestions(current_state))
        suggestions.extend(self._generate_emotion_suggestions(current_state))
        suggestions.extend(self._generate_rule_break_suggestions(current_state))
        suggestions.extend(self._generate_style_suggestions(current_state))

        # Enhance suggestions with better confidence and explanations
        for suggestion in suggestions:
            # Recalculate confidence with unified method
            suggestion.confidence = self._calculate_suggestion_confidence(suggestion, current_state)
            # Enhance explanation
            suggestion.explanation = self._explain_suggestion(suggestion, current_state)

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions[:max_suggestions]

    def _generate_parameter_suggestions(
        self,
        current_state: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate parameter adjustment suggestions."""
        suggestions = []

        if not self.preference_analyzer:
            return suggestions

        current_params = current_state.get("parameters", {})
        preferences = self.preference_analyzer.get_parameter_preferences()

        # Suggest adjustments based on user's typical preferences
        for param_name, pref in preferences.items():
            current_value = current_params.get(param_name)
            if current_value is None:
                continue

            preferred_mean = pref.get("mean", current_value)
            difference = preferred_mean - current_value

            # Only suggest if difference is significant
            if abs(difference) > 0.1:
                direction = "increase" if difference > 0 else "decrease"
                target_value = preferred_mean

                # Calculate confidence from standard deviation (lower std = higher confidence)
                std_dev = pref.get("std_dev", 1.0)
                adjustment_count = pref.get("adjustment_count", 0)

                # Higher confidence if:
                # - Lower standard deviation (more consistent preferences)
                # - More adjustments (more data)
                confidence_base = max(0.3, min(0.9, 1.0 - (std_dev / 2.0)))
                confidence_boost = min(0.1, adjustment_count / 20.0)  # Boost for more data
                confidence_score = min(0.95, confidence_base + confidence_boost)

                suggestion = Suggestion(
                    suggestion_type=SuggestionType.PARAMETER,
                    title=f"{direction.title()} {param_name}",
                    description=f"Based on your history, you typically prefer {param_name} around {preferred_mean:.2f}",
                    action={"parameter": param_name, "target_value": target_value},
                    confidence=confidence_score,
                    explanation=f"Your past {param_name} adjustments average to {preferred_mean:.2f} (based on {adjustment_count} adjustments)",
                    source="user_history"
                )
                suggestions.append(suggestion)

        return suggestions

    def _generate_emotion_suggestions(
        self,
        current_state: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate emotion transition suggestions."""
        suggestions = []

        current_emotion = current_state.get("emotion")
        if not current_emotion:
            return suggestions

        # Get emotion transitions from user history
        if self.preference_analyzer:
            transition_map = self.preference_analyzer.get_emotion_transitions()
            transitions = transition_map.get(current_emotion, {})
            # Fall back to musical theory if no user history
            if not transitions:
                transitions = self.emotion_transitions.get(current_emotion, {})
        else:
            transitions = self.emotion_transitions.get(current_emotion, {})

        # Suggest common transitions
        for target_emotion, frequency in sorted(
            transitions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:
            if target_emotion != current_emotion:
                confidence = min(0.8, frequency / 5.0)  # Normalize by max expected frequency

                suggestion = Suggestion(
                    suggestion_type=SuggestionType.EMOTION,
                    title=f"Try transitioning to {target_emotion}",
                    description=f"{current_emotion} often pairs well with {target_emotion}",
                    action={"emotion": target_emotion},
                    confidence=confidence,
                    explanation=f"Based on musical theory, {current_emotion} → {target_emotion} creates emotional progression",
                    source="musical_theory" if not self.preference_model else "user_history"
                )
                suggestions.append(suggestion)

        return suggestions

    def _generate_rule_break_suggestions(
        self,
        current_state: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate rule-breaking suggestions based on emotion."""
        suggestions = []

        current_emotion = current_state.get("emotion", "").lower()
        current_rule_breaks = set(current_state.get("rule_breaks", []))

        # Get user's rule-break preferences
        user_kept_breaks = set()
        user_removed_breaks = set()
        if self.preference_analyzer:
            rule_prefs = self.preference_analyzer.get_rule_break_preferences()
            user_kept_breaks = set(rule_prefs.get("kept", {}).keys())
            user_removed_breaks = set(rule_prefs.get("removed", {}).keys())

        # Map emotions to appropriate rule breaks
        emotion_to_rule_breaks = {
            "grief": ["HARMONY_AvoidTonicResolution", "HARMONY_ModalInterchange"],
            "anger": ["RHYTHM_ConstantDisplacement", "PRODUCTION_Distortion"],
            "longing": ["HARMONY_UnresolvedDissonance", "MELODY_AvoidResolution"],
            "melancholy": ["HARMONY_ModalInterchange", "PRODUCTION_PitchImperfection"],
            "euphoria": ["ARRANGEMENT_ExtremeDynamicRange", "MELODY_ExcessiveRepetition"],
            "anxiety": ["RHYTHM_ConstantDisplacement", "HARMONY_UnresolvedDissonance"],
            "hope": ["HARMONY_ModalInterchange", "MELODY_AntiClimax"],
        }

        suggested_breaks = emotion_to_rule_breaks.get(current_emotion, [])

        # Prioritize rule-breaks user has kept before
        prioritized_breaks = []
        other_breaks = []
        for rule_break in suggested_breaks:
            if rule_break in user_removed_breaks:
                continue  # Don't suggest rule-breaks user has removed
            if rule_break not in current_rule_breaks:
                if rule_break in user_kept_breaks:
                    prioritized_breaks.append(rule_break)
                else:
                    other_breaks.append(rule_break)

        # Add prioritized breaks first
        for rule_break in prioritized_breaks + other_breaks:
            confidence = 0.8 if rule_break in user_kept_breaks else 0.7
            source = "user_history" if rule_break in user_kept_breaks else "emotion_analysis"

            suggestion = Suggestion(
                suggestion_type=SuggestionType.RULE_BREAK,
                title=f"Add {rule_break}",
                description=f"This rule-break enhances the {current_emotion} emotion",
                action={"add_rule_break": rule_break},
                confidence=confidence,
                explanation=f"{rule_break} is commonly used to express {current_emotion}" +
                           (". You've used this before." if rule_break in user_kept_breaks else ""),
                source=source
            )
            suggestions.append(suggestion)

        return suggestions

    def _generate_style_suggestions(
        self,
        current_state: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate style-based suggestions."""
        suggestions = []

        if not self.preference_analyzer:
            return suggestions

        current_params = current_state.get("parameters", {})
        current_tempo = current_params.get("tempo", 120)
        current_emotion = current_state.get("emotion", "").lower()

        # Get user's style preferences
        style_prefs = self.preference_analyzer.get_style_preferences()
        tempo_range = self.preference_analyzer.get_preferred_tempo_range()

        # Suggest tempo based on emotion and user history
        emotion_tempo_map = {
            "grief": 70,
            "melancholy": 75,
            "longing": 80,
            "hope": 100,
            "euphoria": 130,
            "anger": 140,
        }

        # Use user's preferred tempo range if available, otherwise use emotion-based default
        if tempo_range:
            suggested_tempo = (tempo_range[0] + tempo_range[1]) // 2
            confidence = 0.75
            source = "user_history"
            explanation = f"Based on your history, you prefer tempos around {suggested_tempo} BPM"
        elif current_emotion in emotion_tempo_map:
            suggested_tempo = emotion_tempo_map[current_emotion]
            confidence = 0.65
            source = "musical_theory"
            explanation = f"Tempo {suggested_tempo} BPM is typical for {current_emotion} emotion"
        else:
            return suggestions

        if abs(current_tempo - suggested_tempo) > 10:
            suggestion = Suggestion(
                suggestion_type=SuggestionType.STYLE,
                title=f"Try tempo {suggested_tempo} BPM",
                description=f"{current_emotion} songs often work well at {suggested_tempo} BPM",
                action={"parameter": "tempo", "target_value": suggested_tempo},
                confidence=confidence,
                explanation=explanation,
                source=source
            )
            suggestions.append(suggestion)

        # Suggest style preferences if user has strong preferences
        if style_prefs:
            top_style = max(style_prefs.items(), key=lambda x: x[1])
            if top_style[1] > 0.5:  # Strong preference
                suggestion = Suggestion(
                    suggestion_type=SuggestionType.STYLE,
                    title=f"Try {top_style[0]} style",
                    description=f"You often prefer {top_style[0]} style in your music",
                    action={"style": top_style[0]},
                    confidence=top_style[1],
                    explanation=f"Based on your accepted generations, you prefer {top_style[0]} style",
                    source="user_history"
                )
                suggestions.append(suggestion)

        return suggestions

    # -------------------------------------------------------------------------
    # LLaMA ONNX (optional) integration
    # -------------------------------------------------------------------------
    def _init_llama_generator(self):
        """Best-effort load of LLaMA ONNX generator; never raises."""
        try:
            from music_brain.integrations.llama_onnx import (
                build_llama_generator,
                default_llama_config_path,
            )
            # Allow override via env/config in the future; use default for now.
            return build_llama_generator(default_llama_config_path())
        except Exception:
            return None

    def _maybe_llama_suggestion(
        self,
        current_state: Dict[str, Any],
    ) -> Optional[Suggestion]:
        """
        Optionally generate a single LLaMA-backed textual hint.

        Activation (opt-in):
        - current_state["use_llama"] is truthy, OR
        - current_state["llama_prompt"] provided.
        """
        if not self._llama_generator:
            return None

        use_llama = bool(current_state.get("use_llama"))
        custom_prompt = current_state.get("llama_prompt")
        if not (use_llama or custom_prompt):
            return None

        prompt = custom_prompt or self._build_llama_prompt(current_state)
        try:
            text = self._llama_generator.generate(prompt)
        except Exception:
            return None

        if not text:
            return None

        return Suggestion(
            suggestion_type=SuggestionType.STYLE,
            title="LLM idea",
            description=text,
            action={"llama_text": text},
            confidence=0.55,
            explanation="Generated via LLaMA ONNX (optional, user-triggered)",
            source="llama_onnx",
        )

    def _build_llama_prompt(self, current_state: Dict[str, Any]) -> str:
        """Build a compact prompt from current state."""
        emotion = current_state.get("emotion", "neutral")
        params = current_state.get("parameters", {})
        chords = current_state.get("chords", [])
        style = current_state.get("style", "")
        rule_breaks = current_state.get("rule_breaks", [])

        return (
            "You are a music co-pilot. Provide one concise suggestion.\n"
            f"Emotion: {emotion}\n"
            f"Style: {style}\n"
            f"Chords: {chords}\n"
            f"Parameters: {params}\n"
            f"RuleBreaks: {rule_breaks}\n"
            "Return one short idea (<=25 words)."
        )

    def _build_emotion_transition_map(self) -> Dict[str, Dict[str, int]]:
        """Build map of common emotion transitions."""
        return {
            "grief": {"longing": 5, "acceptance": 4, "hope": 3},
            "longing": {"grief": 4, "hope": 5, "euphoria": 2},
            "hope": {"euphoria": 5, "longing": 3, "acceptance": 4},
            "anger": {"defiance": 4, "determination": 3, "catharsis": 5},
            "melancholy": {"grief": 4, "nostalgia": 5, "acceptance": 3},
            "anxiety": {"fear": 4, "determination": 3, "catharsis": 2},
        }

    def _build_parameter_correlations(self) -> Dict[str, List[str]]:
        """Build map of correlated parameters."""
        return {
            "valence": ["intensity", "dynamics"],
            "arousal": ["tempo", "complexity"],
            "intensity": ["dynamics", "complexity"],
        }

    def _get_user_similar_states(
        self,
        current_state: Dict[str, Any],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find past states similar to current state.

        Compares parameter combinations to find what user did in similar situations.

        Args:
            current_state: Current state dictionary
            max_results: Maximum number of similar states to return

        Returns:
            List of similar state dictionaries with similarity score
        """
        if not self.preference_model:
            return []

        current_params = current_state.get("parameters", {})
        similar_states = []

        # Compare with past MIDI generation events
        for event in self.preference_model.get_profile().midi_generations:
            if not event.parameters:
                continue

            # Calculate similarity score (simple Euclidean distance on normalized parameters)
            similarity = self._calculate_state_similarity(current_params, event.parameters)

            if similarity > 0.5:  # Threshold for "similar"
                similar_states.append({
                    "event": event,
                    "similarity": similarity,
                    "parameters": event.parameters,
                    "emotion": event.emotion,
                    "rule_breaks": event.rule_breaks,
                    "accepted": event.accepted,
                })

        # Sort by similarity and return top N
        similar_states.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_states[:max_results]

    def _calculate_state_similarity(
        self,
        params1: Dict[str, float],
        params2: Dict[str, float]
    ) -> float:
        """
        Calculate similarity between two parameter states.

        Returns:
            Similarity score from 0.0 (completely different) to 1.0 (identical)
        """
        if not params1 or not params2:
            return 0.0

        # Get common parameters
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 0.0

        # Calculate normalized Euclidean distance
        squared_diffs = []
        for param in common_params:
            val1 = params1[param]
            val2 = params2[param]

            # Normalize based on parameter range
            if param in ["valence", "feel"]:
                # Range is -1 to 1, so max diff is 2
                diff = abs(val1 - val2) / 2.0
            else:
                # Range is 0 to 1, so max diff is 1
                diff = abs(val1 - val2)

            squared_diffs.append(diff ** 2)

        # Average squared difference
        avg_squared_diff = sum(squared_diffs) / len(squared_diffs) if squared_diffs else 1.0

        # Convert to similarity (1.0 = identical, 0.0 = completely different)
        similarity = 1.0 - (avg_squared_diff ** 0.5)

        return max(0.0, min(1.0, similarity))

    def _calculate_suggestion_confidence(
        self,
        suggestion: Suggestion,
        current_state: Dict[str, Any]
    ) -> float:
        """
        Calculate unified confidence score for a suggestion.

        Factors in:
        - User history strength
        - Parameter correlation
        - Musical theory support

        Args:
            suggestion: The suggestion to calculate confidence for
            current_state: Current musical state

        Returns:
            Confidence score from 0.0 to 1.0
        """
        base_confidence = suggestion.confidence

        # Boost confidence if user has similar past states
        if self.preference_model:
            similar_states = self._get_user_similar_states(current_state, max_results=3)
            if similar_states:
                # Average similarity of top similar states
                avg_similarity = sum(s["similarity"] for s in similar_states) / len(similar_states)
                # Boost confidence by up to 0.15 based on similarity
                confidence_boost = avg_similarity * 0.15
                base_confidence = min(0.95, base_confidence + confidence_boost)

        # Boost confidence if suggestion aligns with parameter correlations
        if suggestion.suggestion_type == SuggestionType.PARAMETER:
            param_name = suggestion.action.get("parameter")
            if param_name and param_name in self.parameter_correlations:
                # If correlated parameters are also being adjusted, boost confidence
                correlated = self.parameter_correlations[param_name]
                current_params = current_state.get("parameters", {})
                correlated_adjusted = sum(1 for p in correlated if p in current_params)
                if correlated_adjusted > 0:
                    base_confidence = min(0.95, base_confidence + 0.1)

        # Reduce confidence if suggestion conflicts with current state
        if suggestion.suggestion_type == SuggestionType.EMOTION:
            current_emotion = current_state.get("emotion", "")
            target_emotion = suggestion.action.get("emotion", "")
            # Large emotion jumps reduce confidence slightly
            # (This is a simplified check - could be enhanced with emotion distance calculation)
            if current_emotion and target_emotion:
                # Simple heuristic: if emotions are very different, slightly reduce confidence
                base_confidence = max(0.3, base_confidence - 0.05)

        return max(0.0, min(1.0, base_confidence))

    def _explain_suggestion(
        self,
        suggestion: Suggestion,
        current_state: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation for a suggestion.

        Args:
            suggestion: The suggestion to explain
            current_state: Current musical state

        Returns:
            Human-readable explanation string
        """
        explanation = suggestion.explanation

        # Enhance explanation based on user history
        if self.preference_model and suggestion.source == "user_history":
            similar_states = self._get_user_similar_states(current_state, max_results=1)
            if similar_states:
                similar = similar_states[0]
                if similar["accepted"] is True:
                    explanation += " You've used similar settings before and accepted them."
                elif similar["accepted"] is False:
                    explanation += " You've tried similar settings before but didn't accept them."

        # Add context-specific details
        if suggestion.suggestion_type == SuggestionType.PARAMETER:
            param_name = suggestion.action.get("parameter")
            current_value = current_state.get("parameters", {}).get(param_name)
            target_value = suggestion.action.get("target_value")
            if current_value is not None and target_value is not None:
                diff = abs(target_value - current_value)
                if diff > 0.3:
                    explanation += f" This is a significant change from your current {param_name} of {current_value:.2f}."

        elif suggestion.suggestion_type == SuggestionType.EMOTION:
            current_emotion = current_state.get("emotion", "")
            target_emotion = suggestion.action.get("emotion", "")
            if current_emotion and target_emotion:
                explanation += f" This would transition from {current_emotion} to {target_emotion}."

        elif suggestion.suggestion_type == SuggestionType.RULE_BREAK:
            rule_break = suggestion.action.get("add_rule_break", "")
            if rule_break:
                explanation += f" This rule-break will modify the harmonic/rhythmic structure to enhance the emotional expression."

        return explanation


class ContextAnalyzer:
    """
    Analyzes current musical context to inform suggestions.
    """

    def analyze_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current musical context.

        Args:
            state: Current state dictionary

        Returns:
            Context analysis dictionary
        """
        analysis = {
            "emotion_category": self._categorize_emotion(state.get("emotion")),
            "parameter_ranges": self._analyze_parameter_ranges(state.get("parameters", {})),
            "complexity_level": self._assess_complexity(state),
        }

        return analysis

    def _categorize_emotion(self, emotion: Optional[str]) -> Optional[str]:
        """Categorize emotion into broad category."""
        if not emotion:
            return None

        emotion_lower = emotion.lower()
        if emotion_lower in ["grief", "sadness", "melancholy", "longing"]:
            return "negative_low_energy"
        elif emotion_lower in ["anger", "rage", "frustration"]:
            return "negative_high_energy"
        elif emotion_lower in ["hope", "joy", "euphoria"]:
            return "positive_high_energy"
        elif emotion_lower in ["peace", "acceptance", "calm"]:
            return "positive_low_energy"
        else:
            return "neutral"

    def _analyze_parameter_ranges(self, parameters: Dict[str, float]) -> Dict[str, str]:
        """Analyze if parameters are in low/medium/high ranges."""
        ranges = {}

        for param_name, value in parameters.items():
            if param_name in ["valence", "feel"]:
                # -1 to 1 range
                if value < -0.33:
                    ranges[param_name] = "low"
                elif value > 0.33:
                    ranges[param_name] = "high"
                else:
                    ranges[param_name] = "medium"
            else:
                # 0 to 1 range
                if value < 0.33:
                    ranges[param_name] = "low"
                elif value > 0.67:
                    ranges[param_name] = "high"
                else:
                    ranges[param_name] = "medium"

        return ranges

    def _assess_complexity(self, state: Dict[str, Any]) -> str:
        """Assess overall complexity level."""
        params = state.get("parameters", {})
        complexity_value = params.get("complexity", 0.5)

        if complexity_value < 0.33:
            return "simple"
        elif complexity_value > 0.67:
            return "complex"
        else:
            return "moderate"


def main():
    """Example usage."""
    from music_brain.learning.user_preferences import UserPreferenceModel

    model = UserPreferenceModel()
    engine = SuggestionEngine(preference_model=model)

    current_state = {
        "emotion": "grief",
        "parameters": {
            "valence": -0.5,
            "arousal": 0.4,
            "intensity": 0.6,
            "tempo": 120,
        },
        "rule_breaks": []
    }

    suggestions = engine.generate_suggestions(current_state, max_suggestions=5)

    print("Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion.title}")
        print(f"   {suggestion.description}")
        print(f"   Confidence: {suggestion.confidence:.2f} ({suggestion.get_confidence_level().value})")
        print(f"   Explanation: {suggestion.explanation}")


if __name__ == "__main__":
    main()
