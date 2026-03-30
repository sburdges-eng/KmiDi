"""
Context Analyzer

Provides context-aware analysis of musical state for better suggestions.
Part of Phase 3 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MusicalContext:
    """Musical context information."""
    emotion_category: Optional[str] = None
    parameter_ranges: Dict[str, str] = None  # "low", "medium", "high"
    complexity_level: str = "moderate"
    harmonic_state: Optional[str] = None  # "tonic", "dominant", "subdominant", etc.
    rhythmic_state: Optional[str] = None  # "straight", "swung", "polyrhythmic", etc.

    def __post_init__(self):
        if self.parameter_ranges is None:
            self.parameter_ranges = {}


class ContextAnalyzer:
    """
    Analyzes current musical context to inform suggestions.

    Usage:
        analyzer = ContextAnalyzer()
        context = analyzer.analyze(state)
    """

    def analyze(self, state: Dict[str, Any]) -> MusicalContext:
        """
        Analyze current musical context.

        Args:
            state: Current state dictionary with parameters, emotion, chords, etc.

        Returns:
            MusicalContext object
        """
        context = MusicalContext()

        context.emotion_category = self._categorize_emotion(state.get("emotion"))
        context.parameter_ranges = self._analyze_parameter_ranges(state.get("parameters", {}))
        context.complexity_level = self._assess_complexity(state)
        context.harmonic_state = self._analyze_harmonic_state(state)
        context.rhythmic_state = self._analyze_rhythmic_state(state)

        return context

    def _categorize_emotion(self, emotion: Optional[str]) -> Optional[str]:
        """Categorize emotion into broad category."""
        if not emotion:
            return None

        emotion_lower = emotion.lower()

        categories = {
            "negative_low_energy": ["grief", "sadness", "melancholy", "longing", "sorrow"],
            "negative_high_energy": ["anger", "rage", "frustration", "defiance", "anxiety"],
            "positive_high_energy": ["hope", "joy", "euphoria", "excitement", "determination"],
            "positive_low_energy": ["peace", "acceptance", "calm", "tenderness", "surrender"],
            "neutral": ["nostalgia", "dissociation"],
        }

        for category, emotions in categories.items():
            if emotion_lower in emotions:
                return category

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

    def _analyze_harmonic_state(self, state: Dict[str, Any]) -> Optional[str]:
        """Analyze harmonic state from chords/progression."""
        chords = state.get("chords", [])
        if not chords:
            return None

        # Simple analysis: look at last chord
        # In production, this would do proper harmonic analysis
        last_chord = chords[-1] if chords else None
        if last_chord:
            chord_lower = str(last_chord).lower()
            if "maj" in chord_lower or chord_lower.endswith("m"):
                return "tonic"
            elif "dom" in chord_lower or "7" in chord_lower:
                return "dominant"
            elif "sus" in chord_lower:
                return "suspended"

        return None

    def _analyze_rhythmic_state(self, state: Dict[str, Any]) -> Optional[str]:
        """Analyze rhythmic state from feel parameter."""
        params = state.get("parameters", {})
        feel = params.get("feel", 0.0)
        humanize = params.get("humanize", 0.5)

        if abs(feel) < 0.2 and humanize < 0.3:
            return "straight"
        elif feel < -0.3:
            return "laid_back"
        elif feel > 0.3:
            return "pushed"
        elif humanize > 0.7:
            return "humanized"
        else:
            return "moderate"

    def get_contextual_suggestions(self, context: MusicalContext) -> List[str]:
        """
        Get contextual suggestion hints based on analysis.

        Args:
            context: MusicalContext object

        Returns:
            List of suggestion hints
        """
        hints = []

        # Complexity-based hints
        if context.complexity_level == "simple":
            hints.append("Consider increasing complexity for more interest")
        elif context.complexity_level == "complex":
            hints.append("Consider simplifying for clearer expression")

        # Emotion category hints
        if context.emotion_category == "negative_low_energy":
            hints.append("Low energy emotions often benefit from slower tempos and darker harmonies")
        elif context.emotion_category == "negative_high_energy":
            hints.append("High energy negative emotions work well with aggressive rhythms and distortion")

        # Parameter range hints
        low_params = [p for p, r in context.parameter_ranges.items() if r == "low"]
        if low_params:
            hints.append(f"Parameters currently low: {', '.join(low_params)}")

        return hints


def main():
    """Example usage."""
    analyzer = ContextAnalyzer()

    state = {
        "emotion": "grief",
        "parameters": {
            "valence": -0.7,
            "arousal": 0.3,
            "intensity": 0.6,
            "complexity": 0.4,
            "humanize": 0.5,
            "feel": 0.0,
        },
        "chords": ["Am", "Dm", "F", "C"]
    }

    context = analyzer.analyze(state)
    print("Musical Context:")
    print(f"  Emotion Category: {context.emotion_category}")
    print(f"  Complexity: {context.complexity_level}")
    print(f"  Parameter Ranges: {context.parameter_ranges}")
    print(f"  Harmonic State: {context.harmonic_state}")
    print(f"  Rhythmic State: {context.rhythmic_state}")

    hints = analyzer.get_contextual_suggestions(context)
    print("\nContextual Hints:")
    for hint in hints:
        print(f"  - {hint}")


if __name__ == "__main__":
    main()
