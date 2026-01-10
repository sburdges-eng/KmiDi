"""
Natural Language Processing for Music Feedback

Interprets user descriptions and maps them to musical parameters.
Part of Phase 6 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict


class FeedbackType(Enum):
    """Types of user feedback."""
    NEGATIVE = "negative"  # "less X", "not enough Y", "too much Z"
    POSITIVE = "positive"  # "more X", "add Y", "needs Z"
    EMOTIONAL = "emotional"  # "more melancholic", "less aggressive"
    MUSICAL = "musical"  # "more groove", "better rhythm"
    PART_SPECIFIC = "part_specific"  # "bass line doesn't slap"


class Intent(Enum):
    """User intent extracted from feedback."""
    INCREASE = "increase"
    DECREASE = "decrease"
    ADD = "add"
    REMOVE = "remove"
    CHANGE = "change"
    KEEP = "keep"


@dataclass
class InterpretedFeedback:
    """Result of natural language interpretation."""
    intent: Intent
    target_element: Optional[str]  # "bass", "drums", "melody", "harmony", etc.
    direction: str  # "more", "less", "add", "remove"
    descriptor: str  # "groove", "melancholic", "chug", etc.
    confidence: float  # 0.0 to 1.0
    parameter_changes: Dict[str, float]  # Proposed parameter changes
    explanation: str  # Why these changes were proposed


class NaturalLanguageProcessor:
    """
    Processes natural language feedback and maps to musical parameters.

    Usage:
        processor = NaturalLanguageProcessor()
        result = processor.interpret("bass line doesn't slap")
        # Returns InterpretedFeedback with parameter changes
    """

    def __init__(self, user_vocabulary: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize NLP processor.

        Args:
            user_vocabulary: User-specific terminology dictionary (learned over time)
        """
        self.user_vocabulary = user_vocabulary or {}
        self.musical_terms = self._build_musical_term_map()
        self.emotional_terms = self._build_emotional_term_map()
        self.part_keywords = self._build_part_keywords()

    def interpret(self, feedback_text: str, current_state: Optional[Dict[str, Any]] = None) -> InterpretedFeedback:
        """
        Interpret user feedback and generate parameter changes.

        Args:
            feedback_text: User's natural language feedback
            current_state: Current musical state (parameters, emotion, etc.)

        Returns:
            InterpretedFeedback with proposed changes
        """
        feedback_lower = feedback_text.lower().strip()

        # Preprocess: remove punctuation, handle slang
        feedback_clean = self._preprocess(feedback_lower)

        # Extract intent
        intent = self._extract_intent(feedback_clean)

        # Identify target element (part-specific or general)
        target_element = self._identify_target(feedback_clean)

        # Extract direction and descriptor
        direction, descriptor = self._extract_direction_and_descriptor(feedback_clean)

        # Map to parameters
        parameter_changes, confidence = self._map_to_parameters(
            descriptor, direction, target_element, current_state
        )

        # Generate explanation
        explanation = self._generate_explanation(intent, descriptor, direction, target_element)

        return InterpretedFeedback(
            intent=intent,
            target_element=target_element,
            direction=direction,
            descriptor=descriptor,
            confidence=confidence,
            parameter_changes=parameter_changes,
            explanation=explanation
        )

    def _preprocess(self, text: str) -> str:
        """Preprocess text: remove punctuation, handle slang."""
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def _extract_intent(self, text: str) -> Intent:
        """Extract user intent from text."""
        increase_keywords = ["more", "add", "increase", "boost", "enhance", "needs", "wants"]
        decrease_keywords = ["less", "reduce", "decrease", "lower", "remove", "don't want", "doesn't need"]
        remove_keywords = ["remove", "delete", "get rid of", "eliminate"]
        keep_keywords = ["keep", "maintain", "preserve"]

        text_lower = text.lower()

        if any(kw in text_lower for kw in remove_keywords):
            return Intent.REMOVE
        elif any(kw in text_lower for kw in decrease_keywords):
            return Intent.DECREASE
        elif any(kw in text_lower for kw in increase_keywords):
            return Intent.INCREASE
        elif any(kw in text_lower for kw in keep_keywords):
            return Intent.KEEP
        else:
            # Default to change if unclear
            return Intent.CHANGE

    def _identify_target(self, text: str) -> Optional[str]:
        """Identify target element (which part: bass, drums, melody, etc.)."""
        text_lower = text.lower()

        for part, keywords in self.part_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return part

        return None  # General feedback

    def _extract_direction_and_descriptor(self, text: str) -> Tuple[str, str]:
        """Extract direction (more/less) and descriptor (groove, melancholic, etc.)."""
        direction = "more"  # Default
        descriptor = ""

        # Check for direction keywords
        if any(kw in text for kw in ["less", "not enough", "too little", "lacks"]):
            direction = "less"
        elif any(kw in text for kw in ["more", "needs", "wants", "add"]):
            direction = "more"
        elif any(kw in text for kw in ["remove", "delete", "get rid of"]):
            direction = "remove"

        # Extract descriptor (musical or emotional term)
        all_terms = {**self.musical_terms, **self.emotional_terms}

        # Check user vocabulary first
        for user_term, mapping in self.user_vocabulary.items():
            if user_term in text:
                descriptor = user_term
                break

        # Check standard terms
        if not descriptor:
            for term in all_terms.keys():
                if term in text:
                    descriptor = term
                    break

        # If no descriptor found, try to extract from context
        if not descriptor:
            # Look for adjectives or descriptive words
            words = text.split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    descriptor = word
                    break

        return direction, descriptor or "general"

    def _map_to_parameters(
        self,
        descriptor: str,
        direction: str,
        target_element: Optional[str],
        current_state: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], float]:
        """
        Map descriptor and direction to parameter changes.

        Returns:
            Tuple of (parameter_changes dict, confidence score)
        """
        changes = {}
        confidence = 0.7  # Default confidence

        # Check user vocabulary first
        if descriptor in self.user_vocabulary:
            mapping = self.user_vocabulary[descriptor]
            changes = mapping.get("parameters", {}).copy()
            confidence = mapping.get("confidence", 0.8)
            # Apply direction
            if direction == "less":
                changes = {k: -v for k, v in changes.items()}
            elif direction == "remove":
                changes = {k: -v * 1.5 for k, v in changes.items()}
        # Check musical terms
        elif descriptor in self.musical_terms:
            mapping = self.musical_terms[descriptor]
            changes = mapping.get("parameters", {}).copy()
            confidence = mapping.get("confidence", 0.75)
            # Apply direction
            if direction == "less":
                changes = {k: -v for k, v in changes.items()}
            elif direction == "remove":
                changes = {k: -v * 1.5 for k, v in changes.items()}
        # Check emotional terms
        elif descriptor in self.emotional_terms:
            mapping = self.emotional_terms[descriptor]
            changes = mapping.get("parameters", {}).copy()
            confidence = mapping.get("confidence", 0.7)
            # Apply direction
            if direction == "less":
                changes = {k: -v for k, v in changes.items()}
            elif direction == "remove":
                changes = {k: -v * 1.5 for k, v in changes.items()}
        else:
            # Generic mapping based on direction
            if direction == "more":
                changes = {"intensity": 0.1, "dynamics": 0.1}
            elif direction == "less":
                changes = {"intensity": -0.1, "dynamics": -0.1}
            confidence = 0.5  # Lower confidence for generic

        # Adjust for target element
        if target_element:
            part_adjustments = self._get_part_specific_adjustments(target_element, direction)
            # Merge adjustments (part-specific takes priority)
            for param, value in part_adjustments.items():
                changes[param] = changes.get(param, 0.0) + value

        # Clamp changes to reasonable ranges
        changes = {k: max(-0.5, min(0.5, v)) for k, v in changes.items()}

        return changes, confidence

    def _get_part_specific_adjustments(self, part: str, direction: str) -> Dict[str, float]:
        """Get parameter adjustments specific to a part."""
        adjustments = {
            "bass": {"dynamics": 0.15, "intensity": 0.1},
            "drums": {"dynamics": 0.2, "humanize": 0.1},
            "melody": {"intensity": 0.1, "complexity": 0.05},
            "harmony": {"complexity": 0.1, "intensity": 0.05},
        }

        base = adjustments.get(part.lower(), {})
        if direction == "less":
            return {k: -v for k, v in base.items()}
        elif direction == "remove":
            return {k: -v * 1.5 for k, v in base.items()}
        return base

    def _generate_explanation(
        self,
        intent: Intent,
        descriptor: str,
        direction: str,
        target_element: Optional[str]
    ) -> str:
        """Generate explanation for the interpretation."""
        parts = []

        if target_element:
            parts.append(f"For the {target_element} part,")

        if intent == Intent.INCREASE:
            parts.append(f"increasing {descriptor} will")
        elif intent == Intent.DECREASE:
            parts.append(f"decreasing {descriptor} will")
        elif intent == Intent.REMOVE:
            parts.append(f"removing {descriptor} will")
        else:
            parts.append(f"adjusting {descriptor} will")

        parts.append("enhance the musical expression.")

        return " ".join(parts)

    def learn_term(self, term: str, parameter_mapping: Dict[str, float], confidence: float = 0.8):
        """
        Learn a user-specific term and its parameter mapping.

        Args:
            term: User's term (e.g., "slaps", "chugs")
            parameter_mapping: Dictionary of parameter -> value changes
            confidence: Confidence in this mapping
        """
        self.user_vocabulary[term.lower()] = {
            "parameters": parameter_mapping,
            "confidence": confidence
        }

    def _build_musical_term_map(self) -> Dict[str, Dict[str, Any]]:
        """Build map of musical terms to parameter changes."""
        return {
            "chug": {
                "parameters": {"intensity": 0.2, "dynamics": 0.15, "humanize": -0.1},
                "confidence": 0.8
            },
            "slap": {
                "parameters": {"dynamics": 0.2, "humanize": 0.1, "intensity": 0.15},
                "confidence": 0.85
            },
            "groove": {
                "parameters": {"humanize": 0.15, "feel": 0.1, "dynamics": 0.1},
                "confidence": 0.8
            },
            "punchy": {
                "parameters": {"dynamics": 0.2, "intensity": 0.15, "humanize": -0.05},
                "confidence": 0.75
            },
            "smooth": {
                "parameters": {"humanize": 0.1, "dynamics": -0.1, "intensity": -0.05},
                "confidence": 0.75
            },
            "tight": {
                "parameters": {"humanize": -0.15, "intensity": 0.1},
                "confidence": 0.8
            },
            "loose": {
                "parameters": {"humanize": 0.2, "feel": -0.1},
                "confidence": 0.75
            },
        }

    def _build_emotional_term_map(self) -> Dict[str, Dict[str, Any]]:
        """Build map of emotional terms to parameter changes."""
        return {
            "melancholic": {
                "parameters": {"valence": -0.2, "intensity": -0.1, "dynamics": -0.1},
                "confidence": 0.8
            },
            "aggressive": {
                "parameters": {"arousal": 0.2, "intensity": 0.2, "dynamics": 0.15},
                "confidence": 0.8
            },
            "dark": {
                "parameters": {"valence": -0.15, "intensity": 0.1},
                "confidence": 0.75
            },
            "bright": {
                "parameters": {"valence": 0.15, "intensity": 0.1},
                "confidence": 0.75
            },
            "soft": {
                "parameters": {"dynamics": -0.2, "intensity": -0.1},
                "confidence": 0.8
            },
            "intense": {
                "parameters": {"intensity": 0.2, "dynamics": 0.15, "arousal": 0.1},
                "confidence": 0.8
            },
        }

    def _build_part_keywords(self) -> Dict[str, List[str]]:
        """Build map of part names to keywords."""
        return {
            "bass": ["bass", "bassline", "low end", "low frequencies"],
            "drums": ["drums", "drum", "percussion", "beat", "rhythm section"],
            "melody": ["melody", "melodic", "lead", "top line"],
            "harmony": ["harmony", "chord", "chords", "harmonic"],
            "pad": ["pad", "pads", "atmosphere", "ambient"],
            "strings": ["strings", "string", "orchestral"],
        }


class FeedbackInterpreter:
    """
    Interprets user feedback and generates parameter change proposals.

    Works with NaturalLanguageProcessor to provide preview and confirmation.
    """

    def __init__(self, nlp_processor: NaturalLanguageProcessor):
        """Initialize feedback interpreter."""
        self.nlp = nlp_processor
        self.feedback_history: List[Tuple[str, InterpretedFeedback]] = []

    def interpret_feedback(
        self,
        feedback_text: str,
        current_state: Dict[str, Any]
    ) -> InterpretedFeedback:
        """
        Interpret feedback and generate parameter change proposal.

        Args:
            feedback_text: User's natural language feedback
            current_state: Current musical state

        Returns:
            InterpretedFeedback with proposed changes
        """
        result = self.nlp.interpret(feedback_text, current_state)
        self.feedback_history.append((feedback_text, result))
        return result

    def refine_interpretation(
        self,
        original_feedback: str,
        correction: str
    ) -> InterpretedFeedback:
        """
        Refine interpretation based on user correction.

        Args:
            original_feedback: Original feedback text
            correction: User's correction/clarification

        Returns:
            Refined interpretation
        """
        # Combine original and correction
        combined = f"{original_feedback} {correction}"
        result = self.nlp.interpret(combined)

        # Learn from correction
        # (In production, update user vocabulary based on correction)

        return result

    def get_feedback_history(self) -> List[Tuple[str, InterpretedFeedback]]:
        """Get history of feedback interpretations."""
        return self.feedback_history.copy()


def main():
    """Example usage."""
    processor = NaturalLanguageProcessor()

    feedback_examples = [
        "I don't want so much chug",
        "bass line doesn't slap",
        "make it more melancholic but keep the energy",
        "drums too quiet",
        "needs more groove",
    ]

    current_state = {
        "parameters": {
            "valence": -0.3,
            "arousal": 0.5,
            "intensity": 0.6,
            "dynamics": 0.5,
        },
        "emotion": "grief"
    }

    print("Natural Language Feedback Interpretation:")
    print("=" * 60)

    for feedback in feedback_examples:
        result = processor.interpret(feedback, current_state)
        print(f"\nFeedback: \"{feedback}\"")
        print(f"Intent: {result.intent.value}")
        print(f"Target: {result.target_element or 'general'}")
        print(f"Direction: {result.direction}")
        print(f"Descriptor: {result.descriptor}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Parameter Changes: {result.parameter_changes}")
        print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    main()
