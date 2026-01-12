"""Three-phase intent processing: Wound → Emotion → Rule-breaks.

This module implements the therapeutic intent processing pipeline that translates
emotional wounds into musical expression through intentional rule-breaking.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionNode, EmotionCategory


class IntentPhase(Enum):
    """Three phases of intent processing."""
    WOUND = "wound"  # Initial trauma or trigger
    EMOTION = "emotion"  # Emotional response
    RULE_BREAK = "rule_break"  # Musical rule violations


@dataclass
class Wound:
    """Represents the initial wound or trigger.
    
    Attributes:
        description: Textual description of the wound
        intensity: Intensity level (0.0 to 1.0)
        source: Origin of the wound (internal/external/user_input)
        timestamp: Optional timestamp when wound occurred
        keywords: Extracted keywords from description
    """
    description: str
    intensity: float  # 0.0 to 1.0
    source: str  # Internal or external
    timestamp: Optional[float] = None
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Extract keywords from description."""
        if not self.keywords:
            # Extract meaningful words (3+ characters, not common stop words)
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
            words = re.findall(r'\b\w{3,}\b', self.description.lower())
            self.keywords = [w for w in words if w not in stop_words]
        
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


@dataclass
class RuleBreak:
    """Represents intentional musical rule violations.
    
    Attributes:
        rule_type: Category of rule break (harmony, rhythm, dynamics, etc.)
        severity: Severity level (0.0 to 1.0)
        description: Human-readable description
        musical_impact: Dictionary of musical parameters affected
        justification: Why this rule break is needed for expression
    """
    rule_type: str  # e.g., "harmony", "rhythm", "dynamics", "melody", "timbre"
    severity: float  # 0.0 to 1.0
    description: str
    musical_impact: Dict[str, Any]
    justification: str = ""


class IntentProcessor:
    """
    Processes therapeutic intent through three phases:
    1. Wound identification and analysis
    2. Emotional mapping using thesaurus
    3. Musical rule-breaking for authentic expression
    
    The processor uses keyword matching and emotional proximity to map
    wounds to emotions, then generates rule breaks that allow authentic
    musical expression of difficult emotions.
    
    Example:
        >>> processor = IntentProcessor()
        >>> wound = Wound("feeling of loss and grief", 0.9, "user_input")
        >>> result = processor.process_intent(wound)
        >>> print(result["emotion"].name)
        'grief'
    """
    
    # Emotion keyword mappings for better wound-to-emotion matching
    EMOTION_KEYWORDS: Dict[str, List[str]] = {
        "grief": ["loss", "grief", "mourning", "bereavement", "death", "died", "passed"],
        "sorrow": ["sad", "sorrow", "unhappy", "down", "blue", "depressed"],
        "melancholy": ["melancholy", "nostalgia", "longing", "yearning", "wistful"],
        "despair": ["despair", "hopeless", "helpless", "defeated", "overwhelmed"],
        "rage": ["rage", "fury", "furious", "enraged", "livid", "incensed"],
        "anger": ["anger", "angry", "mad", "annoyed", "irritated", "frustrated"],
        "resentment": ["resentment", "bitter", "grudge", "spite", "vengeful"],
        "terror": ["terror", "horror", "dread", "panic", "frightened", "scared"],
        "anxiety": ["anxiety", "worried", "nervous", "uneasy", "apprehensive"],
        "dread": ["dread", "foreboding", "ominous", "fearful", "trepidation"],
        "euphoria": ["euphoria", "bliss", "ecstatic", "elated", "jubilant", "triumphant"],
        "contentment": ["content", "satisfied", "peaceful", "calm", "serene", "tranquil"],
        "cheerful": ["cheerful", "happy", "joyful", "merry", "upbeat", "bright"],
        "blissful": ["blissful", "rapturous", "transcendent", "divine", "heavenly"],
    }
    
    def __init__(self) -> None:
        """Initialize the intent processor."""
        self.thesaurus = EmotionThesaurus()
        self.wound_history: List[Wound] = []
        self.rule_breaks: List[RuleBreak] = []
    
    def process_wound(self, wound: Wound) -> EmotionNode:
        """
        Phase 1: Process a wound and map it to an emotion.
        
        Uses keyword matching and emotional proximity to find the best
        matching emotion. If no direct match is found, uses the wound's
        intensity and inferred valence to find nearby emotions.
        
        Args:
            wound: The wound to process
            
        Returns:
            The mapped emotion node (never None)
            
        Raises:
            ValueError: If thesaurus is empty or invalid
        """
        if not self.thesaurus.nodes:
            raise ValueError("Emotion thesaurus is empty")
        
        self.wound_history.append(wound)
        
        # Try keyword-based matching first
        emotion = self._match_by_keywords(wound)
        
        # If no keyword match, try intensity-based matching
        if not emotion:
            emotion = self._match_by_intensity(wound)
        
        # Fallback to default emotion
        if not emotion:
            emotion = self.thesaurus.find_emotion_by_name("melancholy")
        
        # Final fallback: first available emotion
        if not emotion:
            emotion = next(iter(self.thesaurus.nodes.values()))
        
        return emotion
    
    def _match_by_keywords(self, wound: Wound) -> Optional[EmotionNode]:
        """Match emotion by keywords in wound description."""
        description_lower = wound.description.lower()
        
        # Score each emotion based on keyword matches
        emotion_scores: Dict[str, float] = {}
        
        for emotion_name, keywords in self.EMOTION_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in description_lower:
                    # Exact match gets higher score
                    if keyword in wound.keywords:
                        score += 2.0
                    else:
                        score += 1.0
            
            if score > 0:
                emotion_scores[emotion_name] = score
        
        # Find emotion with highest score
        if emotion_scores:
            best_emotion_name = max(emotion_scores, key=emotion_scores.get)
            return self.thesaurus.find_emotion_by_name(best_emotion_name)
        
        return None
    
    def _match_by_intensity(self, wound: Wound) -> Optional[EmotionNode]:
        """Match emotion by intensity and inferred valence."""
        # Infer valence from keywords (negative words suggest negative valence)
        negative_words = ["loss", "grief", "sad", "angry", "fear", "anxiety", "dread"]
        positive_words = ["happy", "joy", "bliss", "content", "peaceful", "calm"]
        
        description_lower = wound.description.lower()
        has_negative = any(word in description_lower for word in negative_words)
        has_positive = any(word in description_lower for word in positive_words)
        
        # Infer valence
        if has_negative and not has_positive:
            target_valence = -0.7
        elif has_positive and not has_negative:
            target_valence = 0.7
        else:
            target_valence = 0.0  # Neutral
        
        # Find emotions matching intensity and valence
        best_match: Optional[Tuple[EmotionNode, float]] = None
        min_distance = float('inf')
        
        for node in self.thesaurus.nodes.values():
            # Calculate distance considering intensity and valence
            intensity_diff = abs(node.intensity - wound.intensity)
            valence_diff = abs(node.valence - target_valence)
            distance = intensity_diff * 0.7 + valence_diff * 0.3
            
            if distance < min_distance:
                min_distance = distance
                best_match = (node, distance)
        
        return best_match[0] if best_match and min_distance < 0.5 else None
    
    def emotion_to_rule_breaks(self, emotion: EmotionNode) -> List[RuleBreak]:
        """
        Phase 2-3: Convert emotion to musical rule-breaks.
        
        Generates rule breaks based on emotion characteristics:
        - High intensity → dynamic contrasts, irregular rhythms
        - Negative valence → dissonance, minor modes, descending melodies
        - High arousal → syncopation, tempo variations, rhythmic complexity
        
        Args:
            emotion: The emotion to express
            
        Returns:
            List of rule breaks for musical expression
        """
        rule_breaks: List[RuleBreak] = []
        
        # High intensity emotions break more rules
        if emotion.intensity > 0.8:
            rule_breaks.append(RuleBreak(
                rule_type="dynamics",
                severity=emotion.intensity,
                description="Extreme dynamic contrasts and sudden changes",
                justification=f"High intensity ({emotion.intensity:.2f}) requires dramatic expression",
                musical_impact={
                    "velocity_range": (int(10 + emotion.intensity * 20), 127),
                    "sudden_changes": True,
                    "crescendo_rate": emotion.intensity,
                    "diminuendo_rate": emotion.intensity * 0.8,
                }
            ))
        
        # Negative valence introduces dissonance and tension
        if emotion.valence < -0.5:
            dissonance_severity = abs(emotion.valence)
            rule_breaks.append(RuleBreak(
                rule_type="harmony",
                severity=dissonance_severity,
                description="Dissonant intervals, clusters, and unresolved tensions",
                justification=f"Negative valence ({emotion.valence:.2f}) requires harmonic tension",
                musical_impact={
                    "allow_dissonance": True,
                    "cluster_probability": dissonance_severity,
                    "avoid_resolution": True,
                    "prefer_minor_intervals": True,
                    "tritone_probability": dissonance_severity * 0.5,
                }
            ))
        
        # High arousal breaks rhythmic conventions
        if emotion.arousal > 0.7:
            rule_breaks.append(RuleBreak(
                rule_type="rhythm",
                severity=emotion.arousal,
                description="Irregular rhythms, syncopation, and metric displacement",
                justification=f"High arousal ({emotion.arousal:.2f}) requires rhythmic complexity",
                musical_impact={
                    "syncopation_level": emotion.arousal,
                    "irregular_meters": True,
                    "polyrhythm_probability": emotion.arousal * 0.6,
                    "tempo_variation": emotion.arousal * 0.3,
                }
            ))
        
        # Low arousal (calm emotions) can break rules through minimalism
        if emotion.arousal < 0.3 and emotion.intensity < 0.5:
            rule_breaks.append(RuleBreak(
                rule_type="arrangement",
                severity=1.0 - emotion.arousal,
                description="Sparse arrangement, extended silences, minimal texture",
                justification=f"Low arousal ({emotion.arousal:.2f}) benefits from space",
                musical_impact={
                    "sparse_arrangement": True,
                    "silence_probability": 0.3,
                    "minimal_voices": True,
                    "extended_durations": True,
                }
            ))
        
        # Extreme emotions (very high or very low) break melodic conventions
        if emotion.intensity > 0.9 or (emotion.valence < -0.8 and emotion.intensity > 0.7):
            rule_breaks.append(RuleBreak(
                rule_type="melody",
                severity=emotion.intensity,
                description="Unconventional melodic contours, wide leaps, chromaticism",
                justification=f"Extreme emotion requires unconventional melodic expression",
                musical_impact={
                    "allow_wide_leaps": True,
                    "chromatic_probability": emotion.intensity * 0.4,
                    "unconventional_contour": True,
                    "range_extension": emotion.intensity * 0.5,
                }
            ))
        
        self.rule_breaks.extend(rule_breaks)
        return rule_breaks
    
    def process_intent(self, wound: Wound) -> Dict[str, Any]:
        """
        Complete three-phase intent processing.
        
        Processes a wound through all three phases:
        1. Wound → Emotion mapping
        2. Emotion → Rule breaks generation
        3. Compilation of musical parameters
        
        Args:
            wound: The initial wound/trigger
            
        Returns:
            Dictionary containing:
            - wound: Original wound object
            - emotion: Mapped emotion node
            - rule_breaks: List of generated rule breaks
            - musical_params: Compiled musical parameters
            - processing_metadata: Information about the processing
            
        Raises:
            ValueError: If wound is invalid or processing fails
        """
        if not wound.description.strip():
            raise ValueError("Wound description cannot be empty")
        
        if not 0.0 <= wound.intensity <= 1.0:
            raise ValueError(f"Wound intensity must be between 0.0 and 1.0, got {wound.intensity}")
        
        # Phase 1: Wound → Emotion
        emotion = self.process_wound(wound)
        
        # Phase 2-3: Emotion → Rule-breaks
        rule_breaks = self.emotion_to_rule_breaks(emotion)
        
        # Compile musical parameters
        musical_params = self._compile_musical_params(emotion, rule_breaks)
        
        # Generate processing metadata
        metadata = {
            "processing_time": datetime.now().timestamp(),
            "emotion_match_method": "keyword" if self._match_by_keywords(wound) else "intensity",
            "rule_breaks_count": len(rule_breaks),
            "wound_keywords": wound.keywords,
        }
        
        return {
            "wound": wound,
            "emotion": emotion,
            "rule_breaks": rule_breaks,
            "musical_params": musical_params,
            "processing_metadata": metadata,
        }
    
    def _compile_musical_params(
        self, emotion: EmotionNode, rule_breaks: List[RuleBreak]
    ) -> Dict[str, Any]:
        """
        Compile final musical parameters from emotion and rule breaks.
        
        Merges emotion attributes with rule break impacts, with rule breaks
        taking precedence for conflicting parameters.
        
        Args:
            emotion: The mapped emotion
            rule_breaks: List of rule breaks to apply
            
        Returns:
            Dictionary of compiled musical parameters
        """
        # Start with emotion's musical attributes
        params = emotion.musical_attributes.copy()
        
        # Apply rule break impacts (rule breaks override emotion defaults)
        for rb in rule_breaks:
            params.update(rb.musical_impact)
        
        # Add summary flags
        params["has_dissonance"] = params.get("allow_dissonance", False)
        params["has_syncopation"] = params.get("syncopation_level", 0.0) > 0.0
        params["has_dynamic_contrast"] = params.get("sudden_changes", False)
        
        return params
    
    def get_wound_history(self) -> List[Wound]:
        """Get history of processed wounds."""
        return self.wound_history.copy()
    
    def get_rule_breaks_history(self) -> List[RuleBreak]:
        """Get history of all generated rule breaks."""
        return self.rule_breaks.copy()
    
    def clear_history(self) -> None:
        """Clear wound and rule break history."""
        self.wound_history.clear()
        self.rule_breaks.clear()
