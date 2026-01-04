"""
Kelly - The Caregiver and Listener.

Kelly provides inspiration through validation of feelings. She helps the user
overcome writer's block by translating emotions into musical concepts.
She guides the "Emotion Optional Creation Workflow".
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import random

from music_brain.emotion.emotion_production import (
    EmotionProductionMapper,
    ProductionPreset,
)
from music_brain.emotion.emotion_thesaurus import EmotionMatch, EmotionThesaurus


@dataclass
class KellyResponse:
    """Kelly's response to the user."""
    validation_message: str
    musical_inspiration: ProductionPreset
    guidance: str


class Kelly:
    """
    The empathetic guide for emotional expression in music.
    """

    def __init__(self):
        self.mapper = EmotionProductionMapper()
        self.thesaurus = EmotionThesaurus()

        self.validations = [
            "I hear you. That sounds incredibly intense.",
            "It's completely valid to feel that way.",
            "Thank you for sharing that with me. Let's channel it.",
            "I can feel the weight of that emotion.",
            "That's a powerful place to start a song from."
        ]

    def listen(self, user_input: str) -> KellyResponse:
        """
        Listens to the user's emotional input and provides validation and musical direction.
        """
        # 1. Identify Emotion (Simple keyword matching for now, could be advanced NLP)
        # Using the thesaurus to find the best match
        emotion_match = self._detect_emotion(user_input)

        # 2. Validate
        validation = f"{random.choice(self.validations)} You're feeling {emotion_match.matched_synonym}."

        # 3. Translate to Music
        preset = self.mapper.get_production_preset(emotion_match)

        # 4. Provide Guidance
        guidance = self._generate_guidance(emotion_match, preset)

        return KellyResponse(
            validation_message=validation,
            musical_inspiration=preset,
            guidance=guidance
        )

    def _detect_emotion(self, text: str) -> EmotionMatch:
        """
        Detects emotion from text.
        """
        text_lower = text.lower()

        # 1. Try to find a match using the thesaurus
        words = text_lower.split()
        for word in words:
            matches = self.thesaurus.find_by_synonym(word)
            if matches:
                # Return the first match found
                return matches[0]

        # 2. Fallback to simple keyword matching with manual construction
        if any(w in text_lower for w in ["sad", "grief", "loss", "cry", "blue"]):
            return EmotionMatch(
                base_emotion="SAD",
                sub_emotion="Sadness",
                sub_sub_emotion="Grief",
                intensity_tier=5,
                matched_synonym="grief",
                all_tier_synonyms=["grief", "heartbreak"],
                emotion_id="SAD.GRIEF",
                description="Deep sorrow"
            )
        elif any(w in text_lower for w in ["happy", "joy", "excited", "great"]):
            return EmotionMatch(
                base_emotion="HAPPY",
                sub_emotion="Joy",
                sub_sub_emotion="Excitement",
                intensity_tier=5,
                matched_synonym="excited",
                all_tier_synonyms=["excited", "elated"],
                emotion_id="HAPPY.EXCITE",
                description="High energy happiness"
            )
        elif any(w in text_lower for w in ["angry", "mad", "furious", "hate"]):
            return EmotionMatch(
                base_emotion="ANGRY",
                sub_emotion="Anger",
                sub_sub_emotion="Rage",
                intensity_tier=6,
                matched_synonym="furious",
                all_tier_synonyms=["furious", "enraged"],
                emotion_id="ANGRY.RAGE",
                description="Intense anger"
            )
        elif any(w in text_lower for w in ["fear", "scared", "anxious", "nervous"]):
            return EmotionMatch(
                base_emotion="FEAR",
                sub_emotion="Fear",
                sub_sub_emotion="Anxiety",
                intensity_tier=4,
                matched_synonym="anxious",
                all_tier_synonyms=["anxious", "worried"],
                emotion_id="FEAR.ANXIETY",
                description="Unease and worry"
            )

        # Default
        return EmotionMatch(
            base_emotion="NEUTRAL",
            sub_emotion="Calm",
            sub_sub_emotion="Neutral",
            intensity_tier=1,
            matched_synonym="neutral",
            all_tier_synonyms=["neutral", "calm"],
            emotion_id="NEUTRAL",
            description="No strong emotion detected"
        )

    def _generate_guidance(self, emotion: EmotionMatch, preset: ProductionPreset) -> str:
        """
        Generates a gentle, guiding message about how to express this emotion musically.
        """
        return (
            f"To capture this {emotion.base_emotion.lower()}, we might want to start with a {preset.drum_style} rhythm. "
            f"Let's keep the dynamics around {preset.dynamics_level}. "
            f"I suggest an arrangement density of {int(preset.arrangement_density * 100)}%. "
            "Does that resonate with what you're feeling?"
        )
