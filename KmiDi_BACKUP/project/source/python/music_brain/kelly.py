"""
Kelly - The Caregiver and Listener.

Translates emotional language into musical inspiration and gentle guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional

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
        """Listens to the user's emotional input and provides validation + direction."""
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
        """Detect an emotion from free text using the thesaurus then fallbacks."""
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
            return self._make_match(
                base="sad",
                sub="grief",
                sub_sub="grief",
                tier=5,
                synonym="grief",
                desc="Deep sorrow",
            )
        if any(w in text_lower for w in ["happy", "joy", "excited", "great"]):
            return self._make_match(
                base="happy",
                sub="joy",
                sub_sub="excitement",
                tier=5,
                synonym="excited",
                desc="High energy happiness",
            )
        if any(w in text_lower for w in ["angry", "mad", "furious", "hate"]):
            return self._make_match(
                base="angry",
                sub="anger",
                sub_sub="rage",
                tier=6,
                synonym="furious",
                desc="Intense anger",
            )
        if any(w in text_lower for w in ["fear", "scared", "anxious", "nervous"]):
            return self._make_match(
                base="fear",
                sub="fear",
                sub_sub="anxiety",
                tier=4,
                synonym="anxious",
                desc="Unease and worry",
            )

        # Default neutral
        return self._make_match(
            base="neutral",
            sub="calm",
            sub_sub="neutral",
            tier=1,
            synonym="neutral",
            desc="No strong emotion detected",
        )

    def _generate_guidance(
        self,
        emotion: EmotionMatch,
        preset: ProductionPreset,
    ) -> str:
        """Generate a gentle, guiding message for musical expression."""
        return (
            f"To capture this {emotion.base_emotion.lower()}, we might start "
            f"with a {preset.drum_style} rhythm. Keep dynamics around "
            f"{preset.dynamics_level} and arrangement density near "
            f"{int(preset.arrangement_density * 100)}%. "
            "Does that resonate with what you're feeling?"
        )

    @staticmethod
    def _make_match(
        base: str,
        sub: str,
        sub_sub: str,
        tier: int,
        synonym: str,
        desc: str,
    ) -> EmotionMatch:
        return EmotionMatch(
            base_emotion=base,
            sub_emotion=sub,
            sub_sub_emotion=sub_sub,
            intensity_tier=tier,
            matched_synonym=synonym,
            all_tier_synonyms=[synonym, sub],
            emotion_id=f"{base.upper()}.{sub.upper()}",
            description=desc,
        )
