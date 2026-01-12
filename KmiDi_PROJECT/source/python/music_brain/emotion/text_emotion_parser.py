"""
Text Emotion Parser for Natural Language -> EmotionalState conversion.

Parses natural language descriptions into structured emotion representations
aligned with the 6x6x6 emotion thesaurus and EmotionalState dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Import from sibling modules
try:
    from .emotion_thesaurus import EmotionThesaurus, EmotionMatch, BlendMatch
except ImportError:
    from emotion_thesaurus import EmotionThesaurus, EmotionMatch, BlendMatch


@dataclass
class ParsedEmotion:
    """Result of parsing natural language into emotion components."""
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    base_emotion: str  # HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST
    sub_emotion: Optional[str] = None
    sub_sub_emotion: Optional[str] = None
    intensity_tier: int = 3  # 1-6
    blend_components: List[str] = field(default_factory=list)
    blend_ratio: List[float] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    confidence: float = 0.5
    matched_words: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_emotional_state_dict(self) -> Dict:
        """Convert to dict compatible with EmotionalState dataclass."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "primary_emotion": self.base_emotion.lower() if self.base_emotion else "neutral",
            "secondary_emotions": [c.lower() for c in self.blend_components] if self.blend_components else [],
            "has_intrusions": "ptsd_intrusion" in self.modifiers,
        }


class TextEmotionParser:
    """
    Parse natural language text into structured emotion representations.

    Uses the 6x6x6 emotion thesaurus for synonym matching and supports
    intensity modifiers, emotional blends, and therapeutic modifiers.
    """

    # Base emotion to valence/arousal mapping
    BASE_EMOTION_VA = {
        "HAPPY": (0.8, 0.6),
        "SAD": (-0.6, 0.3),
        "ANGRY": (-0.5, 0.85),
        "FEAR": (-0.7, 0.75),
        "SURPRISE": (0.1, 0.8),
        "DISGUST": (-0.6, 0.5),
    }

    # Intensity tier to arousal modifier
    INTENSITY_AROUSAL_MOD = {
        1: -0.2,  # subtle
        2: -0.1,  # mild
        3: 0.0,   # moderate
        4: 0.1,   # strong
        5: 0.2,   # intense
        6: 0.3,   # overwhelming
    }

    # Therapeutic modifiers from emotional_mapping.py
    MODIFIER_PATTERNS = {
        "ptsd_intrusion": [
            r"\bflashback\b", r"\btrigger", r"\bintrusi", r"\btrauma",
            r"\breliving\b", r"\bhaunted\b", r"\bvivid\s+memor",
        ],
        "dissociation": [
            r"\bdisconnect", r"\bnumb\b", r"\bdetach", r"\bunreal",
            r"\bfloating\b", r"\boutside\s+(my|the)\s+body", r"\bfoggy\b",
        ],
        "misdirection": [
            r"\bhiding\b", r"\bmask", r"\bpretend", r"\bfake\b",
            r"\bcover\s+up", r"\bdeflect",
        ],
        "suppressed": [
            r"\bsuppress", r"\bbottl", r"\bhold\s+(it\s+)?in\b", r"\bbury\b",
            r"\bswallow", r"\bstuff\s+(down|it)\b",
        ],
        "cathartic_release": [
            r"\breleas", r"\blet\s+(it\s+)?go\b", r"\bbreakthrough\b",
            r"\bcathar", r"\bpurg", r"\bvent\b",
        ],
    }

    # Intensity amplifiers and diminishers
    AMPLIFIERS = [
        "very", "extremely", "incredibly", "deeply", "intensely",
        "overwhelmingly", "profoundly", "desperately", "absolutely",
        "completely", "utterly", "totally",
    ]

    DIMINISHERS = [
        "slightly", "somewhat", "a bit", "a little", "mildly",
        "kind of", "sort of", "fairly", "rather",
    ]

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the parser.

        Args:
            data_dir: Path to emotion_thesaurus data directory.
        """
        self.thesaurus = EmotionThesaurus(data_dir)
        self._build_synonym_cache()

    def _build_synonym_cache(self) -> None:
        """Build a fast lookup cache from the thesaurus."""
        self._synonym_to_emotion: Dict[str, Tuple[str, str, str, int]] = {}

        # Index all synonyms from thesaurus
        for base in self.thesaurus.BASE_EMOTIONS:
            base_upper = base.upper()

            # Index base emotion name itself
            self._synonym_to_emotion[base.lower()] = (base_upper, "", "", 3)

            for sub in self.thesaurus.get_sub_emotions(base_upper):
                # Index sub-emotion name
                sub_lower = sub.lower()
                if sub_lower not in self._synonym_to_emotion:
                    self._synonym_to_emotion[sub_lower] = (base_upper, sub, "", 3)

                for subsub in self.thesaurus.get_sub_sub_emotions(base_upper, sub):
                    # Index sub-sub-emotion name
                    subsub_lower = subsub.lower()
                    if subsub_lower not in self._synonym_to_emotion:
                        self._synonym_to_emotion[subsub_lower] = (base_upper, sub, subsub, 3)

                    for tier in range(1, 7):
                        synonyms = self.thesaurus.get_intensity_synonyms(
                            base_upper, sub, subsub, tier
                        )
                        for syn in synonyms:
                            key = syn.lower().strip()
                            # Keep highest intensity match
                            if key not in self._synonym_to_emotion or tier > self._synonym_to_emotion[key][3]:
                                self._synonym_to_emotion[key] = (base_upper, sub, subsub, tier)

        # Add common emotion words that may not be in thesaurus
        self._add_common_emotion_words()

        self._total_synonyms = len(self._synonym_to_emotion)

    def _add_common_emotion_words(self) -> None:
        """Add commonly used emotion words that might be missing from thesaurus."""
        common_mappings = {
            # Happy family
            "joy": ("HAPPY", "JOY", "joyful", 4),
            "joyous": ("HAPPY", "JOY", "joyful", 4),
            "elated": ("HAPPY", "JOY", "elated", 5),
            "bliss": ("HAPPY", "JOY", "blissful", 5),
            "blissful": ("HAPPY", "JOY", "blissful", 5),
            "excitement": ("HAPPY", "EXCITEMENT", "excited", 4),
            "thrilled": ("HAPPY", "EXCITEMENT", "thrilled", 5),
            "ecstatic": ("HAPPY", "JOY", "ecstatic", 6),

            # Sad family
            "melancholy": ("SAD", "GRIEF", "melancholic", 3),
            "melancholic": ("SAD", "GRIEF", "melancholic", 3),
            "sorrow": ("SAD", "GRIEF", "sorrowful", 4),
            "sorrowful": ("SAD", "GRIEF", "sorrowful", 4),
            "grief": ("SAD", "GRIEF", "grieving", 5),
            "grieving": ("SAD", "GRIEF", "grieving", 5),
            "nostalgia": ("SAD", "GRIEF", "nostalgic", 3),
            "longing": ("SAD", "GRIEF", "longing", 3),
            "wistful": ("SAD", "GRIEF", "wistful", 2),
            "bittersweet": ("SAD", "GRIEF", "bittersweet", 3),

            # Angry family
            "rage": ("ANGRY", "RAGE", "enraged", 6),
            "fury": ("ANGRY", "RAGE", "furious", 5),
            "furious": ("ANGRY", "RAGE", "furious", 5),
            "irritated": ("ANGRY", "ANNOYANCE", "irritated", 2),
            "annoyed": ("ANGRY", "ANNOYANCE", "annoyed", 2),
            "frustrated": ("ANGRY", "FRUSTRATION", "frustrated", 3),
            "resentful": ("ANGRY", "RESENTMENT", "resentful", 4),

            # Fear family
            "terror": ("FEAR", "TERROR", "terrified", 6),
            "terrified": ("FEAR", "TERROR", "terrified", 6),
            "dread": ("FEAR", "DREAD", "dreading", 5),
            "panic": ("FEAR", "PANIC", "panicked", 6),
            "nervous": ("FEAR", "ANXIETY", "nervous", 2),
            "worried": ("FEAR", "ANXIETY", "worried", 3),
            "uneasy": ("FEAR", "ANXIETY", "uneasy", 2),

            # Surprise family
            "amazed": ("SURPRISE", "AMAZEMENT", "amazed", 4),
            "astonished": ("SURPRISE", "AMAZEMENT", "astonished", 5),
            "shocked": ("SURPRISE", "SHOCK", "shocked", 5),
            "stunned": ("SURPRISE", "SHOCK", "stunned", 5),

            # Disgust family
            "revolted": ("DISGUST", "REVULSION", "revolted", 5),
            "repulsed": ("DISGUST", "REVULSION", "repulsed", 4),
            "contempt": ("DISGUST", "CONTEMPT", "contemptuous", 4),
        }

        # Override mappings - these take priority over thesaurus
        override_mappings = {
            "anxious": ("FEAR", "ANXIETY", "anxious", 3),
            "worried": ("FEAR", "ANXIETY", "worried", 3),
            "nervous": ("FEAR", "ANXIETY", "nervous", 2),
            "joyful": ("HAPPY", "JOY", "joyful", 4),
            "excited": ("HAPPY", "EXCITEMENT", "excited", 4),
            "happy": ("HAPPY", "JOY", "happy", 3),
            "sad": ("SAD", "GRIEF", "sad", 3),
            "angry": ("ANGRY", "RAGE", "angry", 3),
            "scared": ("FEAR", "TERROR", "scared", 4),
            "afraid": ("FEAR", "TERROR", "afraid", 3),
        }

        # Force override critical words
        for word, mapping in override_mappings.items():
            self._synonym_to_emotion[word] = mapping

        # Add common words only if not present
        for word, mapping in common_mappings.items():
            if word not in self._synonym_to_emotion:
                self._synonym_to_emotion[word] = mapping

    def parse(self, text: str) -> ParsedEmotion:
        """
        Parse natural language text into a ParsedEmotion.

        Args:
            text: Natural language emotion description.

        Returns:
            ParsedEmotion with extracted emotion components.
        """
        text_lower = text.lower().strip()
        words = re.findall(r'\b[\w-]+\b', text_lower)

        # Detect modifiers
        modifiers = self._detect_modifiers(text_lower)

        # Detect intensity adjustment
        intensity_adj = self._detect_intensity_adjustment(text_lower)

        # Find emotion matches
        matches = []
        matched_words = []

        # Try multi-word phrases first (up to 3 words)
        for n in [3, 2, 1]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if phrase in self._synonym_to_emotion:
                    matches.append(self._synonym_to_emotion[phrase])
                    matched_words.append(phrase)

        # Also check thesaurus for blends
        blend_matches = []
        for word in words:
            blend_results = self.thesaurus.find_blend(word)
            if blend_results:
                blend_matches.extend(blend_results)
                matched_words.append(word)

        # Build result
        if matches:
            # Use the highest-intensity match as primary
            matches.sort(key=lambda x: x[3], reverse=True)
            primary = matches[0]
            base_emotion = primary[0]
            sub_emotion = primary[1]
            sub_sub_emotion = primary[2]
            base_intensity = primary[3]

            # Adjust intensity based on amplifiers/diminishers
            intensity_tier = max(1, min(6, base_intensity + intensity_adj))

            # Get valence/arousal from base emotion
            base_va = self.BASE_EMOTION_VA.get(base_emotion, (0.0, 0.5))
            arousal_mod = self.INTENSITY_AROUSAL_MOD.get(intensity_tier, 0.0)

            valence = base_va[0]
            arousal = max(0.0, min(1.0, base_va[1] + arousal_mod))

            # Apply modifier effects
            if "dissociation" in modifiers:
                arousal *= 0.5  # Flatten arousal
            if "ptsd_intrusion" in modifiers:
                arousal = min(1.0, arousal + 0.2)  # Spike arousal

            confidence = min(1.0, 0.5 + 0.1 * len(matches))

            # Handle blends
            blend_components = []
            blend_ratio = []
            if blend_matches:
                best_blend = blend_matches[0]
                blend_components = best_blend.components
                blend_ratio = best_blend.ratio if best_blend.ratio else [1.0 / len(blend_components)] * len(blend_components)

            return ParsedEmotion(
                valence=valence,
                arousal=arousal,
                base_emotion=base_emotion,
                sub_emotion=sub_emotion,
                sub_sub_emotion=sub_sub_emotion,
                intensity_tier=intensity_tier,
                blend_components=blend_components,
                blend_ratio=blend_ratio,
                modifiers=modifiers,
                confidence=confidence,
                matched_words=list(set(matched_words)),
                raw_text=text,
            )

        elif blend_matches:
            # Only blend match, no direct emotion
            best_blend = blend_matches[0]
            blend_components = best_blend.components
            blend_ratio = best_blend.ratio if best_blend.ratio else [1.0 / len(blend_components)] * len(blend_components)

            # Infer valence/arousal from blend components
            valence = 0.0
            arousal = 0.5
            for comp, ratio in zip(blend_components, blend_ratio):
                comp_upper = comp.upper()
                if comp_upper in self.BASE_EMOTION_VA:
                    v, a = self.BASE_EMOTION_VA[comp_upper]
                    valence += v * ratio
                    arousal += (a - 0.5) * ratio

            arousal = max(0.0, min(1.0, arousal))

            return ParsedEmotion(
                valence=valence,
                arousal=arousal,
                base_emotion=blend_components[0].upper() if blend_components else "NEUTRAL",
                intensity_tier=best_blend.intensity_tier,
                blend_components=blend_components,
                blend_ratio=blend_ratio,
                modifiers=modifiers,
                confidence=0.6,
                matched_words=list(set(matched_words)),
                raw_text=text,
            )

        else:
            # No match - return neutral
            return ParsedEmotion(
                valence=0.0,
                arousal=0.5,
                base_emotion="NEUTRAL",
                intensity_tier=3,
                modifiers=modifiers,
                confidence=0.2,
                matched_words=[],
                raw_text=text,
            )

    def _detect_modifiers(self, text: str) -> List[str]:
        """Detect therapeutic modifiers in text."""
        found = []
        for modifier, patterns in self.MODIFIER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found.append(modifier)
                    break
        return found

    def _detect_intensity_adjustment(self, text: str) -> int:
        """Detect intensity amplifiers/diminishers."""
        adjustment = 0

        for amp in self.AMPLIFIERS:
            if amp in text:
                adjustment += 1
                break

        for dim in self.DIMINISHERS:
            if dim in text:
                adjustment -= 1
                break

        return adjustment

    def parse_batch(self, texts: List[str]) -> List[ParsedEmotion]:
        """Parse multiple texts."""
        return [self.parse(t) for t in texts]

    def stats(self) -> Dict:
        """Return parser statistics."""
        return {
            "total_synonyms_indexed": self._total_synonyms,
            "base_emotions": list(self.BASE_EMOTION_VA.keys()),
            "modifiers_tracked": list(self.MODIFIER_PATTERNS.keys()),
            "amplifiers": self.AMPLIFIERS,
            "diminishers": self.DIMINISHERS,
        }


if __name__ == "__main__":
    # Demo
    print("Text Emotion Parser Demo")
    print("=" * 50)

    try:
        parser = TextEmotionParser()
        print(f"Indexed {parser._total_synonyms} synonyms")

        test_phrases = [
            "I feel deeply melancholic today",
            "overwhelming joy and excitement",
            "slightly anxious about the future",
            "experiencing flashbacks and feeling numb",
            "bittersweet nostalgia",
            "suppressed rage",
            "peaceful and content",
        ]

        print("\nParsing examples:")
        print("-" * 50)
        for phrase in test_phrases:
            result = parser.parse(phrase)
            print(f"\n'{phrase}'")
            print(f"  Base: {result.base_emotion} (tier {result.intensity_tier})")
            print(f"  V/A: {result.valence:.2f} / {result.arousal:.2f}")
            if result.modifiers:
                print(f"  Modifiers: {result.modifiers}")
            if result.blend_components:
                print(f"  Blend: {result.blend_components}")
            print(f"  Matched: {result.matched_words}")
            print(f"  Confidence: {result.confidence:.2f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
