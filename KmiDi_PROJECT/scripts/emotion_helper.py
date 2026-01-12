#!/usr/bin/env python3
"""
Emotion helper for v2 training pipeline - music_brain integration.

Converts music_brain EmotionThesaurus matches to 3D emotion vectors
(valence, arousal, intensity) for training. This replaces placeholder
emotion arrays with production-ready emotion labeling based on the
existing music_brain emotion system.

Architecture:
    music_brain.EmotionThesaurus (6×6×6 taxonomy, intensity tiers)
        ↓
    emotion_helper (converts to VAI vectors)
        ↓
    Manifest builder (stores in JSONL)
        ↓
    Dataset loader (normalizes 3D → 64D)
        ↓
    Training (emotion-conditioned generation)

This enables:
1. Consistent emotion representation across the codebase
2. Path-based weak supervision for quick setup
3. Support for ground-truth human annotations
4. Future: Audio-based emotion extraction
"""

from typing import List, Tuple, Optional
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def emotion_match_to_vector(
    base_emotion: str,
    sub_emotion: str = "",
    intensity_tier: int = 3
) -> List[float]:
    """
    Convert emotion thesaurus match to [valence, arousal, intensity] vector.
    
    Based on Russell's Circumplex Model and music_brain's emotional_mapping.
    
    Args:
        base_emotion: One of happy, sad, angry, fear, surprise, disgust
        sub_emotion: Sub-category (optional)
        intensity_tier: 1-6 intensity tier from thesaurus
        
    Returns:
        [valence, arousal, intensity] where each is in range [-1, 1] or [0, 1]
    """
    # Normalize intensity from tier (1-6) to [0, 1]
    intensity = (intensity_tier - 1) / 5.0  # Maps 1→0.0, 6→1.0
    
    # Base emotion mappings using Russell's Circumplex Model
    # (valence, arousal) - then we'll scale by intensity
    emotion_map = {
        "happy": (0.8, 0.6),      # High valence, moderate-high arousal
        "joy": (0.9, 0.7),         # Very high valence, high arousal
        "sad": (-0.7, -0.4),       # Low valence, low arousal
        "grief": (-0.8, -0.5),     # Very low valence, low arousal
        "angry": (-0.7, 0.8),      # Low valence, high arousal
        "rage": (-0.9, 0.95),      # Very low valence, very high arousal
        "fear": (-0.6, 0.7),       # Low valence, high arousal
        "anxiety": (-0.5, 0.6),    # Moderate-low valence, moderate-high arousal
        "surprise": (0.2, 0.8),    # Slightly positive, high arousal
        "disgust": (-0.6, 0.3),    # Low valence, moderate arousal
        "calm": (0.4, -0.7),       # Moderate valence, low arousal
        "peaceful": (0.5, -0.8),   # Moderate-high valence, very low arousal
        "excited": (0.7, 0.9),     # High valence, very high arousal
        "content": (0.6, -0.3),    # Moderate-high valence, low arousal
    }
    
    # Try to find mapping for base emotion
    base_key = base_emotion.lower()
    if base_key in emotion_map:
        valence, arousal = emotion_map[base_key]
    else:
        # Default neutral
        valence, arousal = (0.0, 0.0)
    
    # Sub-emotion adjustments (examples - can be expanded)
    sub_adjustments = {
        "grief": {"valence": -0.1, "arousal": -0.1},
        "melancholy": {"valence": -0.05, "arousal": -0.15},
        "despair": {"valence": -0.15, "arousal": -0.2},
        "rage": {"arousal": 0.15},
        "fury": {"arousal": 0.2},
        "terror": {"valence": -0.1, "arousal": 0.15},
        "panic": {"arousal": 0.2},
        "bliss": {"valence": 0.1, "arousal": 0.1},
        "ecstasy": {"valence": 0.15, "arousal": 0.2},
    }
    
    if sub_emotion and sub_emotion.lower() in sub_adjustments:
        adj = sub_adjustments[sub_emotion.lower()]
        valence += adj.get("valence", 0.0)
        arousal += adj.get("arousal", 0.0)
    
    # Clamp to valid ranges
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))
    
    # Intensity is already normalized to [0, 1]
    return [valence, arousal, intensity]


def extract_emotion_from_filename(filepath: Path) -> Optional[Tuple[str, str, int]]:
    """
    Extract emotion from file path or filename.
    
    Returns:
        (base_emotion, sub_emotion, intensity_tier) or None if not found
    """
    # Check parent directory name
    parent_name = filepath.parent.name.lower()
    
    # Check filename
    filename = filepath.stem.lower()
    
    # Emotion patterns to look for
    emotions = {
        "happy": ("happy", "", 3),
        "joy": ("happy", "joy", 4),
        "joyful": ("happy", "joy", 4),
        "sad": ("sad", "", 3),
        "sadness": ("sad", "", 3),
        "melancholy": ("sad", "melancholy", 3),
        "grief": ("sad", "grief", 5),
        "bereaved": ("sad", "grief", 6),
        "angry": ("angry", "", 3),
        "rage": ("angry", "rage", 6),
        "fury": ("angry", "fury", 5),
        "calm": ("happy", "calm", 2),
        "peaceful": ("happy", "peaceful", 2),
        "excited": ("happy", "excited", 4),
        "fear": ("fear", "", 3),
        "fearful": ("fear", "", 3),
        "terror": ("fear", "terror", 6),
        "anxiety": ("fear", "anxiety", 4),
        "anxious": ("fear", "anxiety", 4),
        "panic": ("fear", "panic", 6),
        "surprise": ("surprise", "", 3),
        "surprised": ("surprise", "", 3),
        "disgust": ("disgust", "", 3),
        "disgusted": ("disgust", "", 3),
        "neutral": ("happy", "", 1),
        "tender": ("happy", "tender", 3),
        "energetic": ("happy", "energetic", 4),
    }
    
    # Check directory name first
    if parent_name in emotions:
        return emotions[parent_name]
    
    # Check filename for emotion keywords
    for emotion_key, emotion_data in emotions.items():
        if emotion_key in filename:
            return emotion_data
    
    # No explicit emotion hint found
    return None


def get_emotion_vector_from_path(filepath: Path) -> List[float]:
    """
    Get emotion vector from file path.
    
    Convenience function that combines extraction and conversion.
    
    Args:
        filepath: Path to audio or MIDI file
        
    Returns:
        [valence, arousal, intensity] emotion vector
    """
    emotion_data = extract_emotion_from_filename(filepath)
    if emotion_data:
        base, sub, tier = emotion_data
        return emotion_match_to_vector(base, sub, tier)
    else:
        # Default neutral
        return [0.0, 0.0, 0.5]


def use_emotion_thesaurus(word: str) -> Optional[List[float]]:
    """
    Use music_brain's EmotionThesaurus to get emotion vector from a word.
    
    Args:
        word: Emotion word/phrase to look up
        
    Returns:
        [valence, arousal, intensity] or None if not found
    """
    try:
        from music_brain.emotion_thesaurus import EmotionThesaurus
        
        thesaurus = EmotionThesaurus()
        matches = thesaurus.find_by_synonym(word)
        
        if matches:
            # Use first match
            match = matches[0]
            return emotion_match_to_vector(
                match.base_emotion,
                match.sub_emotion,
                match.intensity_tier
            )
    except Exception as e:
        print(f"Warning: Could not use EmotionThesaurus: {e}")
    
    return None


if __name__ == "__main__":
    # Test the emotion mappings
    test_emotions = [
        ("happy", "", 3),
        ("sad", "grief", 5),
        ("angry", "rage", 6),
        ("fear", "anxiety", 4),
        ("calm", "", 2),
    ]
    
    print("Emotion to Vector Mapping:")
    print("-" * 60)
    for base, sub, tier in test_emotions:
        vec = emotion_match_to_vector(base, sub, tier)
        sub_str = f" > {sub}" if sub else ""
        print(f"{base}{sub_str} (tier {tier}): {vec}")
    
    # Test thesaurus lookup
    print("\nThesaurus Lookup Examples:")
    print("-" * 60)
    test_words = ["grief", "melancholy", "rage", "peaceful", "ecstatic"]
    for word in test_words:
        vec = use_emotion_thesaurus(word)
        print(f"{word}: {vec}")
