"""
Phoneme conversion utilities for voice synthesis.

Provides functions for converting text to phonemes and phoneme classification.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class PhonemeType(Enum):
    """Phoneme classification types."""
    VOWEL = "vowel"
    CONSONANT = "consonant"
    SILENCE = "silence"


@dataclass
class Phoneme:
    """Represents a single phoneme."""
    phoneme_type: PhonemeType
    symbol: str  # IPA or phoneme symbol
    duration: float  # Duration in seconds
    pitch: Optional[float] = None  # Optional pitch in Hz


def text_to_phonemes(text: str) -> List[Phoneme]:
    """
    Convert text to phonemes.
    
    Args:
        text: Input text
        
    Returns:
        List of Phoneme objects
    """
    # Simple implementation - split by words and create basic phonemes
    words = text.split()
    phonemes = []
    
    for word in words:
        # Create a simple vowel phoneme for each word
        # In a full implementation, this would use a proper phoneme converter
        phonemes.append(Phoneme(
            phoneme_type=PhonemeType.VOWEL,
            symbol="ah",  # Default vowel
            duration=0.3,
        ))
    
    return phonemes


def phoneme_to_vowel_type(phoneme: Phoneme) -> Optional[str]:
    """
    Convert phoneme to vowel type string.
    
    Args:
        phoneme: Phoneme object
        
    Returns:
        Vowel type string (A, E, I, O, U) or None
    """
    if phoneme.phoneme_type != PhonemeType.VOWEL:
        return None
    
    # Simple mapping - in full implementation would use proper vowel classification
    symbol_lower = phoneme.symbol.lower()
    
    if 'a' in symbol_lower or 'ah' in symbol_lower:
        return 'A'
    elif 'e' in symbol_lower or 'eh' in symbol_lower:
        return 'E'
    elif 'i' in symbol_lower or 'ee' in symbol_lower:
        return 'I'
    elif 'o' in symbol_lower or 'oh' in symbol_lower:
        return 'O'
    elif 'u' in symbol_lower or 'oo' in symbol_lower:
        return 'U'
    
    return 'A'  # Default
