"""
Phoneme Conversion - Text to Phonemes for Parrot Synthesis

Converts text to phonemes with stress markers and timing information.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re


class PhonemeType(Enum):
    """Phoneme classification"""
    VOWEL = "vowel"
    CONSONANT = "consonant"
    SILENCE = "silence"


@dataclass
class Phoneme:
    """Represents a single phoneme"""
    symbol: str  # IPA symbol
    phoneme_type: PhonemeType
    duration: float  # Duration in seconds
    stress: int = 0  # Stress level (0=unstressed, 1=primary, 2=secondary)
    pitch: Optional[float] = None  # Target pitch (Hz)


# CMU Pronouncing Dictionary-style mappings (simplified)
PHONEME_MAP = {
    # Vowels
    'AA': ('ɑ', PhonemeType.VOWEL),  # "father"
    'AE': ('æ', PhonemeType.VOWEL),  # "cat"
    'AH': ('ə', PhonemeType.VOWEL),  # "about"
    'AO': ('ɔ', PhonemeType.VOWEL),  # "law"
    'AW': ('aʊ', PhonemeType.VOWEL),  # "cow"
    'AY': ('aɪ', PhonemeType.VOWEL),  # "hide"
    'EH': ('ɛ', PhonemeType.VOWEL),  # "red"
    'ER': ('ɜ', PhonemeType.VOWEL),  # "her"
    'EY': ('eɪ', PhonemeType.VOWEL),  # "ate"
    'IH': ('ɪ', PhonemeType.VOWEL),  # "it"
    'IY': ('i', PhonemeType.VOWEL),  # "eat"
    'OW': ('oʊ', PhonemeType.VOWEL),  # "show"
    'OY': ('ɔɪ', PhonemeType.VOWEL),  # "toy"
    'UH': ('ʊ', PhonemeType.VOWEL),  # "book"
    'UW': ('u', PhonemeType.VOWEL),  # "blue"
    
    # Consonants
    'B': ('b', PhonemeType.CONSONANT),
    'CH': ('tʃ', PhonemeType.CONSONANT),
    'D': ('d', PhonemeType.CONSONANT),
    'DH': ('ð', PhonemeType.CONSONANT),
    'F': ('f', PhonemeType.CONSONANT),
    'G': ('g', PhonemeType.CONSONANT),
    'HH': ('h', PhonemeType.CONSONANT),
    'JH': ('dʒ', PhonemeType.CONSONANT),
    'K': ('k', PhonemeType.CONSONANT),
    'L': ('l', PhonemeType.CONSONANT),
    'M': ('m', PhonemeType.CONSONANT),
    'N': ('n', PhonemeType.CONSONANT),
    'NG': ('ŋ', PhonemeType.CONSONANT),
    'P': ('p', PhonemeType.CONSONANT),
    'R': ('r', PhonemeType.CONSONANT),
    'S': ('s', PhonemeType.CONSONANT),
    'SH': ('ʃ', PhonemeType.CONSONANT),
    'T': ('t', PhonemeType.CONSONANT),
    'TH': ('θ', PhonemeType.CONSONANT),
    'V': ('v', PhonemeType.CONSONANT),
    'W': ('w', PhonemeType.CONSONANT),
    'Y': ('j', PhonemeType.CONSONANT),
    'Z': ('z', PhonemeType.CONSONANT),
    'ZH': ('ʒ', PhonemeType.CONSONANT),
}

# Simple word-to-phoneme mapping (would use proper dictionary in production)
SIMPLE_PHONEME_DICT = {
    'hello': ['HH', 'EH', 'L', 'OW'],
    'world': ['W', 'ER', 'L', 'D'],
    'the': ['DH', 'AH'],
    'a': ['AH'],
    'an': ['AE', 'N'],
    'and': ['AE', 'N', 'D'],
    'to': ['T', 'UW'],
    'of': ['AH', 'V'],
    'in': ['IH', 'N'],
    'is': ['IH', 'Z'],
    'it': ['IH', 'T'],
    'that': ['DH', 'AE', 'T'],
    'this': ['DH', 'IH', 'S'],
    'with': ['W', 'IH', 'DH'],
    'for': ['F', 'AO', 'R'],
    'on': ['AO', 'N'],
    'at': ['AE', 'T'],
    'by': ['B', 'AY'],
    'from': ['F', 'R', 'AH', 'M'],
    'up': ['AH', 'P'],
    'about': ['AH', 'B', 'AW', 'T'],
    'into': ['IH', 'N', 'T', 'UW'],
    'through': ['TH', 'R', 'UW'],
    'during': ['D', 'UH', 'R', 'IH', 'NG'],
    'including': ['IH', 'N', 'K', 'L', 'UW', 'D', 'IH', 'NG'],
    'following': ['F', 'AO', 'L', 'OW', 'IH', 'NG'],
    'across': ['AH', 'K', 'R', 'AO', 'S'],
    'against': ['AH', 'G', 'EH', 'N', 'S', 'T'],
    'throughout': ['TH', 'R', 'UW', 'AW', 'T'],
    'concerning': ['K', 'AH', 'N', 'S', 'ER', 'N', 'IH', 'NG'],
    'considering': ['K', 'AH', 'N', 'S', 'IH', 'D', 'ER', 'IH', 'NG'],
    'regarding': ['R', 'IH', 'G', 'AA', 'R', 'D', 'IH', 'NG'],
}


def text_to_phonemes(text: str) -> List[Phoneme]:
    """
    Convert text to phonemes.
    
    Args:
        text: Input text
    
    Returns:
        List of Phoneme objects
    """
    # Normalize text
    text = text.lower().strip()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into words
    words = text.split()
    
    phonemes = []
    for word in words:
        # Get phonemes for word
        word_phonemes = _word_to_phonemes(word)
        
        # Add stress (primary stress on first syllable for simplicity)
        for i, ph in enumerate(word_phonemes):
            if ph.phoneme_type == PhonemeType.VOWEL:
                if i == 0:
                    ph.stress = 1  # Primary stress on first vowel
                else:
                    ph.stress = 0  # Unstressed
        
        phonemes.extend(word_phonemes)
        
        # Add silence between words
        if word != words[-1]:  # Not last word
            phonemes.append(Phoneme(
                symbol=' ',
                phoneme_type=PhonemeType.SILENCE,
                duration=0.1
            ))
    
    return phonemes


def _word_to_phonemes(word: str) -> List[Phoneme]:
    """Convert a single word to phonemes."""
    # Check dictionary first
    if word in SIMPLE_PHONEME_DICT:
        phoneme_codes = SIMPLE_PHONEME_DICT[word]
    else:
        # Simple rule-based conversion (fallback)
        phoneme_codes = _rule_based_phonemes(word)
    
    # Convert codes to Phoneme objects
    phonemes = []
    for code in phoneme_codes:
        if code in PHONEME_MAP:
            symbol, ph_type = PHONEME_MAP[code]
            # Estimate duration
            if ph_type == PhonemeType.VOWEL:
                duration = 0.12
            elif ph_type == PhonemeType.CONSONANT:
                duration = 0.08
            else:
                duration = 0.1
            
            phonemes.append(Phoneme(
                symbol=symbol,
                phoneme_type=ph_type,
                duration=duration
            ))
    
    return phonemes


def _rule_based_phonemes(word: str) -> List[str]:
    """Simple rule-based phoneme conversion (fallback)."""
    phonemes = []
    i = 0
    
    while i < len(word):
        char = word[i]
        
        # Check for multi-character phonemes first
        if i < len(word) - 1:
            two_char = word[i:i+2].upper()
            if two_char in PHONEME_MAP:
                phonemes.append(two_char)
                i += 2
                continue
        
        # Single character
        char_upper = char.upper()
        if char_upper in PHONEME_MAP:
            phonemes.append(char_upper)
        elif char in 'aeiou':
            # Vowel approximation
            if char == 'a':
                phonemes.append('AE')
            elif char == 'e':
                phonemes.append('EH')
            elif char == 'i':
                phonemes.append('IH')
            elif char == 'o':
                phonemes.append('AO')
            elif char == 'u':
                phonemes.append('UH')
        
        i += 1
    
    return phonemes if phonemes else ['AH']  # Default to schwa


def phoneme_to_vowel_type(phoneme: Phoneme) -> Optional[str]:
    """Map phoneme to VowelType for formant lookup."""
    symbol = phoneme.symbol.lower()
    
    # Map IPA symbols to VowelType
    vowel_map = {
        'ɑ': 'A', 'a': 'A',  # "ah"
        'æ': 'A',  # "cat" (closer to A)
        'ɛ': 'E', 'e': 'E',  # "eh"
        'ɪ': 'I', 'i': 'I',  # "ee"
        'ɔ': 'O', 'o': 'O',  # "oh"
        'ʊ': 'U', 'u': 'U',  # "oo"
        'ə': 'SCHWA',  # "uh"
        'aɪ': 'I',  # "eye"
        'aʊ': 'A',  # "cow"
        'eɪ': 'E',  # "ate"
        'oʊ': 'O',  # "show"
        'ɔɪ': 'O',  # "toy"
    }
    
    return vowel_map.get(symbol)

