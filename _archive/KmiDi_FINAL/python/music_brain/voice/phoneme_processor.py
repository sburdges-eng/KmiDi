"""
Phoneme Processor - Text to Phoneme Conversion

Converts lyrics text to phoneme sequences with duration estimation.
Uses CMU phoneme set and g2p_en for grapheme-to-phoneme conversion.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

# Try to import g2p_en, fallback to simple rules if not available
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    print("Warning: g2p_en not available. Using simple phoneme rules.")


@dataclass
class Phoneme:
    """Represents a single phoneme with timing."""
    symbol: str  # CMU phoneme symbol (e.g., "AH", "K", "S")
    duration_ms: float  # Duration in milliseconds
    start_time_ms: float = 0.0  # Start time relative to phrase start

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "duration_ms": self.duration_ms,
            "start_time_ms": self.start_time_ms
        }


@dataclass
class PhonemeSequence:
    """Sequence of phonemes with timing information."""
    phonemes: List[Phoneme]
    total_duration_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "phonemes": [p.to_dict() for p in self.phonemes],
            "total_duration_ms": self.total_duration_ms
        }


# CMU Phoneme Set (39 phonemes)
CMU_PHONEMES = {
    # Vowels
    "AA": "father", "AE": "cat", "AH": "but", "AO": "law", "AW": "cow",
    "AY": "hide", "EH": "red", "ER": "her", "EY": "ate", "IH": "it",
    "IY": "eat", "OW": "show", "OY": "toy", "UH": "book", "UW": "two",

    # Consonants - Stops
    "B": "bat", "D": "dog", "G": "go", "K": "cat", "P": "pat", "T": "top",

    # Consonants - Fricatives
    "CH": "church", "DH": "the", "F": "fat", "HH": "hat", "JH": "judge",
    "S": "sat", "SH": "ship", "TH": "thin", "V": "vat", "Z": "zoo", "ZH": "measure",

    # Consonants - Nasals
    "M": "mat", "N": "not", "NG": "sing",

    # Consonants - Liquids/Glides
    "L": "let", "R": "red", "W": "wet", "Y": "yet",

    # Silence
    "SIL": "silence", "SP": "space"
}

# Phoneme duration estimates (relative, will be scaled)
PHONEME_DURATIONS = {
    # Vowels (longer)
    "AA": 1.2, "AE": 1.0, "AH": 1.0, "AO": 1.2, "AW": 1.3, "AY": 1.3,
    "EH": 1.0, "ER": 1.1, "EY": 1.2, "IH": 0.9, "IY": 1.1, "OW": 1.2,
    "OY": 1.3, "UH": 1.0, "UW": 1.1,

    # Stops (short, with closure)
    "B": 0.3, "D": 0.3, "G": 0.3, "K": 0.3, "P": 0.3, "T": 0.3,

    # Fricatives (medium)
    "CH": 0.4, "DH": 0.4, "F": 0.4, "HH": 0.3, "JH": 0.4,
    "S": 0.5, "SH": 0.5, "TH": 0.4, "V": 0.4, "Z": 0.5, "ZH": 0.5,

    # Nasals (medium-long)
    "M": 0.4, "N": 0.4, "NG": 0.5,

    # Liquids/Glides
    "L": 0.4, "R": 0.4, "W": 0.3, "Y": 0.3,

    # Silence
    "SIL": 0.2, "SP": 0.1
}


class PhonemeProcessor:
    """
    Processes text into phoneme sequences with duration estimation.
    """

    def __init__(self):
        """Initialize phoneme processor."""
        if G2P_AVAILABLE:
            self.g2p = G2p()
        else:
            self.g2p = None
            print("Using simple phoneme rules (g2p_en not available)")

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence.

        Args:
            text: Input text (lyrics)

        Returns:
            List of phoneme symbols
        """
        # Clean text
        text = text.strip().upper()

        if self.g2p:
            # Use g2p_en for accurate conversion
            try:
                phonemes = self.g2p(text)
                # Filter out stress markers and convert to list
                phoneme_list = [p for p in phonemes if p in CMU_PHONEMES or p in [" ", ".", ","]]
                return phoneme_list
            except Exception as e:
                print(f"g2p_en error: {e}, falling back to simple rules")
                return self._simple_phoneme_rules(text)
        else:
            return self._simple_phoneme_rules(text)

    def _simple_phoneme_rules(self, text: str) -> List[str]:
        """
        Simple rule-based phoneme conversion (fallback).

        This is a very basic implementation - g2p_en is much better.
        """
        phonemes = []

        # Simple vowel mapping
        vowel_map = {
            "A": "AE", "E": "EH", "I": "IH", "O": "AO", "U": "UH"
        }

        # Simple consonant mapping
        consonant_map = {
            "B": "B", "C": "K", "D": "D", "F": "F", "G": "G",
            "H": "HH", "J": "JH", "K": "K", "L": "L", "M": "M",
            "N": "N", "P": "P", "Q": "K", "R": "R", "S": "S",
            "T": "T", "V": "V", "W": "W", "X": "K S", "Y": "Y", "Z": "Z"
        }

        for char in text:
            if char.upper() in vowel_map:
                phonemes.append(vowel_map[char.upper()])
            elif char.upper() in consonant_map:
                phonemes.append(consonant_map[char.upper()])
            elif char == " ":
                phonemes.append("SP")
            elif char in ".,!?":
                phonemes.append("SIL")

        return phonemes if phonemes else ["SIL"]

    def estimate_durations(
        self,
        phonemes: List[str],
        tempo_bpm: float = 120.0,
        total_duration_ms: Optional[float] = None
    ) -> PhonemeSequence:
        """
        Estimate phoneme durations.

        Args:
            phonemes: List of phoneme symbols
            tempo_bpm: Tempo in BPM (affects overall speed)
            total_duration_ms: Optional fixed total duration

        Returns:
            PhonemeSequence with timing
        """
        # Base duration per phoneme (scaled by tempo)
        tempo_factor = 120.0 / tempo_bpm  # Faster tempo = shorter durations

        # Get relative durations
        relative_durations = []
        for phoneme in phonemes:
            if phoneme in PHONEME_DURATIONS:
                relative_durations.append(PHONEME_DURATIONS[phoneme])
            else:
                relative_durations.append(0.5)  # Default

        # Normalize to total duration or use tempo-based scaling
        if total_duration_ms:
            total_relative = sum(relative_durations)
            if total_relative > 0:
                scale = total_duration_ms / total_relative
            else:
                scale = 50.0  # Default 50ms per phoneme
        else:
            # Use tempo-based scaling
            base_ms_per_phoneme = 100.0 * tempo_factor
            scale = base_ms_per_phoneme

        # Create phoneme objects with timing
        phoneme_objects = []
        current_time = 0.0

        for i, (phoneme, rel_dur) in enumerate(zip(phonemes, relative_durations)):
            duration_ms = rel_dur * scale

            phoneme_obj = Phoneme(
                symbol=phoneme,
                duration_ms=duration_ms,
                start_time_ms=current_time
            )
            phoneme_objects.append(phoneme_obj)
            current_time += duration_ms

        return PhonemeSequence(
            phonemes=phoneme_objects,
            total_duration_ms=current_time
        )

    def align_to_melody(
        self,
        phonemes: List[str],
        melody_notes: List[int],
        tempo_bpm: float = 120.0
    ) -> PhonemeSequence:
        """
        Align phonemes to melody notes.

        Args:
            phonemes: List of phoneme symbols
            melody_notes: List of MIDI note numbers
            tempo_bpm: Tempo in BPM

        Returns:
            PhonemeSequence aligned to melody
        """
        # Calculate note duration
        beat_duration_ms = (60.0 / tempo_bpm) * 1000.0
        note_duration_ms = beat_duration_ms  # One note per beat (can be adjusted)

        # Distribute phonemes across notes
        total_notes = len(melody_notes)
        total_phonemes = len(phonemes)

        if total_notes == 0:
            # No melody, use default durations
            return self.estimate_durations(phonemes, tempo_bpm)

        # Calculate phonemes per note
        phonemes_per_note = max(1, total_phonemes / total_notes)

        phoneme_objects = []
        current_time = 0.0
        phoneme_idx = 0

        for note_idx in range(total_notes):
            # Determine how many phonemes for this note
            start_phoneme = int(note_idx * phonemes_per_note)
            end_phoneme = int((note_idx + 1) * phonemes_per_note)
            end_phoneme = min(end_phoneme, total_phonemes)

            note_phonemes = phonemes[start_phoneme:end_phoneme]

            if not note_phonemes:
                # No phonemes for this note, add silence
                phoneme_obj = Phoneme(
                    symbol="SIL",
                    duration_ms=note_duration_ms,
                    start_time_ms=current_time
                )
                phoneme_objects.append(phoneme_obj)
            else:
                # Distribute note duration across phonemes
                phoneme_duration = note_duration_ms / len(note_phonemes)

                for phoneme in note_phonemes:
                    phoneme_obj = Phoneme(
                        symbol=phoneme,
                        duration_ms=phoneme_duration,
                        start_time_ms=current_time
                    )
                    phoneme_objects.append(phoneme_obj)
                    current_time += phoneme_duration

            current_time = (note_idx + 1) * note_duration_ms

        return PhonemeSequence(
            phonemes=phoneme_objects,
            total_duration_ms=current_time
        )

    def process_lyrics(
        self,
        lyrics: str,
        melody_notes: Optional[List[int]] = None,
        tempo_bpm: float = 120.0
    ) -> PhonemeSequence:
        """
        Complete processing: text → phonemes → durations.

        Args:
            lyrics: Lyrics text
            melody_notes: Optional MIDI notes for alignment
            tempo_bpm: Tempo in BPM

        Returns:
            PhonemeSequence ready for synthesis
        """
        # Convert text to phonemes
        phonemes = self.text_to_phonemes(lyrics)

        # Estimate or align durations
        if melody_notes:
            return self.align_to_melody(phonemes, melody_notes, tempo_bpm)
        else:
            return self.estimate_durations(phonemes, tempo_bpm)


# Convenience function
def process_lyrics(
    lyrics: str,
    melody_notes: Optional[List[int]] = None,
    tempo_bpm: float = 120.0
) -> PhonemeSequence:
    """
    Process lyrics into phoneme sequence.

    Args:
        lyrics: Lyrics text
        melody_notes: Optional MIDI notes
        tempo_bpm: Tempo in BPM

    Returns:
        PhonemeSequence
    """
    processor = PhonemeProcessor()
    return processor.process_lyrics(lyrics, melody_notes, tempo_bpm)
