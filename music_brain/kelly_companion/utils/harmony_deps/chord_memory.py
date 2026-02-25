"""
Chord Memory - Long-term context and pattern tracking.

Provides:
- ChordMemorySystem: Track chord history and patterns
- SectionType: Song section types
- EmotionalArc: Emotional progression types
- ChordEvent: Individual chord event
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class SectionType(Enum):
    """Song section types."""
    UNKNOWN = "unknown"
    INTRO = "intro"
    VERSE = "verse"
    PRECHORUS = "prechorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"
    SOLO = "solo"


class EmotionalArc(Enum):
    """Emotional progression types."""
    STABLE = "stable"
    BUILDING = "building"
    CLIMAX = "climax"
    RELEASING = "releasing"
    COLLAPSING = "collapsing"


@dataclass
class ChordEvent:
    """Individual chord event with metadata."""
    root: str
    quality: str
    bass: Optional[str]
    roman_numeral: str
    duration: float
    notes: List[int]
    tension_level: float
    timestamp: float = 0.0


@dataclass
class Phrase:
    """Musical phrase with emotional arc."""
    section: SectionType
    progression: List[str]
    emotional_arc: EmotionalArc
    is_cadential: bool = False
    start_time: float = 0.0
    end_time: float = 0.0


class PhraseMemory:
    """Track musical phrases."""
    
    def __init__(self):
        self.phrases: List[Phrase] = []
        self.current_phrase_chords: List[str] = []
        self.current_section: SectionType = SectionType.UNKNOWN
    
    def add_chord(self, chord: str):
        """Add chord to current phrase."""
        self.current_phrase_chords.append(chord)
    
    def finalize_phrase(self, section: SectionType, is_cadential: bool = False):
        """Finalize current phrase."""
        if self.current_phrase_chords:
            # Determine emotional arc
            arc = self._determine_arc(self.current_phrase_chords)
            
            phrase = Phrase(
                section=section,
                progression=self.current_phrase_chords.copy(),
                emotional_arc=arc,
                is_cadential=is_cadential
            )
            self.phrases.append(phrase)
            self.current_phrase_chords = []
    
    def _determine_arc(self, progression: List[str]) -> EmotionalArc:
        """Determine emotional arc from progression."""
        if not progression:
            return EmotionalArc.STABLE
        
        # Simple heuristic: check for tension/resolution patterns
        if progression[-1] == 'I':
            return EmotionalArc.RELEASING
        elif 'V' in progression:
            return EmotionalArc.BUILDING
        else:
            return EmotionalArc.STABLE


class MotifTracker:
    """Track recurring harmonic patterns."""
    
    def __init__(self):
        self.patterns: Dict[Tuple[str, ...], int] = {}
        self.window_size = 4
    
    def add_progression(self, progression: List[str]):
        """Track progression patterns."""
        if len(progression) < self.window_size:
            return
        
        # Extract sliding windows
        for i in range(len(progression) - self.window_size + 1):
            window = tuple(progression[i:i+self.window_size])
            self.patterns[window] = self.patterns.get(window, 0) + 1
    
    def get_signature_motifs(self, min_count: int = 2) -> List[Dict]:
        """Get recurring patterns."""
        motifs = []
        for pattern, count in self.patterns.items():
            if count >= min_count:
                motifs.append({
                    'pattern': list(pattern),
                    'count': count
                })
        return sorted(motifs, key=lambda m: m['count'], reverse=True)


class ChordMemorySystem:
    """Complete chord memory system."""
    
    def __init__(self):
        self.phrase_memory = PhraseMemory()
        self.motif_tracker = MotifTracker()
        self.chord_history: deque = deque(maxlen=50)
        self.current_key: Optional[str] = None
        self.current_section: SectionType = SectionType.UNKNOWN
    
    def set_key(self, key: str):
        """Set current key."""
        self.current_key = key
    
    def set_section(self, section: SectionType):
        """Set current section."""
        self.current_section = section
        self.phrase_memory.current_section = section
    
    def process_chord(
        self,
        root: str,
        quality: str,
        bass: Optional[str],
        roman_numeral: str,
        duration: float,
        notes: List[int],
        tension: float
    ) -> ChordEvent:
        """Process and store chord event."""
        event = ChordEvent(
            root=root,
            quality=quality,
            bass=bass,
            roman_numeral=roman_numeral,
            duration=duration,
            notes=notes,
            tension_level=tension
        )
        
        self.chord_history.append(event)
        self.phrase_memory.add_chord(roman_numeral)
        self.motif_tracker.add_progression([roman_numeral])
        
        return event
    
    def get_context(self) -> Dict:
        """Get current harmonic context."""
        last_chord = self.chord_history[-1] if self.chord_history else None
        
        return {
            'last_chord': last_chord,
            'key': self.current_key,
            'section': self.current_section.value,
            'recent_chords': [c.roman_numeral for c in list(self.chord_history)[-8:]],
            'phrase_count': len(self.phrase_memory.phrases)
        }
    
    def predict_context(self) -> Dict:
        """Predict upcoming harmonic context."""
        # Simple predictions based on current state
        predictions = {
            'building_to_climax': False,
            'likely_resolution': None,
            'section_transition_likely': False
        }
        
        if self.chord_history:
            last = self.chord_history[-1]
            if last.roman_numeral == 'V':
                predictions['likely_resolution'] = 'I'
                predictions['building_to_climax'] = True
        
        return predictions
