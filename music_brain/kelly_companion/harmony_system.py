"""
INTELLIGENT HARMONY SYSTEM
Unified integration of all four modules:
1. Chord Detector (rule-based + ML hybrid)
2. Key + Scale Analyzer
3. Probabilistic Harmony Engine
4. Chord Memory + Long-Term Context
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

from .chord_detector import ChordDetector, PolyphonicScorer, JazzChordAnalyzer, ChordMatch
from .key_analyzer import (
    KeyScaleAnalyzer, KeyAnalyzer, KeyEstimate, Mode,
    ModalInterchangeDetector, SecondaryDominantDetector
)
from .harmony_engine import (
    ProbabilisticHarmonyEngine, Genre, CadenceDetector,
    VoiceLeadingAnalyzer, TensionResolutionAnalyzer
)
from .chord_memory import (
    ChordMemorySystem, SectionType, EmotionalArc, ChordEvent
)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class HarmonyAnalysis:
    """Complete analysis result"""
    # Chord identification
    chord_name: str
    root: str
    quality: str
    bass_note: Optional[str]
    confidence: float
    alternatives: List[str]
    
    # Key context
    detected_key: str
    roman_numeral: str
    harmonic_function: str
    
    # Advanced analysis
    is_borrowed: bool
    borrowed_from: Optional[str]
    is_secondary_dominant: bool
    secondary_target: Optional[str]
    
    # Probabilistic
    tension_level: float
    next_chord_predictions: List[Tuple[str, float]]
    cadence: Optional[Dict]
    
    # Context
    position_in_phrase: int
    emotional_context: str


class IntelligentHarmonySystem:
    """
    Main class that integrates all harmony analysis modules.
    
    Usage:
        system = IntelligentHarmonySystem(genre=Genre.JAZZ)
        
        # Process notes
        result = system.analyze_notes(['C', 'E', 'G', 'Bb'])
        
        # Get predictions
        predictions = system.predict_next()
        
        # Get full context
        context = system.get_harmonic_context()
    """
    
    def __init__(self, genre: Genre = Genre.POP):
        # Initialize all modules
        self.chord_detector = ChordDetector()
        self.polyphonic_scorer = PolyphonicScorer()
        self.jazz_analyzer = JazzChordAnalyzer()
        
        self.key_analyzer = KeyScaleAnalyzer()
        self.key_detector = KeyAnalyzer()
        
        self.harmony_engine = ProbabilisticHarmonyEngine(genre)
        self.tension_analyzer = TensionResolutionAnalyzer()
        self.voice_leading = VoiceLeadingAnalyzer()
        
        self.memory = ChordMemorySystem()
        
        self.genre = genre
        
        # State
        self.current_key: Optional[KeyEstimate] = None
        self.pitch_history: List[int] = []
        self.progression_history: List[str] = []
        self.last_voicing: List[int] = []
    
    def set_genre(self, genre: Genre):
        """Change genre context"""
        self.genre = genre
        self.harmony_engine = ProbabilisticHarmonyEngine(genre)
    
    def set_key(self, root: str, mode: str = "major"):
        """Manually set key"""
        mode_enum = Mode.IONIAN if mode.lower() in ['major', 'ionian'] else Mode.AEOLIAN
        pc = NOTE_NAMES.index(root) if root in NOTE_NAMES else 0
        self.current_key = KeyEstimate(root, mode_enum, 1.0, pc)
        self.memory.set_key(f"{root} {mode}")
    
    def set_section(self, section: str):
        """Set current song section"""
        section_map = {
            'intro': SectionType.INTRO,
            'verse': SectionType.VERSE,
            'prechorus': SectionType.PRECHORUS,
            'chorus': SectionType.CHORUS,
            'bridge': SectionType.BRIDGE,
            'breakdown': SectionType.BREAKDOWN,
            'outro': SectionType.OUTRO,
            'solo': SectionType.SOLO,
        }
        self.memory.set_section(section_map.get(section.lower(), SectionType.UNKNOWN))
    
    def analyze_notes(self, notes: List[str], duration: float = 1.0) -> HarmonyAnalysis:
        """
        Main analysis method - takes raw notes, returns full analysis
        """
        # Step 1: Detect chord
        pitch_classes = self.chord_detector.notes_to_pitch_classes(notes)
        matches = self.chord_detector.detect_chord(notes)
        
        if not matches:
            return self._empty_analysis()
        
        best_match = matches[0]
        
        # Update pitch history for key detection
        self.pitch_history.extend(pitch_classes)
        if len(self.pitch_history) > 100:
            self.pitch_history = self.pitch_history[-100:]
        
        # Step 2: Key context
        if self.current_key is None or len(self.pitch_history) % 8 == 0:
            self._update_key_estimate()
        
        # Step 3: Analyze chord in key
        root_pc = NOTE_NAMES.index(best_match.root)
        chord_analysis = {}
        
        if self.current_key:
            chord_analysis = self.key_analyzer.analyze_chord_in_key(
                root_pc, best_match.quality.value,
                self.current_key,
                prev_chord=self._get_prev_chord_tuple(),
                next_chord=None  # We don't know the future
            )
        
        # Get Roman numeral
        roman = chord_analysis.get('roman_numeral', '?')
        
        # Step 4: Update memory
        event = self.memory.process_chord(
            root=best_match.root,
            quality=best_match.quality.value,
            bass=best_match.bass_note,
            roman_numeral=roman,
            duration=duration,
            notes=pitch_classes,
            tension=self.tension_analyzer.chord_tension(set(best_match.intervals_matched))
        )
        
        # Update progression history
        self.progression_history.append(roman)
        if len(self.progression_history) > 20:
            self.progression_history = self.progression_history[-20:]
        
        # Step 5: Get predictions
        predictions = self.harmony_engine.suggest_next_chord(
            self.progression_history[-4:] if len(self.progression_history) >= 4 else self.progression_history
        )
        
        # Step 6: Cadence detection
        cadence = self.harmony_engine.cadence.detect_cadence(self.progression_history)
        
        # Step 7: Voice leading (if we have previous voicing)
        if self.last_voicing:
            vl_score = self.voice_leading.voice_leading_score(self.last_voicing, pitch_classes)
        self.last_voicing = pitch_classes
        
        # Build result
        return HarmonyAnalysis(
            chord_name=self.chord_detector.get_slash_chord(best_match),
            root=best_match.root,
            quality=best_match.quality.value,
            bass_note=best_match.bass_note,
            confidence=best_match.confidence,
            alternatives=[m.name for m in matches[1:4]],
            
            detected_key=f"{self.current_key.root} {self.current_key.mode.value}" if self.current_key else "unknown",
            roman_numeral=roman,
            harmonic_function=chord_analysis.get('function', '?'),
            
            is_borrowed='borrowed_chord' in chord_analysis,
            borrowed_from=chord_analysis.get('borrowed_chord', {}).get('borrowed_from'),
            is_secondary_dominant='secondary_function' in chord_analysis,
            secondary_target=chord_analysis.get('secondary_function', {}).get('target'),
            
            tension_level=event.tension_level,
            next_chord_predictions=[(p['chord'], p['score']) for p in predictions[:3]],
            cadence=cadence,
            
            position_in_phrase=len(self.memory.phrase_memory.current_phrase_chords),
            emotional_context=self._get_emotional_context()
        )
    
    def analyze_midi_pitches(self, midi_pitches: List[int], duration: float = 1.0) -> HarmonyAnalysis:
        """Analyze from MIDI pitch numbers"""
        notes = [NOTE_NAMES[p % 12] for p in midi_pitches]
        return self.analyze_notes(notes, duration)
    
    def predict_next(self, prefer_resolution: bool = False) -> List[Dict]:
        """Get predictions for next chord"""
        return self.harmony_engine.suggest_next_chord(
            self.progression_history[-4:],
            prefer_resolution=prefer_resolution
        )
    
    def get_harmonic_context(self) -> Dict:
        """Get full current harmonic context"""
        memory_context = self.memory.get_context()
        predictions = self.memory.predict_context()
        
        return {
            **memory_context,
            **predictions,
            'genre': self.genre.value,
            'detected_key': f"{self.current_key.root} {self.current_key.mode.value}" if self.current_key else None,
            'key_confidence': self.current_key.confidence if self.current_key else 0,
            'progression_history': self.progression_history[-8:],
            'recurring_patterns': [
                m.pattern for m in self.memory.motif_tracker.get_signature_motifs()
            ]
        }
    
    def get_emotional_arc(self) -> List[Dict]:
        """Get emotional arc of the piece so far"""
        arcs = []
        for phrase in self.memory.phrase_memory.phrases:
            arcs.append({
                'section': phrase.section.value,
                'progression': phrase.progression,
                'emotion': phrase.emotional_arc.value,
                'cadential': phrase.is_cadential
            })
        return arcs
    
    def suggest_voice_leading(self, target_chord_pcs: Set[int]) -> List[int]:
        """Suggest optimal voicing for next chord"""
        return self.voice_leading.suggest_voicing(target_chord_pcs, self.last_voicing)
    
    def reset(self):
        """Reset all state"""
        self.memory = ChordMemorySystem()
        self.current_key = None
        self.pitch_history = []
        self.progression_history = []
        self.last_voicing = []
    
    def _update_key_estimate(self):
        """Update key estimate from pitch history"""
        if len(self.pitch_history) < 8:
            return
        
        estimates = self.key_detector.detect_key(self.pitch_history)
        if estimates:
            self.current_key = estimates[0]
            self.memory.set_key(f"{self.current_key.root} {self.current_key.mode.value}")
    
    def _get_prev_chord_tuple(self) -> Optional[Tuple[int, str]]:
        """Get previous chord as (pitch_class, quality) tuple"""
        context = self.memory.get_context()
        last = context.get('last_chord')
        if last:
            pc = NOTE_NAMES.index(last.root) if last.root in NOTE_NAMES else 0
            return (pc, last.quality)
        return None
    
    def _get_emotional_context(self) -> str:
        """Get current emotional context string"""
        predictions = self.memory.predict_context()
        
        if predictions.get('building_to_climax'):
            return 'building_to_climax'
        
        # Check recent phrase arcs
        if self.memory.phrase_memory.phrases:
            last_arc = self.memory.phrase_memory.phrases[-1].emotional_arc
            return last_arc.value
        
        return 'stable'
    
    def _empty_analysis(self) -> HarmonyAnalysis:
        """Return empty analysis when no chord detected"""
        return HarmonyAnalysis(
            chord_name='?', root='?', quality='?', bass_note=None,
            confidence=0, alternatives=[],
            detected_key='unknown', roman_numeral='?', harmonic_function='?',
            is_borrowed=False, borrowed_from=None,
            is_secondary_dominant=False, secondary_target=None,
            tension_level=0, next_chord_predictions=[],
            cadence=None, position_in_phrase=0, emotional_context='unknown'
        )


# Convenience functions
def quick_analyze(notes: List[str], key: str = None, genre: str = 'pop') -> Dict:
    """Quick one-shot chord analysis"""
    system = IntelligentHarmonySystem(Genre[genre.upper()])
    if key:
        parts = key.split()
        system.set_key(parts[0], parts[1] if len(parts) > 1 else 'major')
    
    result = system.analyze_notes(notes)
    return {
        'chord': result.chord_name,
        'confidence': f"{result.confidence:.0%}",
        'key': result.detected_key,
        'function': result.harmonic_function,
        'roman': result.roman_numeral,
        'tension': f"{result.tension_level:.2f}",
        'next_likely': result.next_chord_predictions
    }


def analyze_progression(chords: List[List[str]], key: str = None, genre: str = 'pop') -> Dict:
    """Analyze a full chord progression"""
    system = IntelligentHarmonySystem(Genre[genre.upper()])
    if key:
        parts = key.split()
        system.set_key(parts[0], parts[1] if len(parts) > 1 else 'major')
    
    results = []
    for notes in chords:
        r = system.analyze_notes(notes)
        results.append({
            'chord': r.chord_name,
            'roman': r.roman_numeral,
            'function': r.harmonic_function,
            'tension': r.tension_level
        })
    
    context = system.get_harmonic_context()
    
    return {
        'chords': results,
        'key': context.get('detected_key'),
        'recurring_patterns': context.get('recurring_patterns', []),
        'emotional_arc': system.get_emotional_arc()
    }


if __name__ == "__main__":
    # Demo
    print("=== INTELLIGENT HARMONY SYSTEM DEMO ===\n")
    
    system = IntelligentHarmonySystem(Genre.POP)
    system.set_key('C', 'major')
    
    # Process a progression
    progression = [
        ['C', 'E', 'G'],        # C
        ['G', 'B', 'D', 'F'],   # G7
        ['A', 'C', 'E'],        # Am
        ['F', 'A', 'C'],        # F
    ]
    
    print("Analyzing: C → G7 → Am → F\n")
    
    for notes in progression:
        result = system.analyze_notes(notes)
        print(f"Notes: {notes}")
        print(f"  Chord: {result.chord_name} ({result.roman_numeral})")
        print(f"  Function: {result.harmonic_function}")
        print(f"  Tension: {result.tension_level:.2f}")
        print(f"  Key: {result.detected_key}")
        if result.cadence:
            print(f"  Cadence: {result.cadence['name']}")
        print()
    
    print("--- PREDICTIONS ---")
    predictions = system.predict_next()
    for p in predictions[:3]:
        print(f"  {p['chord']}: {p['score']:.2f} ({', '.join(p['reasons'])})")
    
    print("\n--- CONTEXT ---")
    context = system.get_harmonic_context()
    print(f"  Key: {context['detected_key']}")
    print(f"  Recent: {context['recent_chords']}")
    print(f"  Patterns: {context['recurring_patterns']}")
