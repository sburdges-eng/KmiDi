"""
Music Theory Analyzer - Deep analysis of scales, modes, arpeggios, triads, and harmonic patterns.

Detects and identifies:
- Scales (major, minor, pentatonic, blues, chromatic, etc.)
- Modes (Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian)
- Arpeggios and broken chords
- Triads (major, minor, diminished, augmented)
- Seventh chords and extensions
- Melodic patterns and intervals
- Harmonic functions and progressions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from pathlib import Path
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# =================================================================
# CONSTANTS AND DEFINITIONS
# =================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Scale definitions as intervals from root (semitones)
SCALES = {
    # Major and Natural Minor
    'major': [0, 2, 4, 5, 7, 9, 11],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    
    # Modes (built on major scale degrees)
    'ionian': [0, 2, 4, 5, 7, 9, 11],        # Same as major
    'dorian': [0, 2, 3, 5, 7, 9, 10],        # Minor with raised 6th
    'phrygian': [0, 1, 3, 5, 7, 8, 10],      # Minor with flat 2nd
    'lydian': [0, 2, 4, 6, 7, 9, 11],        # Major with raised 4th
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],    # Major with flat 7th
    'aeolian': [0, 2, 3, 5, 7, 8, 10],       # Same as natural minor
    'locrian': [0, 1, 3, 5, 6, 8, 10],       # Diminished scale
    
    # Pentatonic scales
    'major_pentatonic': [0, 2, 4, 7, 9],
    'minor_pentatonic': [0, 3, 5, 7, 10],
    
    # Blues scales
    'blues': [0, 3, 5, 6, 7, 10],            # Minor pentatonic + blue note
    'major_blues': [0, 2, 3, 4, 7, 9],       # Major pentatonic + blue note
    
    # Exotic scales
    'whole_tone': [0, 2, 4, 6, 8, 10],
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11],  # Half-whole
    'diminished_half_whole': [0, 1, 3, 4, 6, 7, 9, 10],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    
    # World scales
    'hungarian_minor': [0, 2, 3, 6, 7, 8, 11],
    'phrygian_dominant': [0, 1, 4, 5, 7, 8, 10],  # Spanish/Jewish
    'double_harmonic': [0, 1, 4, 5, 7, 8, 11],    # Byzantine/Arabic
    'japanese': [0, 1, 5, 7, 8],                   # In scale
    'hirajoshi': [0, 2, 3, 7, 8],
    'persian': [0, 1, 4, 5, 6, 8, 11],
    
    # Jazz scales
    'bebop_dominant': [0, 2, 4, 5, 7, 9, 10, 11],
    'bebop_major': [0, 2, 4, 5, 7, 8, 9, 11],
    'altered': [0, 1, 3, 4, 6, 8, 10],           # Super Locrian
    'lydian_dominant': [0, 2, 4, 6, 7, 9, 10],   # Lydian b7
}

# Mode characteristics for emotional mapping
MODE_CHARACTERISTICS = {
    'ionian': {'brightness': 1.0, 'tension': 0.0, 'mood': 'happy, bright, stable'},
    'dorian': {'brightness': 0.6, 'tension': 0.2, 'mood': 'jazzy, soulful, melancholic hope'},
    'phrygian': {'brightness': 0.2, 'tension': 0.6, 'mood': 'dark, Spanish, exotic, tense'},
    'lydian': {'brightness': 1.0, 'tension': 0.3, 'mood': 'dreamy, ethereal, floating'},
    'mixolydian': {'brightness': 0.8, 'tension': 0.2, 'mood': 'bluesy, rock, dominant'},
    'aeolian': {'brightness': 0.3, 'tension': 0.4, 'mood': 'sad, minor, natural'},
    'locrian': {'brightness': 0.1, 'tension': 0.9, 'mood': 'unstable, dissonant, dark'},
}

# Chord/Triad definitions (intervals from root)
TRIADS = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'diminished': [0, 3, 6],
    'augmented': [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
}

SEVENTH_CHORDS = {
    'major7': [0, 4, 7, 11],
    'minor7': [0, 3, 7, 10],
    'dominant7': [0, 4, 7, 10],
    'diminished7': [0, 3, 6, 9],
    'half_diminished7': [0, 3, 6, 10],
    'minor_major7': [0, 3, 7, 11],
    'augmented7': [0, 4, 8, 10],
    'augmented_major7': [0, 4, 8, 11],
}

EXTENDED_CHORDS = {
    'add9': [0, 4, 7, 14],
    'add11': [0, 4, 7, 17],
    '6': [0, 4, 7, 9],
    '6/9': [0, 4, 7, 9, 14],
    '9': [0, 4, 7, 10, 14],
    'major9': [0, 4, 7, 11, 14],
    'minor9': [0, 3, 7, 10, 14],
    '11': [0, 4, 7, 10, 14, 17],
    '13': [0, 4, 7, 10, 14, 17, 21],
}

# Interval names
INTERVALS = {
    0: 'unison',
    1: 'minor_2nd',
    2: 'major_2nd',
    3: 'minor_3rd',
    4: 'major_3rd',
    5: 'perfect_4th',
    6: 'tritone',
    7: 'perfect_5th',
    8: 'minor_6th',
    9: 'major_6th',
    10: 'minor_7th',
    11: 'major_7th',
    12: 'octave',
}


# =================================================================
# DATA CLASSES
# =================================================================

@dataclass
class ScaleDetection:
    """Result of scale detection."""
    scale_name: str
    root: str
    intervals: List[int]
    confidence: float
    pitch_classes_present: List[int]
    pitch_classes_missing: List[int]
    is_mode: bool = False
    parent_scale: Optional[str] = None
    characteristics: Dict = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        return f"{self.root} {self.scale_name}"
    
    def to_dict(self) -> Dict:
        return {
            "scale": self.scale_name,
            "root": self.root,
            "full_name": self.full_name,
            "confidence": self.confidence,
            "intervals": self.intervals,
            "is_mode": self.is_mode,
            "characteristics": self.characteristics,
        }


@dataclass
class TriadDetection:
    """Result of triad/chord detection."""
    chord_name: str
    root: str
    quality: str
    intervals: List[int]
    notes: List[int]  # MIDI notes or pitch classes
    inversion: int  # 0 = root, 1 = first, 2 = second
    confidence: float
    voicing: str = "close"  # close, open, spread
    
    @property
    def full_name(self) -> str:
        inv_suffix = ["", "/1st", "/2nd", "/3rd"][min(self.inversion, 3)]
        return f"{self.root}{self.quality}{inv_suffix}"
    
    def to_dict(self) -> Dict:
        return {
            "chord": self.full_name,
            "root": self.root,
            "quality": self.quality,
            "inversion": self.inversion,
            "confidence": self.confidence,
            "voicing": self.voicing,
        }


@dataclass
class ArpeggioDetection:
    """Result of arpeggio pattern detection."""
    chord_base: str  # e.g., "Cmaj", "Am7"
    pattern: str  # e.g., "ascending", "descending", "broken"
    notes: List[int]  # MIDI notes in order
    intervals: List[int]  # Intervals in pattern
    speed_category: str  # slow, medium, fast
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "chord": self.chord_base,
            "pattern": self.pattern,
            "speed": self.speed_category,
            "confidence": self.confidence,
            "note_count": len(self.notes),
        }


@dataclass
class IntervalAnalysis:
    """Analysis of melodic intervals."""
    intervals: List[Tuple[int, str]]  # (semitones, name) pairs
    interval_histogram: Dict[str, int]
    consonant_ratio: float  # Ratio of consonant intervals
    average_interval_size: float
    melodic_contour: str  # "ascending", "descending", "arch", "wave"
    
    def to_dict(self) -> Dict:
        return {
            "interval_histogram": self.interval_histogram,
            "consonant_ratio": self.consonant_ratio,
            "average_interval_size": self.average_interval_size,
            "melodic_contour": self.melodic_contour,
        }


@dataclass
class TheoryAnalysis:
    """Complete music theory analysis result."""
    # Scale/Mode
    detected_scales: List[ScaleDetection] = field(default_factory=list)
    primary_scale: Optional[ScaleDetection] = None
    
    # Chords
    detected_triads: List[TriadDetection] = field(default_factory=list)
    detected_sevenths: List[TriadDetection] = field(default_factory=list)
    
    # Patterns
    detected_arpeggios: List[ArpeggioDetection] = field(default_factory=list)
    
    # Intervals
    interval_analysis: Optional[IntervalAnalysis] = None
    
    # Summary
    key_center: Optional[str] = None
    mode: Optional[str] = None
    harmonic_complexity: float = 0.0  # 0-1
    
    def to_dict(self) -> Dict:
        return {
            "key_center": self.key_center,
            "mode": self.mode,
            "primary_scale": self.primary_scale.to_dict() if self.primary_scale else None,
            "detected_scales": [s.to_dict() for s in self.detected_scales[:5]],
            "detected_chords": [t.to_dict() for t in self.detected_triads[:10]],
            "detected_arpeggios": [a.to_dict() for a in self.detected_arpeggios[:5]],
            "harmonic_complexity": self.harmonic_complexity,
            "interval_analysis": self.interval_analysis.to_dict() if self.interval_analysis else None,
        }


# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def midi_to_pitch_class(midi_note: int) -> int:
    """Convert MIDI note to pitch class (0-11)."""
    return midi_note % 12


def pitch_class_to_note_name(pc: int, use_flats: bool = False) -> str:
    """Convert pitch class to note name."""
    if use_flats:
        return NOTE_NAMES_FLAT[pc]
    return NOTE_NAMES[pc]


def get_pitch_class_set(notes: List[int]) -> Set[int]:
    """Get unique pitch classes from MIDI notes."""
    return set(midi_to_pitch_class(n) for n in notes)


def rotate_scale(intervals: List[int], steps: int) -> List[int]:
    """Rotate scale intervals (for mode detection)."""
    n = len(intervals)
    if n == 0:
        return intervals
    
    # Normalize steps
    steps = steps % n
    
    # Rotate and normalize to start from 0
    rotated = intervals[steps:] + [i + 12 for i in intervals[:steps]]
    base = rotated[0]
    return [(i - base) % 12 for i in rotated]


def calculate_scale_match(pitch_classes: Set[int], scale_intervals: List[int], root: int) -> float:
    """Calculate how well pitch classes match a scale."""
    # Transpose scale to root
    scale_pcs = set((root + i) % 12 for i in scale_intervals)
    
    # Calculate overlap
    present_in_scale = len(pitch_classes & scale_pcs)
    outside_scale = len(pitch_classes - scale_pcs)
    
    if len(pitch_classes) == 0:
        return 0.0
    
    # Score: reward notes in scale, penalize notes outside
    match_score = present_in_scale / len(scale_pcs)
    penalty = outside_scale * 0.15  # Each outside note reduces score
    
    return max(0.0, min(1.0, match_score - penalty))


def is_consonant_interval(semitones: int) -> bool:
    """Check if interval is consonant."""
    consonant = {0, 3, 4, 5, 7, 8, 9, 12}  # Unison, 3rds, 4ths, 5ths, 6ths, octave
    return (semitones % 12) in consonant


def detect_melodic_contour(notes: List[int]) -> str:
    """Detect overall melodic contour."""
    if len(notes) < 3:
        return "static"
    
    # Calculate direction changes
    directions = []
    for i in range(1, len(notes)):
        diff = notes[i] - notes[i-1]
        if diff > 0:
            directions.append(1)
        elif diff < 0:
            directions.append(-1)
        else:
            directions.append(0)
    
    # Analyze pattern
    ascending_count = sum(1 for d in directions if d > 0)
    descending_count = sum(1 for d in directions if d < 0)
    
    total = len(directions)
    if total == 0:
        return "static"
    
    asc_ratio = ascending_count / total
    desc_ratio = descending_count / total
    
    # Determine contour
    if asc_ratio > 0.7:
        return "ascending"
    elif desc_ratio > 0.7:
        return "descending"
    elif len(notes) > 4:
        # Check for arch (up then down) or wave
        midpoint = len(notes) // 2
        first_half = notes[:midpoint]
        second_half = notes[midpoint:]
        
        if len(first_half) > 1 and len(second_half) > 1:
            first_trend = first_half[-1] - first_half[0]
            second_trend = second_half[-1] - second_half[0]
            
            if first_trend > 0 and second_trend < 0:
                return "arch"
            elif first_trend < 0 and second_trend > 0:
                return "valley"
    
    return "wave"


# =================================================================
# THEORY ANALYZER CLASS
# =================================================================

class TheoryAnalyzer:
    """
    Comprehensive music theory analyzer.
    
    Detects scales, modes, triads, arpeggios, and harmonic patterns
    from MIDI notes or audio.
    """
    
    def __init__(self):
        """Initialize theory analyzer."""
        self.scales = SCALES
        self.triads = TRIADS
        self.seventh_chords = SEVENTH_CHORDS
    
    def analyze_notes(
        self,
        notes: List[int],
        durations: Optional[List[float]] = None,
        velocities: Optional[List[int]] = None,
    ) -> TheoryAnalysis:
        """
        Analyze a sequence of MIDI notes.
        
        Args:
            notes: List of MIDI note numbers
            durations: Optional list of note durations
            velocities: Optional list of velocities
        
        Returns:
            TheoryAnalysis with all detected patterns
        """
        if not notes:
            return TheoryAnalysis(
                detected_scales=[],
                detected_triads=[],
                detected_sevenths=[],
                detected_arpeggios=[],
            )
        
        # Get pitch classes
        pitch_classes = get_pitch_class_set(notes)
        
        # Detect scales and modes
        detected_scales = self.detect_scales(pitch_classes)
        primary_scale = detected_scales[0] if detected_scales else None
        
        # Detect triads and chords
        detected_triads = self.detect_triads_in_sequence(notes)
        detected_sevenths = self.detect_seventh_chords_in_sequence(notes)
        
        # Detect arpeggios
        detected_arpeggios = self.detect_arpeggios(notes, durations)
        
        # Analyze intervals
        interval_analysis = self.analyze_intervals(notes)
        
        # Determine key center and mode
        key_center = None
        mode = None
        if primary_scale:
            key_center = primary_scale.root
            mode = primary_scale.scale_name
        
        # Calculate harmonic complexity
        complexity = self._calculate_complexity(
            pitch_classes, detected_triads, detected_sevenths
        )
        
        return TheoryAnalysis(
            detected_scales=detected_scales,
            primary_scale=primary_scale,
            detected_triads=detected_triads,
            detected_sevenths=detected_sevenths,
            detected_arpeggios=detected_arpeggios,
            interval_analysis=interval_analysis,
            key_center=key_center,
            mode=mode,
            harmonic_complexity=complexity,
        )
    
    def detect_scales(
        self,
        pitch_classes: Set[int],
        top_n: int = 5,
    ) -> List[ScaleDetection]:
        """
        Detect possible scales from pitch classes.
        
        Args:
            pitch_classes: Set of pitch classes (0-11)
            top_n: Number of top matches to return
        
        Returns:
            List of ScaleDetection sorted by confidence
        """
        detections = []
        
        for scale_name, intervals in self.scales.items():
            for root in range(12):
                confidence = calculate_scale_match(pitch_classes, intervals, root)
                
                if confidence > 0.3:  # Minimum threshold
                    # Calculate which notes are present/missing
                    scale_pcs = set((root + i) % 12 for i in intervals)
                    present = sorted(pitch_classes & scale_pcs)
                    missing = sorted(scale_pcs - pitch_classes)
                    
                    # Check if this is a mode
                    is_mode = scale_name in MODE_CHARACTERISTICS
                    characteristics = MODE_CHARACTERISTICS.get(scale_name, {})
                    
                    detections.append(ScaleDetection(
                        scale_name=scale_name,
                        root=NOTE_NAMES[root],
                        intervals=intervals,
                        confidence=confidence,
                        pitch_classes_present=present,
                        pitch_classes_missing=missing,
                        is_mode=is_mode,
                        characteristics=characteristics,
                    ))
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections[:top_n]
    
    def detect_triads_in_sequence(
        self,
        notes: List[int],
        window_size: int = 3,
    ) -> List[TriadDetection]:
        """
        Detect triads in a note sequence.
        
        Args:
            notes: List of MIDI notes
            window_size: Notes to consider together
        
        Returns:
            List of detected triads
        """
        detections = []
        
        for i in range(len(notes) - window_size + 1):
            window = notes[i:i + window_size]
            
            # Try to identify as triad
            triad = self._identify_triad(window)
            if triad:
                detections.append(triad)
        
        return detections
    
    def detect_seventh_chords_in_sequence(
        self,
        notes: List[int],
        window_size: int = 4,
    ) -> List[TriadDetection]:
        """
        Detect seventh chords in a note sequence.
        """
        detections = []
        
        for i in range(len(notes) - window_size + 1):
            window = notes[i:i + window_size]
            
            # Try to identify as seventh chord
            chord = self._identify_seventh_chord(window)
            if chord:
                detections.append(chord)
        
        return detections
    
    def _identify_triad(self, notes: List[int]) -> Optional[TriadDetection]:
        """Identify a triad from notes."""
        if len(notes) < 3:
            return None
        
        # Get pitch classes
        pcs = sorted(set(midi_to_pitch_class(n) for n in notes))
        if len(pcs) < 3:
            return None
        
        best_match = None
        best_score = 0.0
        
        for quality, intervals in self.triads.items():
            for root_idx in range(12):
                # Generate expected pitch classes
                expected = set((root_idx + i) % 12 for i in intervals)
                
                # Check match
                match_count = len(set(pcs) & expected)
                if match_count >= 3:
                    score = match_count / len(expected)
                    
                    if score > best_score:
                        best_score = score
                        
                        # Determine inversion
                        bass_pc = midi_to_pitch_class(min(notes))
                        if bass_pc == root_idx:
                            inversion = 0
                        elif bass_pc == (root_idx + intervals[1]) % 12:
                            inversion = 1
                        elif bass_pc == (root_idx + intervals[2]) % 12:
                            inversion = 2
                        else:
                            inversion = 0
                        
                        # Determine voicing
                        note_range = max(notes) - min(notes)
                        if note_range <= 12:
                            voicing = "close"
                        elif note_range <= 24:
                            voicing = "open"
                        else:
                            voicing = "spread"
                        
                        best_match = TriadDetection(
                            chord_name=f"{NOTE_NAMES[root_idx]}{quality}",
                            root=NOTE_NAMES[root_idx],
                            quality=quality,
                            intervals=intervals,
                            notes=list(notes),
                            inversion=inversion,
                            confidence=score,
                            voicing=voicing,
                        )
        
        return best_match
    
    def _identify_seventh_chord(self, notes: List[int]) -> Optional[TriadDetection]:
        """Identify a seventh chord from notes."""
        if len(notes) < 4:
            return None
        
        pcs = sorted(set(midi_to_pitch_class(n) for n in notes))
        if len(pcs) < 4:
            return None
        
        best_match = None
        best_score = 0.0
        
        for quality, intervals in self.seventh_chords.items():
            for root_idx in range(12):
                expected = set((root_idx + i) % 12 for i in intervals)
                
                match_count = len(set(pcs) & expected)
                if match_count >= 4:
                    score = match_count / len(expected)
                    
                    if score > best_score:
                        best_score = score
                        
                        best_match = TriadDetection(
                            chord_name=f"{NOTE_NAMES[root_idx]}{quality}",
                            root=NOTE_NAMES[root_idx],
                            quality=quality,
                            intervals=intervals,
                            notes=list(notes),
                            inversion=0,
                            confidence=score,
                        )
        
        return best_match
    
    def detect_arpeggios(
        self,
        notes: List[int],
        durations: Optional[List[float]] = None,
        min_notes: int = 3,
    ) -> List[ArpeggioDetection]:
        """
        Detect arpeggio patterns in a note sequence.
        
        Args:
            notes: List of MIDI notes
            durations: Optional note durations
            min_notes: Minimum notes for arpeggio
        
        Returns:
            List of detected arpeggios
        """
        if len(notes) < min_notes:
            return []
        
        detections = []
        
        # Sliding window analysis
        for window_size in range(min_notes, min(8, len(notes) + 1)):
            for i in range(len(notes) - window_size + 1):
                window = notes[i:i + window_size]
                
                # Check if notes form a chord arpeggio
                arpeggio = self._identify_arpeggio(window, durations)
                if arpeggio:
                    detections.append(arpeggio)
        
        return detections
    
    def _identify_arpeggio(
        self,
        notes: List[int],
        durations: Optional[List[float]] = None,
    ) -> Optional[ArpeggioDetection]:
        """Identify an arpeggio pattern."""
        if len(notes) < 3:
            return None
        
        # Get pitch classes
        pcs = [midi_to_pitch_class(n) for n in notes]
        unique_pcs = set(pcs)
        
        # Check against chord templates
        for quality, intervals in {**self.triads, **self.seventh_chords}.items():
            for root_idx in range(12):
                expected = set((root_idx + i) % 12 for i in intervals)
                
                if unique_pcs <= expected and len(unique_pcs) >= 3:
                    # This is an arpeggio!
                    
                    # Determine pattern direction
                    if notes == sorted(notes):
                        pattern = "ascending"
                    elif notes == sorted(notes, reverse=True):
                        pattern = "descending"
                    elif notes[:len(notes)//2] == sorted(notes[:len(notes)//2]):
                        pattern = "ascending_then_descending"
                    else:
                        pattern = "broken"
                    
                    # Estimate speed
                    if durations:
                        avg_duration = sum(durations) / len(durations)
                        if avg_duration < 0.15:
                            speed = "fast"
                        elif avg_duration < 0.3:
                            speed = "medium"
                        else:
                            speed = "slow"
                    else:
                        speed = "unknown"
                    
                    # Calculate intervals
                    melodic_intervals = [
                        notes[i+1] - notes[i] 
                        for i in range(len(notes) - 1)
                    ]
                    
                    return ArpeggioDetection(
                        chord_base=f"{NOTE_NAMES[root_idx]}{quality}",
                        pattern=pattern,
                        notes=list(notes),
                        intervals=melodic_intervals,
                        speed_category=speed,
                        confidence=len(unique_pcs) / len(intervals),
                    )
        
        return None
    
    def analyze_intervals(self, notes: List[int]) -> IntervalAnalysis:
        """
        Analyze melodic intervals in a note sequence.
        
        Args:
            notes: List of MIDI notes
        
        Returns:
            IntervalAnalysis with interval statistics
        """
        if len(notes) < 2:
            return IntervalAnalysis(
                intervals=[],
                interval_histogram={},
                consonant_ratio=1.0,
                average_interval_size=0.0,
                melodic_contour="static",
            )
        
        # Calculate intervals
        intervals = []
        for i in range(len(notes) - 1):
            semitones = notes[i + 1] - notes[i]
            abs_semitones = abs(semitones)
            name = INTERVALS.get(abs_semitones % 12, f"{abs_semitones}_semitones")
            intervals.append((semitones, name))
        
        # Build histogram
        histogram = {}
        for _, name in intervals:
            histogram[name] = histogram.get(name, 0) + 1
        
        # Calculate consonance ratio
        consonant_count = sum(1 for s, _ in intervals if is_consonant_interval(abs(s)))
        consonant_ratio = consonant_count / len(intervals) if intervals else 1.0
        
        # Average interval size
        avg_size = sum(abs(s) for s, _ in intervals) / len(intervals) if intervals else 0.0
        
        # Melodic contour
        contour = detect_melodic_contour(notes)
        
        return IntervalAnalysis(
            intervals=intervals,
            interval_histogram=histogram,
            consonant_ratio=consonant_ratio,
            average_interval_size=avg_size,
            melodic_contour=contour,
        )
    
    def _calculate_complexity(
        self,
        pitch_classes: Set[int],
        triads: List[TriadDetection],
        sevenths: List[TriadDetection],
    ) -> float:
        """Calculate harmonic complexity score."""
        # Factors:
        # - Number of unique pitch classes
        # - Presence of chromatic notes
        # - Seventh chord usage
        # - Chord variety
        
        pc_score = min(1.0, len(pitch_classes) / 7.0)  # 7 notes = full scale
        
        # Check for chromatic content
        chromatic_pairs = sum(
            1 for pc in pitch_classes 
            if (pc + 1) % 12 in pitch_classes
        )
        chromatic_score = min(1.0, chromatic_pairs / 3.0)
        
        # Seventh chord presence
        seventh_score = min(1.0, len(sevenths) / 3.0) if sevenths else 0.0
        
        # Combine scores
        complexity = (pc_score * 0.3 + chromatic_score * 0.3 + seventh_score * 0.4)
        
        return complexity
    
    def analyze_audio(
        self,
        filepath: str,
        max_duration: Optional[float] = None,
    ) -> TheoryAnalysis:
        """
        Analyze music theory elements from audio file.
        
        Args:
            filepath: Path to audio file
            max_duration: Maximum duration to analyze
        
        Returns:
            TheoryAnalysis
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for audio analysis")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        # Load audio
        y, sr = librosa.load(str(filepath), sr=None, mono=True, duration=max_duration)
        
        # Extract chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Convert to pitch class presence
        chroma_mean = np.mean(chroma, axis=1)
        threshold = np.mean(chroma_mean) + np.std(chroma_mean) * 0.5
        
        pitch_classes = set(
            i for i in range(12) 
            if chroma_mean[i] > threshold
        )
        
        # Detect scales
        detected_scales = self.detect_scales(pitch_classes)
        primary_scale = detected_scales[0] if detected_scales else None
        
        # For audio, we can't easily detect specific triads/arpeggios
        # without pitch tracking, so return scale analysis primarily
        
        return TheoryAnalysis(
            detected_scales=detected_scales,
            primary_scale=primary_scale,
            detected_triads=[],
            detected_sevenths=[],
            detected_arpeggios=[],
            key_center=primary_scale.root if primary_scale else None,
            mode=primary_scale.scale_name if primary_scale else None,
            harmonic_complexity=len(pitch_classes) / 12.0,
        )


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def detect_scale(notes: List[int]) -> Optional[ScaleDetection]:
    """
    Detect the primary scale from MIDI notes.
    
    Args:
        notes: List of MIDI note numbers
    
    Returns:
        ScaleDetection or None
    """
    analyzer = TheoryAnalyzer()
    pitch_classes = get_pitch_class_set(notes)
    scales = analyzer.detect_scales(pitch_classes, top_n=1)
    return scales[0] if scales else None


def detect_mode(notes: List[int]) -> Optional[str]:
    """
    Detect the mode from MIDI notes.
    
    Args:
        notes: List of MIDI note numbers
    
    Returns:
        Mode name or None
    """
    scale = detect_scale(notes)
    if scale and scale.is_mode:
        return scale.full_name
    return None


def analyze_harmony(notes: List[int]) -> TheoryAnalysis:
    """
    Perform complete harmonic analysis.
    
    Args:
        notes: List of MIDI note numbers
    
    Returns:
        TheoryAnalysis
    """
    analyzer = TheoryAnalyzer()
    return analyzer.analyze_notes(notes)

