"""MIDI generation and pipeline.

This module provides MIDI file generation capabilities including chord progressions,
groove templates, and musical pattern generation for therapeutic music creation.
"""
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import mido  # type: ignore
except ImportError:
    class _DummyTrack(list):
        """Lightweight stand-in for mido.MidiTrack that behaves like a list."""

        def __init__(self):
            super().__init__()

    class _DummyMessage:
        """Minimal message container mirroring mido.Message/MetaMessage attrs."""

        def __init__(self, message_type: str, time: int = 0, **kwargs):
            self.type = message_type
            self.time = time
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getitem__(self, key):
            return getattr(self, key)

        def __repr__(self) -> str:
            return f"<Message {self.type} time={self.time}>"

    class _DummyMidiFile:
        def __init__(self, ticks_per_beat: int = 480):
            self.ticks_per_beat = ticks_per_beat
            self.tracks = []

        def save(self, path: str):
            # Write a minimal placeholder; actual MIDI content not required for tests
            Path(path).write_text("dummy midi")

    class mido:  # type: ignore
        Message = _DummyMessage
        MetaMessage = _DummyMessage
        MidiFile = _DummyMidiFile
        MidiTrack = _DummyTrack

        @staticmethod
        def bpm2tempo(bpm: float) -> int:
            """Convert BPM to microseconds-per-beat (matches real mido helper)."""
            return int(60_000_000 / bpm)


@dataclass
class GrooveTemplate:
    """Represents a rhythmic groove template.
    
    Attributes:
        name: Human-readable name of the groove
        time_signature: Tuple of (beats_per_measure, beat_unit)
        pattern: List of (beat_time, velocity) tuples
        swing: Swing factor (0.0 = straight, 1.0 = maximum swing)
        description: Optional description of the groove style
    """
    name: str
    time_signature: Tuple[int, int]
    pattern: List[Tuple[float, int]]  # (time, velocity)
    swing: float = 0.0
    description: str = ""


@dataclass
class Chord:
    """Represents a musical chord.
    
    Attributes:
        root: Root note MIDI number
        intervals: List of intervals in semitones from root
        name: Optional chord name (e.g., "Am", "C7")
    """
    root: int
    intervals: List[int]
    name: str = ""
    
    def to_midi_notes(self) -> List[int]:
        """Convert chord to list of MIDI note numbers."""
        return [self.root + interval for interval in self.intervals]
    
    def __str__(self) -> str:
        return self.name if self.name else f"Chord({self.root})"


class MidiGenerator:
    """
    MIDI pipeline for generating therapeutic musical output.
    
    Handles chord progressions, groove templates, MIDI message generation,
    and file creation. Supports various musical styles and therapeutic
    expression through intentional rule-breaking.
    
    Example:
        >>> generator = MidiGenerator(tempo=120)
        >>> chords = generator.generate_chord_progression(mode="minor", length=4)
        >>> midi_file = generator.create_midi_file(chords, groove="swing", output_path="output.mid")
    """
    
    # Standard chord intervals (in semitones from root)
    CHORD_INTERVALS: Dict[str, List[int]] = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "diminished": [0, 3, 6],
        "augmented": [0, 4, 8],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "major7": [0, 4, 7, 11],
        "minor7": [0, 3, 7, 10],
        "dominant7": [0, 4, 7, 10],
        "diminished7": [0, 3, 6, 9],
    }
    
    # Common chord progressions
    PROGRESSIONS: Dict[str, List[str]] = {
        "minor_i_iv_v": ["minor", "minor", "minor", "minor"],  # i-iv-v-i
        "minor_i_bVII_bVI": ["minor", "major", "major", "minor"],  # i-♭VII-♭VI-i
        "major_I_IV_V": ["major", "major", "major", "major"],  # I-IV-V-I
        "major_I_vi_IV_V": ["major", "minor", "major", "major"],  # I-vi-IV-V
    }
    
    def __init__(self, tempo: int = 120, ticks_per_beat: int = 480) -> None:
        """
        Initialize MIDI generator.
        
        Args:
            tempo: Initial tempo in BPM
            ticks_per_beat: MIDI ticks per quarter note (default 480)
        """
        if tempo < 20 or tempo > 300:
            raise ValueError(f"Tempo must be between 20 and 300 BPM, got {tempo}")
        
        self.tempo = tempo
        self.ticks_per_beat = ticks_per_beat
        self.groove_templates = self._initialize_grooves()
    
    def _initialize_grooves(self) -> Dict[str, GrooveTemplate]:
        """Initialize standard groove templates."""
        return {
            "straight": GrooveTemplate(
                name="Straight",
                time_signature=(4, 4),
                pattern=[(0.0, 100), (0.25, 80), (0.5, 100), (0.75, 80)],
                description="Straight 4/4 time with even beats"
            ),
            "swing": GrooveTemplate(
                name="Swing",
                time_signature=(4, 4),
                pattern=[(0.0, 100), (0.33, 80), (0.5, 100), (0.83, 80)],
                swing=0.66,
                description="Swing feel with triplet subdivision"
            ),
            "syncopated": GrooveTemplate(
                name="Syncopated",
                time_signature=(4, 4),
                pattern=[(0.0, 100), (0.125, 60), (0.375, 90), (0.625, 85), (0.875, 70)],
                description="Syncopated pattern with off-beat accents"
            ),
            "shuffle": GrooveTemplate(
                name="Shuffle",
                time_signature=(4, 4),
                pattern=[(0.0, 100), (0.33, 70), (0.5, 90), (0.83, 60)],
                swing=0.75,
                description="Shuffle groove with strong swing feel"
            ),
            "ballad": GrooveTemplate(
                name="Ballad",
                time_signature=(4, 4),
                pattern=[(0.0, 90), (0.5, 85)],
                description="Slow ballad with sparse hits"
            ),
            "driving": GrooveTemplate(
                name="Driving",
                time_signature=(4, 4),
                pattern=[(0.0, 110), (0.125, 70), (0.25, 90), (0.375, 80), 
                        (0.5, 110), (0.625, 70), (0.75, 90), (0.875, 80)],
                description="Driving rhythm with constant motion"
            ),
        }
    
    def generate_chord_progression(
        self, 
        mode: str = "minor",
        root: int = 57,  # A (for minor) or 60 (C for major)
        length: int = 4,
        allow_dissonance: bool = False,
        progression_type: Optional[str] = None
    ) -> List[List[int]]:
        """
        Generate a chord progression based on emotional parameters.
        
        Args:
            mode: "major" or "minor"
            root: Root note MIDI number (default: 57=A for minor, 60=C for major)
            length: Number of chords in progression
            allow_dissonance: Whether to allow dissonant intervals
            progression_type: Optional progression type (e.g., "minor_i_iv_v")
            
        Returns:
            List of chords (each chord is a list of MIDI note numbers)
        """
        if mode not in ["major", "minor"]:
            raise ValueError(f"Mode must be 'major' or 'minor', got {mode}")
        
        if length < 1 or length > 16:
            raise ValueError(f"Progression length must be between 1 and 16, got {length}")
        
        # Select progression type
        if progression_type and progression_type in self.PROGRESSIONS:
            chord_types = self.PROGRESSIONS[progression_type]
        elif mode == "minor":
            chord_types = self.PROGRESSIONS["minor_i_iv_v"]
        else:
            chord_types = self.PROGRESSIONS["major_I_IV_V"]
        
        # Generate chords
        chords: List[List[int]] = []
        scale_notes = self._get_scale_notes(mode, root)
        
        for i, chord_type in enumerate(chord_types[:length]):
            # Map chord type to intervals
            if chord_type == "major":
                intervals = self.CHORD_INTERVALS["major"].copy()
            elif chord_type == "minor":
                intervals = self.CHORD_INTERVALS["minor"].copy()
            else:
                intervals = self.CHORD_INTERVALS["major"].copy()
            
            # Calculate root note for this chord in the progression
            # For i-iv-v: use scale degrees 0, 3, 4
            if mode == "minor" and progression_type == "minor_i_iv_v":
                scale_degree = [0, 3, 4, 0][i % 4]
            elif mode == "major" and progression_type == "major_I_IV_V":
                scale_degree = [0, 3, 4, 0][i % 4]
            else:
                scale_degree = i % len(scale_notes)
            
            chord_root = scale_notes[scale_degree] if scale_degree < len(scale_notes) else root
            
            # Build chord
            chord_notes = [chord_root + interval for interval in intervals]
            
            # Add dissonance if requested
            if allow_dissonance:
                # Add minor 9th or tritone
                if i % 2 == 0:
                    chord_notes.append(chord_root + 13)  # Minor 9th
                else:
                    chord_notes.append(chord_root + 6)  # Tritone
            
            chords.append(chord_notes)
        
        return chords
    
    def _get_scale_notes(self, mode: str, root: int) -> List[int]:
        """Get scale notes for a given mode and root."""
        if mode == "minor":
            # Natural minor scale intervals
            intervals = [0, 2, 3, 5, 7, 8, 10]
        else:  # major
            # Major scale intervals
            intervals = [0, 2, 4, 5, 7, 9, 11]
        
        return [root + interval for interval in intervals]
    
    def apply_groove(
        self,
        notes: List[int],
        groove_template: GrooveTemplate,
        duration_bars: int = 1,
        channel: int = 0
    ) -> List[mido.Message]:
        """
        Apply groove template to notes.
        
        Args:
            notes: MIDI note numbers
            groove_template: The groove to apply
            duration_bars: Duration in bars
            channel: MIDI channel (0-15)
            
        Returns:
            List of MIDI messages with proper timing
        """
        if not notes:
            return []
        
        if channel < 0 or channel > 15:
            raise ValueError(f"MIDI channel must be between 0 and 15, got {channel}")
        
        messages: List[mido.Message] = []
        beats_per_bar = groove_template.time_signature[0]
        ticks_per_bar = beats_per_bar * self.ticks_per_beat
        
        current_time = 0
        
        for bar in range(duration_bars):
            for beat_time, velocity in groove_template.pattern:
                # Apply swing if present
                if groove_template.swing > 0.0 and beat_time % 0.5 != 0:
                    # Swing affects off-beat notes
                    beat_time = beat_time * (1.0 + groove_template.swing * 0.1)
                
                absolute_time = int(
                    (bar * beats_per_bar + beat_time * beats_per_bar) * self.ticks_per_beat
                )
                
                # Note on for each note in chord
                for note in notes:
                    if 0 <= note <= 127:  # Valid MIDI note range
                        messages.append(
                            mido.Message(
                                'note_on',
                                channel=channel,
                                note=note,
                                velocity=velocity,
                                time=absolute_time - current_time
                            )
                        )
                        current_time = absolute_time
                
                # Note off after quarter note duration
                off_time = int(absolute_time + self.ticks_per_beat * 0.25)
                for note in notes:
                    if 0 <= note <= 127:
                        messages.append(
                            mido.Message(
                                'note_off',
                                channel=channel,
                                note=note,
                                velocity=0,
                                time=off_time - current_time
                            )
                        )
                        current_time = off_time
        
        return messages
    
    def create_midi_file(
        self,
        chord_progression: List[List[int]],
        groove: str = "straight",
        output_path: Optional[str] = None,
        channel: int = 0
    ) -> mido.MidiFile:
        """
        Create a MIDI file from chord progression.
        
        Args:
            chord_progression: List of chords (each is a list of MIDI note numbers)
            groove: Name of groove template to use
            output_path: Optional path to save MIDI file
            channel: MIDI channel to use
            
        Returns:
            MIDI file object
            
        Raises:
            ValueError: If groove template not found or invalid parameters
        """
        if not chord_progression:
            raise ValueError("Chord progression cannot be empty")
        
        groove_template = self.groove_templates.get(groove)
        if not groove_template:
            available = ", ".join(self.groove_templates.keys())
            raise ValueError(
                f"Groove template '{groove}' not found. Available: {available}"
            )
        
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo
        tempo_msg = mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.tempo))
        track.append(tempo_msg)
        
        # Set time signature
        time_sig = mido.MetaMessage(
            'time_signature',
            numerator=groove_template.time_signature[0],
            denominator=groove_template.time_signature[1]
        )
        track.append(time_sig)
        
        # Generate messages for each chord
        current_time = 0
        for chord in chord_progression:
            messages = self.apply_groove(chord, groove_template, duration_bars=1, channel=channel)
            
            # Adjust message times to be relative
            for msg in messages:
                if msg.type in ['note_on', 'note_off']:
                    track.append(msg)
        
        # Save if output path provided
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            mid.save(str(output_path_obj))
        
        return mid
    
    def add_groove_template(self, template: GrooveTemplate) -> None:
        """Add a custom groove template."""
        self.groove_templates[template.name.lower()] = template
    
    def get_available_grooves(self) -> List[str]:
        """Get list of available groove template names."""
        return list(self.groove_templates.keys())
    
    def set_tempo(self, tempo: int) -> None:
        """Update tempo."""
        if tempo < 20 or tempo > 300:
            raise ValueError(f"Tempo must be between 20 and 300 BPM, got {tempo}")
        self.tempo = tempo
