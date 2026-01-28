"""
DAW Functions Reference - Comprehensive reference for basic DAW operations.

This module provides reference implementations and documentation for all
standard DAW functions including transport, tracks, MIDI editing, mixing, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pathlib import Path


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class TransportState(Enum):
    """Transport control states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    RECORDING = "recording"
    PAUSED = "paused"
    LOOPING = "looping"


class TrackType(Enum):
    """Track types."""
    AUDIO = "audio"
    MIDI = "midi"
    INSTRUMENT = "instrument"
    AUX = "aux"
    BUS = "bus"
    OUTPUT = "output"


class QuantizeValue(Enum):
    """Quantization values."""
    WHOLE = 1
    HALF = 2
    QUARTER = 4
    EIGHTH = 8
    SIXTEENTH = 16
    THIRTY_SECOND = 32
    SIXTY_FOURTH = 64
    TRIPLET_QUARTER = 3
    TRIPLET_EIGHTH = 6
    TRIPLET_SIXTEENTH = 12
    SWING_8 = "swing_8"
    SWING_16 = "swing_16"


# =============================================================================
# TRANSPORT CONTROLS
# =============================================================================

@dataclass
class Transport:
    """Transport control reference implementation."""
    state: TransportState = TransportState.STOPPED
    position: float = 0.0  # Position in bars
    tempo: float = 120.0
    loop_start: float = 0.0
    loop_end: float = 4.0
    is_looping: bool = False
    
    def play(self) -> None:
        """Start playback."""
        self.state = TransportState.PLAYING
    
    def stop(self) -> None:
        """Stop playback and return to start."""
        self.state = TransportState.STOPPED
        self.position = 0.0
    
    def pause(self) -> None:
        """Pause playback at current position."""
        self.state = TransportState.PAUSED
    
    def record(self) -> None:
        """Start recording."""
        self.state = TransportState.RECORDING
    
    def set_loop(self, start: float, end: float) -> None:
        """Set loop region in bars."""
        self.loop_start = start
        self.loop_end = end
        self.is_looping = True
    
    def toggle_loop(self) -> None:
        """Toggle loop on/off."""
        self.is_looping = not self.is_looping
    
    def go_to_position(self, position: float) -> None:
        """Jump to position in bars."""
        self.position = position


# =============================================================================
# TRACK MANAGEMENT
# =============================================================================

@dataclass
class Track:
    """Track reference implementation."""
    name: str
    track_type: TrackType
    channel: int = 1
    volume: float = 0.0  # dB, 0 = unity
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    mute: bool = False
    solo: bool = False
    record_enable: bool = False
    input_channel: Optional[int] = None
    output_bus: Optional[str] = None
    instrument: Optional[int] = None  # GM program number
    color: str = "#667eea"
    
    # Effects
    inserts: List[str] = field(default_factory=list)  # Effect names/IDs
    sends: Dict[str, float] = field(default_factory=dict)  # Bus name -> send level
    
    def set_volume(self, db: float) -> None:
        """Set track volume in dB."""
        self.volume = max(-60.0, min(12.0, db))
    
    def set_pan(self, pan: float) -> None:
        """Set pan position (-1.0 to 1.0)."""
        self.pan = max(-1.0, min(1.0, pan))
    
    def toggle_mute(self) -> None:
        """Toggle mute."""
        self.mute = not self.mute
    
    def toggle_solo(self) -> None:
        """Toggle solo."""
        self.solo = not self.solo
    
    def add_insert(self, effect_name: str, position: int = -1) -> None:
        """Add insert effect."""
        if position == -1:
            self.inserts.append(effect_name)
        else:
            self.inserts.insert(position, effect_name)
    
    def remove_insert(self, effect_name: str) -> None:
        """Remove insert effect."""
        if effect_name in self.inserts:
            self.inserts.remove(effect_name)
    
    def set_send(self, bus_name: str, level: float) -> None:
        """Set send level to bus."""
        self.sends[bus_name] = max(0.0, min(1.0, level))


# =============================================================================
# MIDI EDITING
# =============================================================================

@dataclass
class MIDINote:
    """MIDI note reference."""
    pitch: int  # 0-127
    velocity: int  # 0-127
    start_time: float  # Bars
    duration: float  # Bars
    channel: int = 0
    
    def transpose(self, semitones: int) -> None:
        """Transpose note."""
        self.pitch = max(0, min(127, self.pitch + semitones))
    
    def set_velocity(self, velocity: int) -> None:
        """Set velocity."""
        self.velocity = max(0, min(127, velocity))


@dataclass
class MIDIEditor:
    """MIDI editing functions reference."""
    
    @staticmethod
    def quantize(
        notes: List[MIDINote],
        grid: QuantizeValue,
        strength: float = 1.0,
    ) -> List[MIDINote]:
        """
        Quantize notes to grid.
        
        Args:
            notes: List of MIDI notes
            grid: Quantization grid
            strength: Quantization strength (0.0-1.0)
        
        Returns:
            Quantized notes
        """
        # Reference implementation
        if isinstance(grid, QuantizeValue) and isinstance(grid.value, int):
            grid_value = 1.0 / grid.value
        else:
            grid_value = 1.0 / 16  # Default to 16th notes
        
        quantized = []
        for note in notes:
            new_note = MIDINote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                duration=note.duration,
                channel=note.channel,
            )
            
            # Quantize start time
            original = new_note.start_time
            quantized_time = round(original / grid_value) * grid_value
            new_note.start_time = original + (quantized_time - original) * strength
            
            quantized.append(new_note)
        
        return quantized
    
    @staticmethod
    def humanize(
        notes: List[MIDINote],
        timing_variance: float = 0.01,
        velocity_variance: float = 5,
    ) -> List[MIDINote]:
        """
        Humanize notes (add slight randomness).
        
        Args:
            notes: List of MIDI notes
            timing_variance: Timing variance in bars
            velocity_variance: Velocity variance (0-127)
        
        Returns:
            Humanized notes
        """
        import random
        
        humanized = []
        for note in notes:
            new_note = MIDINote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                duration=note.duration,
                channel=note.channel,
            )
            
            # Add timing variance
            new_note.start_time += random.uniform(-timing_variance, timing_variance)
            
            # Add velocity variance
            vel_change = random.randint(-int(velocity_variance), int(velocity_variance))
            new_note.velocity = max(1, min(127, new_note.velocity + vel_change))
            
            humanized.append(new_note)
        
        return humanized
    
    @staticmethod
    def transpose_notes(notes: List[MIDINote], semitones: int) -> List[MIDINote]:
        """Transpose all notes."""
        transposed = []
        for note in notes:
            new_note = MIDINote(
                pitch=max(0, min(127, note.pitch + semitones)),
                velocity=note.velocity,
                start_time=note.start_time,
                duration=note.duration,
                channel=note.channel,
            )
            transposed.append(new_note)
        return transposed
    
    @staticmethod
    def scale_velocity(notes: List[MIDINote], factor: float) -> List[MIDINote]:
        """Scale all velocities by factor."""
        scaled = []
        for note in notes:
            new_note = MIDINote(
                pitch=note.pitch,
                velocity=max(1, min(127, int(note.velocity * factor))),
                start_time=note.start_time,
                duration=note.duration,
                channel=note.channel,
            )
            scaled.append(new_note)
        return scaled


# =============================================================================
# MIXING FUNCTIONS
# =============================================================================

@dataclass
class MixerChannel:
    """Mixer channel reference."""
    track_name: str
    volume: float = 0.0  # dB
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    
    # EQ (simplified - 3-band)
    eq_low_gain: float = 0.0  # dB
    eq_mid_gain: float = 0.0
    eq_high_gain: float = 0.0
    
    # Compression (simplified)
    compressor_threshold: float = -12.0  # dB
    compressor_ratio: float = 4.0
    compressor_attack: float = 0.003  # seconds
    compressor_release: float = 0.1  # seconds
    
    # Reverb send
    reverb_send: float = 0.0  # 0.0-1.0


# =============================================================================
# PROJECT MANAGEMENT
# =============================================================================

@dataclass
class DAWProject:
    """DAW project reference structure."""
    name: str
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    key: str = "C"
    mode: str = "major"
    
    tracks: List[Track] = field(default_factory=list)
    transport: Transport = field(default_factory=Transport)
    
    # Timeline markers
    markers: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_track(self, track: Track) -> None:
        """Add track to project."""
        self.tracks.append(track)
    
    def remove_track(self, track_name: str) -> None:
        """Remove track by name."""
        self.tracks = [t for t in self.tracks if t.name != track_name]
    
    def get_track(self, track_name: str) -> Optional[Track]:
        """Get track by name."""
        for track in self.tracks:
            if track.name == track_name:
                return track
        return None
    
    def add_marker(self, name: str, position: float) -> None:
        """Add timeline marker."""
        self.markers.append({
            "name": name,
            "position": position,  # Bars
        })


# =============================================================================
# REFERENCE DOCUMENTATION
# =============================================================================

DAW_FUNCTIONS_REFERENCE = {
    "transport": {
        "play": "Start playback from current position",
        "stop": "Stop playback and return to start",
        "pause": "Pause playback at current position",
        "record": "Start recording",
        "loop": "Enable/disable loop playback",
        "go_to": "Jump to specific position",
        "set_tempo": "Change project tempo",
    },
    "tracks": {
        "create": "Create new track (audio/MIDI/instrument)",
        "delete": "Delete track",
        "rename": "Rename track",
        "mute": "Mute/unmute track",
        "solo": "Solo/unsolo track",
        "volume": "Set track volume (dB)",
        "pan": "Set track pan (-1.0 to 1.0)",
        "record_enable": "Enable/disable recording on track",
    },
    "midi_editing": {
        "quantize": "Quantize notes to grid",
        "humanize": "Add human feel to notes",
        "transpose": "Transpose notes up/down",
        "scale_velocity": "Scale note velocities",
        "delete": "Delete selected notes",
        "copy": "Copy selected notes",
        "paste": "Paste notes",
        "duplicate": "Duplicate selected notes",
    },
    "mixing": {
        "eq": "Equalization (low/mid/high)",
        "compression": "Dynamic compression",
        "reverb": "Reverb send/return",
        "delay": "Delay effect",
        "automation": "Parameter automation",
        "bus": "Route to bus/aux",
        "send": "Send to effect bus",
    },
    "project": {
        "save": "Save project",
        "load": "Load project",
        "export": "Export audio/MIDI",
        "import": "Import audio/MIDI",
        "undo": "Undo last action",
        "redo": "Redo last action",
        "markers": "Add/remove timeline markers",
    },
}


def get_daw_function_reference() -> Dict[str, Dict[str, str]]:
    """Get comprehensive DAW functions reference."""
    return DAW_FUNCTIONS_REFERENCE


def get_transport_reference() -> Dict[str, str]:
    """Get transport control reference."""
    return DAW_FUNCTIONS_REFERENCE["transport"]


def get_track_reference() -> Dict[str, str]:
    """Get track management reference."""
    return DAW_FUNCTIONS_REFERENCE["tracks"]


def get_midi_editing_reference() -> Dict[str, str]:
    """Get MIDI editing reference."""
    return DAW_FUNCTIONS_REFERENCE["midi_editing"]


def get_mixing_reference() -> Dict[str, str]:
    """Get mixing reference."""
    return DAW_FUNCTIONS_REFERENCE["mixing"]


def get_project_reference() -> Dict[str, str]:
    """Get project management reference."""
    return DAW_FUNCTIONS_REFERENCE["project"]

