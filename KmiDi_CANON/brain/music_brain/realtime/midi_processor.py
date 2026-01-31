"""
Real-time MIDI Processor - Core engine for live MIDI processing.

Handles MIDI input/output ports, message routing, and real-time analysis.
"""

import time
import threading
from typing import Callable, Optional, List, Dict, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

from music_brain.structure.chord import detect_chord_from_notes, Chord
from music_brain.groove.extractor import GrooveTemplate


# =================================================================
# CALLBACK TYPES
# =================================================================

ChordDetectionCallback = Callable[[Optional[Chord], List[int]], None]
GrooveAnalysisCallback = Callable[[Dict], None]
MidiTransformCallback = Callable[[mido.Message], Optional[mido.Message]]


# =================================================================
# CONFIGURATION
# =================================================================

@dataclass
class MidiProcessorConfig:
    """Configuration for real-time MIDI processing."""
    
    # MIDI ports
    input_port_name: Optional[str] = None  # None = auto-detect first available
    output_port_name: Optional[str] = None  # None = no output
    
    # Chord detection
    enable_chord_detection: bool = True
    chord_window_ms: float = 200.0  # Time window for grouping notes into chords
    chord_min_notes: int = 2  # Minimum notes to detect a chord
    
    # Groove analysis
    enable_groove_analysis: bool = False
    groove_window_beats: float = 4.0  # Window size for groove analysis
    
    # MIDI transformation
    enable_transformation: bool = False
    transform_callback: Optional[MidiTransformCallback] = None
    
    # Performance
    buffer_size: int = 100  # Number of messages to buffer
    processing_thread: bool = True  # Use separate thread for processing


# =================================================================
# REAL-TIME MIDI PROCESSOR
# =================================================================

class RealtimeMidiProcessor:
    """
    Real-time MIDI processor for live input/output and analysis.
    
    Features:
    - MIDI input capture from hardware/software ports
    - Real-time chord detection
    - Real-time groove analysis
    - MIDI message transformation
    - MIDI output routing
    """
    
    def __init__(self, config: Optional[MidiProcessorConfig] = None):
        """
        Initialize real-time MIDI processor.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        if not MIDO_AVAILABLE:
            raise ImportError(
                "mido package required for real-time MIDI. "
                "Install with: pip install mido"
            )
        
        self.config = config or MidiProcessorConfig()
        self.running = False
        self.input_port: Optional[mido.ports.BaseInput] = None
        self.output_port: Optional[mido.ports.BaseOutput] = None
        
        # Processing state
        self.active_notes: Set[int] = set()  # Currently held notes
        self.note_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.chord_history: List[Chord] = []
        self.last_chord_time: float = 0.0
        
        # Callbacks
        self.chord_callback: Optional[ChordDetectionCallback] = None
        self.groove_callback: Optional[GrooveAnalysisCallback] = None
        self.transform_callback: Optional[MidiTransformCallback] = self.config.transform_callback
        
        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Groove analysis state
        self.groove_notes: List[Dict] = []  # List of {note, velocity, time_ms}
        self.tempo_bpm: float = 120.0  # Estimated tempo
    
    def list_input_ports(self) -> List[str]:
        """List available MIDI input ports."""
        if not MIDO_AVAILABLE:
            return []
        return mido.get_input_names()
    
    def list_output_ports(self) -> List[str]:
        """List available MIDI output ports."""
        if not MIDO_AVAILABLE:
            return []
        return mido.get_output_names()
    
    def set_chord_callback(self, callback: ChordDetectionCallback):
        """Set callback for chord detection events."""
        self.chord_callback = callback
    
    def set_groove_callback(self, callback: GrooveAnalysisCallback):
        """Set callback for groove analysis updates."""
        self.groove_callback = callback
    
    def set_transform_callback(self, callback: MidiTransformCallback):
        """Set callback for MIDI message transformation."""
        self.transform_callback = callback
    
    def open_ports(self) -> bool:
        """
        Open MIDI input and output ports.
        
        Returns:
            True if ports opened successfully
        """
        try:
            # Open input port
            input_names = self.list_input_ports()
            if not input_names:
                raise RuntimeError("No MIDI input ports available")
            
            if self.config.input_port_name:
                if self.config.input_port_name not in input_names:
                    raise RuntimeError(
                        f"Input port '{self.config.input_port_name}' not found. "
                        f"Available: {input_names}"
                    )
                port_name = self.config.input_port_name
            else:
                port_name = input_names[0]  # Use first available
            
            self.input_port = mido.open_input(port_name)
            print(f"Opened MIDI input: {port_name}")
            
            # Open output port (optional)
            if self.config.output_port_name:
                output_names = self.list_output_ports()
                if self.config.output_port_name not in output_names:
                    print(f"Warning: Output port '{self.config.output_port_name}' not found")
                else:
                    self.output_port = mido.open_output(self.config.output_port_name)
                    print(f"Opened MIDI output: {self.config.output_port_name}")
            
            return True
        
        except Exception as e:
            print(f"Error opening MIDI ports: {e}")
            return False
    
    def close_ports(self):
        """Close MIDI input and output ports."""
        if self.input_port:
            self.input_port.close()
            self.input_port = None
        if self.output_port:
            self.output_port.close()
            self.output_port = None
    
    def _process_message(self, msg: mido.Message):
        """Process a single MIDI message."""
        current_time = time.time()
        
        # Handle note events
        if msg.type == 'note_on' and msg.velocity > 0:
            self.active_notes.add(msg.note)
            self.note_buffer.append({
                'note': msg.note,
                'velocity': msg.velocity,
                'time': current_time,
                'type': 'on'
            })
            
            # Add to groove analysis buffer
            if self.config.enable_groove_analysis:
                self.groove_notes.append({
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'time_ms': current_time * 1000,
                })
        
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            self.active_notes.discard(msg.note)
            self.note_buffer.append({
                'note': msg.note,
                'velocity': 0,
                'time': current_time,
                'type': 'off'
            })
        
        # Chord detection
        if self.config.enable_chord_detection and self.chord_callback:
            self._detect_chord(current_time)
        
        # Groove analysis
        if self.config.enable_groove_analysis and self.groove_callback:
            self._analyze_groove()
        
        # MIDI transformation
        if self.config.enable_transformation and self.transform_callback:
            transformed = self.transform_callback(msg)
            if transformed:
                msg = transformed
        
        # Output transformed/original message
        if self.output_port:
            try:
                self.output_port.send(msg)
            except Exception as e:
                print(f"Error sending MIDI message: {e}")
    
    def _detect_chord(self, current_time: float):
        """Detect chord from active notes within time window."""
        # Get notes that are still active within the time window
        window_start = current_time - (self.config.chord_window_ms / 1000.0)
        
        # Collect notes from buffer within window
        window_notes = []
        for note_event in self.note_buffer:
            if note_event['time'] >= window_start and note_event['type'] == 'on':
                window_notes.append(note_event['note'])
        
        # Also include currently active notes
        window_notes.extend(self.active_notes)
        
        # Remove duplicates
        window_notes = list(set(window_notes))
        
        if len(window_notes) >= self.config.chord_min_notes:
            chord = detect_chord_from_notes(window_notes)
            
            # Only trigger callback if chord changed or enough time passed
            time_since_last = current_time - self.last_chord_time
            chord_changed = (
                not self.chord_history or
                (chord and self.chord_history[-1].name != chord.name)
            )
            
            if chord_changed or time_since_last > 0.5:  # Update at least every 0.5s
                if self.chord_callback:
                    self.chord_callback(chord, window_notes)
                
                if chord:
                    self.chord_history.append(chord)
                    if len(self.chord_history) > 10:  # Keep last 10 chords
                        self.chord_history.pop(0)
                
                self.last_chord_time = current_time
    
    def _analyze_groove(self):
        """Analyze groove from recent note events."""
        if not self.groove_notes:
            return
        
        # Calculate window size in milliseconds
        window_ms = (self.config.groove_window_beats * 60.0 / self.tempo_bpm) * 1000.0
        current_time_ms = time.time() * 1000.0
        
        # Keep only notes within window
        self.groove_notes = [
            n for n in self.groove_notes
            if (current_time_ms - n['time_ms']) < window_ms
        ]
        
        if len(self.groove_notes) < 4:  # Need at least 4 notes for analysis
            return
        
        # Calculate basic groove metrics
        if len(self.groove_notes) >= 2:
            # Timing deviations
            intervals = []
            for i in range(1, len(self.groove_notes)):
                interval = self.groove_notes[i]['time_ms'] - self.groove_notes[i-1]['time_ms']
                intervals.append(interval)
            
            if intervals:
                mean_interval = sum(intervals) / len(intervals)
                deviations = [abs(i - mean_interval) for i in intervals]
                mean_deviation = sum(deviations) / len(deviations) if deviations else 0.0
                
                # Velocity statistics
                velocities = [n['velocity'] for n in self.groove_notes]
                vel_mean = sum(velocities) / len(velocities)
                vel_std = (
                    sum((v - vel_mean) ** 2 for v in velocities) / len(velocities)
                ) ** 0.5
                
                groove_data = {
                    'mean_deviation_ms': mean_deviation,
                    'velocity_mean': vel_mean,
                    'velocity_std': vel_std,
                    'note_count': len(self.groove_notes),
                    'tempo_bpm': self.tempo_bpm,
                }
                
                if self.groove_callback:
                    self.groove_callback(groove_data)
    
    def _processing_loop(self):
        """Main processing loop (runs in separate thread if enabled)."""
        if not self.input_port:
            return
        
        try:
            for msg in self.input_port:
                if self._stop_event.is_set():
                    break
                
                self._process_message(msg)
        
        except Exception as e:
            print(f"Error in MIDI processing loop: {e}")
    
    def start(self) -> bool:
        """
        Start real-time MIDI processing.
        
        Returns:
            True if started successfully
        """
        if self.running:
            print("Processor already running")
            return False
        
        if not self.open_ports():
            return False
        
        self.running = True
        self._stop_event.clear()
        
        if self.config.processing_thread:
            # Start processing in separate thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            print("Real-time MIDI processing started (threaded)")
        else:
            # Process in current thread (blocking)
            print("Real-time MIDI processing started (blocking)")
            # Note: This will block - consider using threading for production
            print("Warning: Blocking mode - use processing_thread=True for non-blocking")
        
        return True
    
    def stop(self):
        """Stop real-time MIDI processing."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        self.close_ports()
        print("Real-time MIDI processing stopped")
    
    def get_active_notes(self) -> List[int]:
        """Get list of currently active (held) notes."""
        return list(self.active_notes)
    
    def get_chord_history(self) -> List[Chord]:
        """Get history of detected chords."""
        return self.chord_history.copy()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

