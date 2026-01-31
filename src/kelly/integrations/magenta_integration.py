"""
Kelly Project - Magenta Integration Layer

Bridges kellymidicompanion with Google Magenta:
- NoteSequence <-> mido conversion
- VAD emotion vector -> MusicVAE latent space mapping
- GrooVAE emotion-parameterized humanization
- Real-time MIDI stream processing

Dependencies:
    pip install note-seq magenta mido numpy

Note: Magenta requires Python 3.7-3.9 for full compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict
import numpy as np
import time

# Magenta imports (guarded for environments without magenta)
try:
    import note_seq
    from note_seq import midi_io, sequences_lib
    from note_seq.protobuf import music_pb2
    MAGENTA_AVAILABLE = True
except ImportError:
    MAGENTA_AVAILABLE = False
    print("Warning: note-seq not installed. Run: pip install note-seq")

# mido for real-time MIDI
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


# =============================================================================
# VAD State (mirrors C++ VADState for interop)
# =============================================================================

@dataclass
class VADState:
    """Valence-Arousal-Dominance emotional state."""
    valence: float = 0.0      # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.5      # 0.0 (calm) to 1.0 (excited)
    dominance: float = 0.5    # 0.0 (submissive) to 1.0 (dominant)
    timestamp: float = 0.0
    
    def clamp(self) -> 'VADState':
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        return self
    
    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance])


# =============================================================================
# NoteSequence <-> mido Conversion
# =============================================================================

class NoteSequenceConverter:
    """
    Bidirectional conversion between Magenta NoteSequence and mido.
    
    NoteSequence uses absolute time in seconds with explicit durations.
    mido uses relative delta times in ticks.
    """
    
    def __init__(self, ticks_per_beat: int = 480, default_tempo: int = 500000):
        self.ticks_per_beat = ticks_per_beat
        self.default_tempo = default_tempo  # microseconds per beat (120 BPM)
    
    def midi_file_to_notesequence(self, filepath: str) -> 'music_pb2.NoteSequence':
        """Load MIDI file to NoteSequence."""
        if not MAGENTA_AVAILABLE:
            raise RuntimeError("note-seq not available")
        return midi_io.midi_file_to_note_sequence(filepath)
    
    def notesequence_to_midi_file(self, sequence: 'music_pb2.NoteSequence', 
                                   filepath: str) -> None:
        """Save NoteSequence to MIDI file."""
        if not MAGENTA_AVAILABLE:
            raise RuntimeError("note-seq not available")
        midi_io.note_sequence_to_midi_file(sequence, filepath)
    
    def mido_messages_to_notesequence(
        self, 
        messages: List[mido.Message],
        qpm: float = 120.0
    ) -> 'music_pb2.NoteSequence':
        """
        Convert list of mido messages to NoteSequence.
        
        Tracks active notes and computes durations from note_on/note_off pairs.
        """
        if not MAGENTA_AVAILABLE:
            raise RuntimeError("note-seq not available")
        
        sequence = music_pb2.NoteSequence()
        sequence.tempos.add(qpm=qpm, time=0.0)
        sequence.ticks_per_quarter = self.ticks_per_beat
        
        # Track active notes: (pitch, channel) -> (start_time, velocity)
        active_notes: Dict[Tuple[int, int], Tuple[float, int]] = {}
        current_time = 0.0
        tempo_us = 60_000_000 / qpm  # microseconds per beat
        
        for msg in messages:
            # Accumulate time
            if hasattr(msg, 'time'):
                delta_seconds = mido.tick2second(
                    msg.time, self.ticks_per_beat, int(tempo_us)
                )
                current_time += delta_seconds
            
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.note, msg.channel)
                active_notes[key] = (current_time, msg.velocity)
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active_notes:
                    start_time, velocity = active_notes.pop(key)
                    note = sequence.notes.add()
                    note.pitch = msg.note
                    note.velocity = velocity
                    note.start_time = start_time
                    note.end_time = current_time
                    note.instrument = msg.channel
                    note.program = 0  # Default piano
                    
            elif msg.type == 'set_tempo':
                tempo_us = msg.tempo
                sequence.tempos.add(qpm=60_000_000 / tempo_us, time=current_time)
        
        # Close any remaining active notes
        for key, (start_time, velocity) in active_notes.items():
            pitch, channel = key
            note = sequence.notes.add()
            note.pitch = pitch
            note.velocity = velocity
            note.start_time = start_time
            note.end_time = current_time + 0.1  # Small duration for unclosed notes
            note.instrument = channel
        
        sequence.total_time = current_time
        return sequence
    
    def notesequence_to_mido_messages(
        self, 
        sequence: 'music_pb2.NoteSequence'
    ) -> List[mido.Message]:
        """
        Convert NoteSequence to list of mido messages with delta times.
        """
        if not MIDO_AVAILABLE:
            raise RuntimeError("mido not available")
        
        # Get tempo
        qpm = 120.0
        if sequence.tempos:
            qpm = sequence.tempos[0].qpm
        tempo_us = int(60_000_000 / qpm)
        
        # Build event list: (time, type, pitch, velocity, channel)
        events = []
        for note in sequence.notes:
            events.append((note.start_time, 'note_on', note.pitch, note.velocity, note.instrument))
            events.append((note.end_time, 'note_off', note.pitch, 0, note.instrument))
        
        # Sort by time
        events.sort(key=lambda x: x[0])
        
        # Convert to mido messages with delta times
        messages = []
        prev_time = 0.0
        
        for time_sec, msg_type, pitch, velocity, channel in events:
            delta_sec = time_sec - prev_time
            delta_ticks = mido.second2tick(delta_sec, self.ticks_per_beat, tempo_us)
            
            messages.append(mido.Message(
                msg_type,
                note=pitch,
                velocity=velocity,
                channel=channel,
                time=int(delta_ticks)
            ))
            prev_time = time_sec
        
        return messages


# =============================================================================
# VAD -> MusicVAE Latent Space Mapping
# =============================================================================

@dataclass
class EmotionAttributeVectors:
    """
    Pre-computed attribute vectors for emotion-to-latent mapping.
    
    Computed offline from EMOPIA dataset by averaging latent codes
    of high/low valence/arousal/dominance samples.
    
    z_target = z_base + α_V*(z_HV - z_LV) + α_A*(z_HA - z_LA) + α_D*(z_HD - z_LD)
    """
    latent_dim: int = 512
    
    # Valence vectors (positive - negative mood)
    high_valence: np.ndarray = field(default_factory=lambda: np.zeros(512))
    low_valence: np.ndarray = field(default_factory=lambda: np.zeros(512))
    
    # Arousal vectors (excited - calm)
    high_arousal: np.ndarray = field(default_factory=lambda: np.zeros(512))
    low_arousal: np.ndarray = field(default_factory=lambda: np.zeros(512))
    
    # Dominance vectors (powerful - submissive)
    high_dominance: np.ndarray = field(default_factory=lambda: np.zeros(512))
    low_dominance: np.ndarray = field(default_factory=lambda: np.zeros(512))
    
    @property
    def valence_direction(self) -> np.ndarray:
        return self.high_valence - self.low_valence
    
    @property
    def arousal_direction(self) -> np.ndarray:
        return self.high_arousal - self.low_arousal
    
    @property
    def dominance_direction(self) -> np.ndarray:
        return self.high_dominance - self.low_dominance


class VADLatentMapper:
    """
    Maps VAD emotion vectors to MusicVAE latent space.
    
    Uses attribute vector arithmetic to modulate a base latent code
    based on emotional intent.
    """
    
    def __init__(
        self,
        attribute_vectors: Optional[EmotionAttributeVectors] = None,
        latent_dim: int = 512
    ):
        self.latent_dim = latent_dim
        self.attribute_vectors = attribute_vectors or self._init_default_vectors()
        
        # Scaling factors for each dimension
        self.valence_scale = 1.0
        self.arousal_scale = 0.8
        self.dominance_scale = 0.5
    
    def _init_default_vectors(self) -> EmotionAttributeVectors:
        """
        Initialize with heuristic vectors based on musical correlates.
        
        These should be replaced with vectors computed from EMOPIA dataset.
        The heuristics encode known musical-emotional relationships:
        - High valence: major mode, higher pitch center
        - High arousal: more notes, shorter durations, higher velocity variance
        - High dominance: wider pitch range, stronger bass presence
        """
        vectors = EmotionAttributeVectors(latent_dim=self.latent_dim)
        
        # Seed for reproducibility
        rng = np.random.RandomState(42)
        
        # Create orthogonal-ish direction vectors
        # In practice, compute these from actual MusicVAE encodings of EMOPIA samples
        
        # Valence: affects mode/harmony dimensions (indices 0-170 heuristically)
        vectors.high_valence = np.zeros(self.latent_dim)
        vectors.high_valence[:170] = rng.randn(170) * 0.3
        vectors.low_valence = -vectors.high_valence
        
        # Arousal: affects density/rhythm dimensions (indices 170-340)
        vectors.high_arousal = np.zeros(self.latent_dim)
        vectors.high_arousal[170:340] = rng.randn(170) * 0.4
        vectors.low_arousal = -vectors.high_arousal
        
        # Dominance: affects dynamics/range dimensions (indices 340-512)
        vectors.high_dominance = np.zeros(self.latent_dim)
        vectors.high_dominance[340:] = rng.randn(self.latent_dim - 340) * 0.25
        vectors.low_dominance = -vectors.high_dominance
        
        return vectors
    
    def vad_to_latent_offset(self, vad: VADState) -> np.ndarray:
        """
        Compute latent space offset from VAD state.
        
        Returns a vector to add to a base latent code.
        """
        # Normalize valence from [-1, 1] to scaling factor
        v_scale = vad.valence * self.valence_scale
        
        # Normalize arousal from [0, 1] to [-1, 1] then scale
        a_scale = (vad.arousal - 0.5) * 2.0 * self.arousal_scale
        
        # Normalize dominance from [0, 1] to [-1, 1] then scale
        d_scale = (vad.dominance - 0.5) * 2.0 * self.dominance_scale
        
        offset = (
            v_scale * self.attribute_vectors.valence_direction +
            a_scale * self.attribute_vectors.arousal_direction +
            d_scale * self.attribute_vectors.dominance_direction
        )
        
        return offset
    
    def modulate_latent(
        self, 
        base_latent: np.ndarray, 
        vad: VADState,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply VAD-based modulation to a base latent code.
        
        Args:
            base_latent: Original MusicVAE latent vector (512-dim)
            vad: Target emotional state
            strength: Interpolation strength (0.0 = no change, 1.0 = full modulation)
        
        Returns:
            Modulated latent vector
        """
        offset = self.vad_to_latent_offset(vad)
        return base_latent + strength * offset
    
    def interpolate_emotion(
        self,
        base_latent: np.ndarray,
        vad_start: VADState,
        vad_end: VADState,
        num_steps: int = 8
    ) -> List[np.ndarray]:
        """
        Generate latent trajectory from one emotion to another.
        
        Useful for emotional arcs within a piece.
        """
        latents = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0.0
            
            # Linear interpolation of VAD
            vad_interp = VADState(
                valence=vad_start.valence + t * (vad_end.valence - vad_start.valence),
                arousal=vad_start.arousal + t * (vad_end.arousal - vad_start.arousal),
                dominance=vad_start.dominance + t * (vad_end.dominance - vad_start.dominance)
            )
            
            latents.append(self.modulate_latent(base_latent, vad_interp))
        
        return latents


# =============================================================================
# GrooVAE Humanization with Emotion Parameters
# =============================================================================

@dataclass
class HumanizationParams:
    """Parameters for GrooVAE-style humanization."""
    timing_intensity: float = 0.5    # 0.0 = quantized, 1.0 = maximum swing
    velocity_variance: float = 0.3   # Velocity randomization amount
    swing_amount: float = 0.0        # Swing feel (0.0 = straight, 1.0 = heavy swing)
    humanize_downbeats: bool = True  # Apply less humanization to strong beats


class EmotionHumanizer:
    """
    Applies emotion-parameterized humanization to MIDI.
    
    Maps VAD state to humanization parameters:
    - High arousal -> tighter timing, higher velocity variance
    - Low arousal -> more swing, compressed dynamics
    - High valence -> slight forward push
    - Low valence -> slight drag
    """
    
    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = ticks_per_beat
        self.tick_16th = ticks_per_beat // 4
    
    def vad_to_humanization_params(self, vad: VADState) -> HumanizationParams:
        """Convert VAD state to humanization parameters."""
        params = HumanizationParams()
        
        # Arousal affects timing tightness
        # High arousal = tighter, more precise
        # Low arousal = looser, more swing
        params.timing_intensity = 0.2 + (1.0 - vad.arousal) * 0.6
        params.swing_amount = (1.0 - vad.arousal) * 0.4
        
        # Arousal affects velocity variance
        # High arousal = more dynamic range
        params.velocity_variance = 0.1 + vad.arousal * 0.4
        
        return params
    
    def humanize_notesequence(
        self,
        sequence: 'music_pb2.NoteSequence',
        vad: VADState,
        seed: Optional[int] = None
    ) -> 'music_pb2.NoteSequence':
        """
        Apply emotion-based humanization to NoteSequence.
        """
        if not MAGENTA_AVAILABLE:
            raise RuntimeError("note-seq not available")
        
        params = self.vad_to_humanization_params(vad)
        rng = np.random.RandomState(seed)
        
        # Get tempo for timing calculations
        qpm = sequence.tempos[0].qpm if sequence.tempos else 120.0
        beat_duration = 60.0 / qpm
        
        # Create new sequence with humanized notes
        humanized = music_pb2.NoteSequence()
        humanized.CopyFrom(sequence)
        humanized.ClearField('notes')
        
        for note in sequence.notes:
            new_note = humanized.notes.add()
            new_note.CopyFrom(note)
            
            # Timing offset based on beat position
            beat_position = note.start_time / beat_duration
            is_downbeat = (beat_position % 1.0) < 0.1
            
            # Reduce humanization on downbeats
            timing_scale = 0.3 if (is_downbeat and params.humanize_downbeats) else 1.0
            
            # Apply timing variation
            max_offset = 0.02 * params.timing_intensity * timing_scale  # Max 20ms
            timing_offset = rng.uniform(-max_offset, max_offset)
            
            # Apply swing (delay off-beats)
            if params.swing_amount > 0 and (beat_position * 2) % 1.0 > 0.4:
                swing_offset = params.swing_amount * 0.03  # Up to 30ms swing
                timing_offset += swing_offset
            
            # Apply valence-based push/pull
            # Positive valence = slight forward push (anticipation)
            # Negative valence = slight drag (behind the beat)
            valence_offset = -vad.valence * 0.005  # Up to 5ms
            timing_offset += valence_offset
            
            new_note.start_time = max(0, note.start_time + timing_offset)
            new_note.end_time = note.end_time + timing_offset
            
            # Velocity variation
            velocity_offset = int(rng.uniform(
                -params.velocity_variance * 30,
                params.velocity_variance * 30
            ))
            new_note.velocity = max(1, min(127, note.velocity + velocity_offset))
        
        return humanized
    
    def humanize_mido_messages(
        self,
        messages: List[mido.Message],
        vad: VADState,
        seed: Optional[int] = None
    ) -> List[mido.Message]:
        """Apply emotion-based humanization to mido messages."""
        params = self.vad_to_humanization_params(vad)
        rng = np.random.RandomState(seed)
        
        humanized = []
        for msg in messages:
            new_msg = msg.copy()
            
            if msg.type in ('note_on', 'note_off'):
                # Timing variation (applied to delta time)
                if msg.time > 0:
                    timing_var = int(msg.time * params.timing_intensity * 0.1)
                    offset = rng.randint(-timing_var, timing_var + 1)
                    new_msg = msg.copy(time=max(0, msg.time + offset))
                
                # Velocity variation (note_on only)
                if msg.type == 'note_on' and msg.velocity > 0:
                    vel_offset = int(rng.uniform(
                        -params.velocity_variance * 20,
                        params.velocity_variance * 20
                    ))
                    new_msg = new_msg.copy(
                        velocity=max(1, min(127, msg.velocity + vel_offset))
                    )
            
            humanized.append(new_msg)
        
        return humanized


# =============================================================================
# Real-Time MIDI Stream Processor
# =============================================================================

class RealtimeMIDIProcessor:
    """
    Processes real-time MIDI input for Magenta model inference.
    
    Accumulates notes into windows, triggers generation on phrase boundaries.
    """
    
    def __init__(
        self,
        window_duration: float = 2.0,  # Seconds per generation window
        overlap: float = 0.5,          # Window overlap ratio
        on_window_complete: Optional[Callable] = None
    ):
        self.window_duration = window_duration
        self.overlap = overlap
        self.on_window_complete = on_window_complete
        
        self.converter = NoteSequenceConverter()
        self.active_notes: Dict[Tuple[int, int], Tuple[float, int]] = {}
        self.note_buffer: List[Tuple[float, str, int, int, int]] = []
        self.start_time = None
        self.last_window_time = 0.0
    
    def process_message(self, msg: mido.Message) -> Optional['music_pb2.NoteSequence']:
        """
        Process incoming MIDI message.
        
        Returns NoteSequence when a window completes.
        """
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed = current_time - self.start_time
        
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (msg.note, msg.channel)
            self.active_notes[key] = (elapsed, msg.velocity)
            
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.note, msg.channel)
            if key in self.active_notes:
                start_time, velocity = self.active_notes.pop(key)
                self.note_buffer.append((
                    start_time, 'note', msg.note, velocity, msg.channel
                ))
        
        # Check if window complete
        window_elapsed = elapsed - self.last_window_time
        if window_elapsed >= self.window_duration:
            sequence = self._flush_window()
            self.last_window_time = elapsed - (self.window_duration * self.overlap)
            
            if self.on_window_complete:
                self.on_window_complete(sequence)
            
            return sequence
        
        return None
    
    def _flush_window(self) -> 'music_pb2.NoteSequence':
        """Convert buffered notes to NoteSequence."""
        if not MAGENTA_AVAILABLE:
            raise RuntimeError("note-seq not available")
        
        sequence = music_pb2.NoteSequence()
        sequence.tempos.add(qpm=120.0, time=0.0)
        
        # Find window boundaries
        window_end = self.last_window_time + self.window_duration
        window_start = window_end - self.window_duration
        
        for start, event_type, pitch, velocity, channel in self.note_buffer:
            if window_start <= start <= window_end:
                note = sequence.notes.add()
                note.pitch = pitch
                note.velocity = velocity
                note.start_time = start - window_start
                note.end_time = note.start_time + 0.25  # Default quarter note
                note.instrument = channel
        
        sequence.total_time = self.window_duration
        
        # Keep notes that might extend into next window
        self.note_buffer = [
            n for n in self.note_buffer 
            if n[0] >= window_start
        ]
        
        return sequence
    
    def reset(self) -> None:
        """Reset processor state."""
        self.active_notes.clear()
        self.note_buffer.clear()
        self.start_time = None
        self.last_window_time = 0.0


# =============================================================================
# Integration with Kelly ChordPredictor
# =============================================================================

def extract_chords_from_notesequence(
    sequence: 'music_pb2.NoteSequence',
    quantization_steps_per_beat: int = 4
) -> List[str]:
    """
    Extract chord symbols from NoteSequence.
    
    Uses note_seq's chord inference utilities.
    """
    if not MAGENTA_AVAILABLE:
        return []
    
    try:
        # Quantize sequence
        quantized = sequences_lib.quantize_note_sequence(
            sequence, 
            steps_per_quarter=quantization_steps_per_beat
        )
        
        # Infer chords (requires music21 or similar)
        # Simplified: return placeholder
        # In production, use music21.harmony.chordSymbolFigureFromChord
        return []
        
    except Exception:
        return []


# =============================================================================
# Factory Functions
# =============================================================================

def create_kelly_magenta_pipeline(
    vad_mapper: Optional[VADLatentMapper] = None,
    humanizer: Optional[EmotionHumanizer] = None
) -> Dict[str, object]:
    """
    Create complete Kelly-Magenta integration pipeline.
    
    Returns dict with all components for easy access.
    """
    return {
        'converter': NoteSequenceConverter(),
        'vad_mapper': vad_mapper or VADLatentMapper(),
        'humanizer': humanizer or EmotionHumanizer(),
        'realtime_processor': RealtimeMIDIProcessor(),
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Demo: VAD to latent mapping
    mapper = VADLatentMapper()
    
    # Grief state (low valence, medium-low arousal)
    grief_vad = VADState(valence=-0.6, arousal=0.3, dominance=0.4)
    
    # Generate latent offset
    offset = mapper.vad_to_latent_offset(grief_vad)
    print(f"Grief offset magnitude: {np.linalg.norm(offset):.3f}")
    
    # Demo: Emotion-based humanization params
    humanizer = EmotionHumanizer()
    params = humanizer.vad_to_humanization_params(grief_vad)
    print(f"Grief humanization - timing: {params.timing_intensity:.2f}, "
          f"swing: {params.swing_amount:.2f}, "
          f"velocity_var: {params.velocity_variance:.2f}")
    
    # Demo: Euphoria state (high valence, high arousal)
    euphoria_vad = VADState(valence=0.8, arousal=0.9, dominance=0.7)
    params = humanizer.vad_to_humanization_params(euphoria_vad)
    print(f"Euphoria humanization - timing: {params.timing_intensity:.2f}, "
          f"swing: {params.swing_amount:.2f}, "
          f"velocity_var: {params.velocity_variance:.2f}")
