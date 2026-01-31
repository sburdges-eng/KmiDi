"""
MIDI Transformers - Real-time MIDI message transformation functions.

Provides common transformation patterns for live MIDI processing:
- Transposition
- Velocity scaling
- Chord generation
- Arpeggiation
- Humanization
"""

from typing import Optional, List, Dict
import random
import time

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

from music_brain.realtime.midi_processor import MidiTransformCallback


def create_transpose_transformer(semitones: int) -> MidiTransformCallback:
    """
    Create a transformer that transposes notes by semitones.
    
    Args:
        semitones: Number of semitones to transpose (positive = up, negative = down)
    
    Returns:
        Transform callback function
    """
    def transform(msg: mido.Message) -> Optional[mido.Message]:
        if msg.type in ('note_on', 'note_off'):
            new_note = msg.note + semitones
            # Clamp to valid MIDI range (0-127)
            new_note = max(0, min(127, new_note))
            msg = msg.copy(note=new_note)
        return msg
    
    return transform


def create_velocity_scale_transformer(scale: float, min_vel: int = 1, max_vel: int = 127) -> MidiTransformCallback:
    """
    Create a transformer that scales note velocities.
    
    Args:
        scale: Velocity multiplier (0.5 = half velocity, 2.0 = double)
        min_vel: Minimum velocity after scaling
        max_vel: Maximum velocity after scaling
    
    Returns:
        Transform callback function
    """
    def transform(msg: mido.Message) -> Optional[mido.Message]:
        if msg.type == 'note_on' and msg.velocity > 0:
            new_vel = int(msg.velocity * scale)
            new_vel = max(min_vel, min(max_vel, new_vel))
            msg = msg.copy(velocity=new_vel)
        return msg
    
    return transform


def create_chord_generator_transformer(
    chord_notes: List[int],
    root_note: Optional[int] = None
) -> MidiTransformCallback:
    """
    Create a transformer that generates chords from single notes.
    
    When a note is played, it generates a chord based on the chord_notes intervals.
    
    Args:
        chord_notes: List of intervals (in semitones) for chord voicing
        root_note: If provided, use this as fixed root (otherwise use input note)
    
    Returns:
        Transform callback function
    """
    def transform(msg: mido.Message) -> Optional[mido.Message]:
        if msg.type == 'note_on' and msg.velocity > 0:
            # Determine root
            if root_note is not None:
                root = root_note
            else:
                root = msg.note
            
            # Generate chord notes
            chord_messages = []
            for interval in chord_notes:
                note = root + interval
                if 0 <= note <= 127:
                    chord_msg = msg.copy(note=note)
                    chord_messages.append(chord_msg)
            
            # Return first note, others would need to be sent separately
            # For now, return the root with original note
            # (Full chord generation requires multiple message sends)
            return msg
        
        return msg
    
    return transform


def create_arpeggiator_transformer(
    pattern: List[int],  # Pattern of intervals
    speed_ms: int = 100,
    direction: str = 'up'  # 'up', 'down', 'updown', 'random'
) -> MidiTransformCallback:
    """
    Create an arpeggiator transformer.
    
    Note: This is a simplified version. Full arpeggiator needs state management.
    
    Args:
        pattern: List of intervals for arpeggio pattern
        speed_ms: Speed of arpeggio (not used in simple version)
        direction: Arpeggio direction
    
    Returns:
        Transform callback function
    """
    state = {'index': 0, 'last_note': None, 'last_time': 0.0}
    
    def transform(msg: mido.Message) -> Optional[mido.Message]:
        if msg.type == 'note_on' and msg.velocity > 0:
            current_time = time.time()
            
            # Check if enough time has passed
            if current_time - state['last_time'] < (speed_ms / 1000.0):
                return None  # Suppress message (too fast)
            
            # Get pattern index
            if direction == 'up':
                idx = state['index'] % len(pattern)
            elif direction == 'down':
                idx = (len(pattern) - 1) - (state['index'] % len(pattern))
            elif direction == 'random':
                idx = random.randint(0, len(pattern) - 1)
            else:  # updown
                cycle_len = len(pattern) * 2 - 2
                pos = state['index'] % cycle_len
                if pos < len(pattern):
                    idx = pos
                else:
                    idx = cycle_len - pos
            
            # Calculate note
            interval = pattern[idx]
            new_note = msg.note + interval
            new_note = max(0, min(127, new_note))
            
            state['index'] += 1
            state['last_note'] = new_note
            state['last_time'] = current_time
            
            return msg.copy(note=new_note)
        
        return msg
    
    return transform


def create_humanize_transformer(
    timing_jitter_ms: float = 5.0,
    velocity_variation: int = 5
) -> MidiTransformCallback:
    """
    Create a humanization transformer (adds slight timing/velocity variation).
    
    Note: Timing jitter requires delayed sending, which is complex in real-time.
    This version only applies velocity variation.
    
    Args:
        timing_jitter_ms: Timing variation in milliseconds (not implemented in simple version)
        velocity_variation: Maximum velocity variation
    
    Returns:
        Transform callback function
    """
    def transform(msg: mido.Message) -> Optional[mido.Message]:
        if msg.type == 'note_on' and msg.velocity > 0:
            # Apply velocity variation
            variation = random.randint(-velocity_variation, velocity_variation)
            new_vel = msg.velocity + variation
            new_vel = max(1, min(127, new_vel))
            return msg.copy(velocity=new_vel)
        
        return msg
    
    return transform


def create_channel_router_transformer(output_channel: int) -> MidiTransformCallback:
    """
    Create a transformer that routes all messages to a specific channel.
    
    Args:
        output_channel: Target MIDI channel (0-15)
    
    Returns:
        Transform callback function
    """
    def transform(msg: mido.Message) -> Optional[mido.Message]:
        if hasattr(msg, 'channel'):
            return msg.copy(channel=output_channel)
        return msg
    
    return transform


def create_filter_transformer(
    min_note: int = 0,
    max_note: int = 127,
    min_velocity: int = 0,
    max_velocity: int = 127
) -> MidiTransformCallback:
    """
    Create a transformer that filters notes by range and velocity.
    
    Args:
        min_note: Minimum note number (0-127)
        max_note: Maximum note number (0-127)
        min_velocity: Minimum velocity
        max_velocity: Maximum velocity
    
    Returns:
        Transform callback function (returns None to filter out messages)
    """
    def transform(msg: mido.Message) -> Optional[mido.Message]:
        if msg.type == 'note_on':
            if not (min_note <= msg.note <= max_note):
                return None  # Filter out
            if not (min_velocity <= msg.velocity <= max_velocity):
                return None  # Filter out
        elif msg.type == 'note_off':
            if not (min_note <= msg.note <= max_note):
                return None  # Filter out
        
        return msg
    
    return transform

