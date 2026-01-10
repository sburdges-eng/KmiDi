"""
MIDI Editor Utilities

Python utilities for MIDI editing operations.
Part of Phase 5 of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class EditOperation(Enum):
    """Types of edit operations."""
    ADD_NOTE = "add_note"
    DELETE_NOTE = "delete_note"
    MOVE_NOTE = "move_note"
    RESIZE_NOTE = "resize_note"
    CHANGE_VELOCITY = "change_velocity"
    CHANGE_PITCH = "change_pitch"
    QUANTIZE = "quantize"
    HUMANIZE = "humanize"
    TRANSPOSE = "transpose"
    COPY = "copy"
    PASTE = "paste"
    CUT = "cut"


@dataclass
class MidiEditCommand:
    """Represents a single MIDI edit operation (for undo/redo)."""
    operation: EditOperation
    target_notes: List[int]  # Indices of notes affected
    old_state: Dict[str, Any]  # Old state before edit
    new_state: Dict[str, Any]  # New state after edit
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MidiEditor:
    """
    Python utility class for MIDI editing operations.

    Provides editing functions that can be called from UI components.

    Usage:
        editor = MidiEditor(generated_midi)
        editor.add_note(pitch=60, start_tick=0, duration=480, velocity=100)
        editor.quantize_notes(quantize_value=240)
    """

    def __init__(self, midi_data=None):
        """
        Initialize MIDI editor.

        Args:
            midi_data: GeneratedMidi object or similar structure
        """
        self.midi_data = midi_data
        self.undo_stack: List[MidiEditCommand] = []
        self.redo_stack: List[MidiEditCommand] = []
        self.clipboard: List[Dict[str, Any]] = []  # For copy/paste

    def add_note(
        self,
        pitch: int,
        start_tick: int,
        duration: int,
        velocity: int = 100,
        channel: int = 0,
        part: str = "melody"  # Which part to add to
    ) -> bool:
        """
        Add a note to the MIDI data.

        Returns:
            True if successful
        """
        if not self.midi_data:
            return False

        note = {
            "pitch": pitch,
            "startTick": start_tick,
            "durationTicks": duration,
            "velocity": velocity,
            "channel": channel,
        }

        # Get appropriate note list
        note_list = self._get_note_list(part)
        if note_list is not None:
            note_list.append(note)
            return True

        return False

    def delete_note(self, part: str, note_index: int) -> bool:
        """Delete a note by index."""
        note_list = self._get_note_list(part)
        if note_list and 0 <= note_index < len(note_list):
            del note_list[note_index]
            return True
        return False

    def move_note(
        self,
        part: str,
        note_index: int,
        new_start_tick: int,
        new_pitch: Optional[int] = None
    ) -> bool:
        """Move a note (change position and/or pitch)."""
        note_list = self._get_note_list(part)
        if note_list and 0 <= note_index < len(note_list):
            note = note_list[note_index]
            note["startTick"] = new_start_tick
            if new_pitch is not None:
                note["pitch"] = new_pitch
            return True
        return False

    def resize_note(
        self,
        part: str,
        note_index: int,
        new_duration: int
    ) -> bool:
        """Change note duration."""
        note_list = self._get_note_list(part)
        if note_list and 0 <= note_index < len(note_list):
            note = note_list[note_index]
            note["durationTicks"] = max(1, new_duration)  # Minimum 1 tick
            return True
        return False

    def change_velocity(
        self,
        part: str,
        note_index: int,
        new_velocity: int
    ) -> bool:
        """Change note velocity."""
        note_list = self._get_note_list(part)
        if note_list and 0 <= note_index < len(note_list):
            note = note_list[note_index]
            note["velocity"] = max(0, min(127, new_velocity))
            return True
        return False

    def quantize_notes(
        self,
        part: str,
        quantize_value: int,  # Ticks to quantize to
        selected_indices: Optional[List[int]] = None
    ) -> int:
        """
        Quantize notes to grid.

        Args:
            part: Part to quantize
            quantize_value: Grid size in ticks
            selected_indices: Optional list of note indices to quantize (all if None)

        Returns:
            Number of notes quantized
        """
        note_list = self._get_note_list(part)
        if not note_list:
            return 0

        indices = selected_indices if selected_indices is not None else range(len(note_list))
        quantized_count = 0

        for idx in indices:
            if 0 <= idx < len(note_list):
                note = note_list[idx]
                old_tick = note["startTick"]
                # Round to nearest quantize_value
                new_tick = round(old_tick / quantize_value) * quantize_value
                if new_tick != old_tick:
                    note["startTick"] = new_tick
                    quantized_count += 1

        return quantized_count

    def humanize_notes(
        self,
        part: str,
        timing_variance: float = 5.0,  # +/- ticks
        velocity_variance: float = 5.0,  # +/- velocity
        selected_indices: Optional[List[int]] = None
    ) -> int:
        """
        Add human-like variations to notes.

        Returns:
            Number of notes humanized
        """
        import random

        note_list = self._get_note_list(part)
        if not note_list:
            return 0

        indices = selected_indices if selected_indices is not None else range(len(note_list))
        humanized_count = 0

        for idx in indices:
            if 0 <= idx < len(note_list):
                note = note_list[idx]

                # Random timing offset
                timing_offset = random.uniform(-timing_variance, timing_variance)
                note["startTick"] = max(0, int(note["startTick"] + timing_offset))

                # Random velocity variation
                vel_offset = random.uniform(-velocity_variance, velocity_variance)
                new_velocity = int(note["velocity"] + vel_offset)
                note["velocity"] = max(1, min(127, new_velocity))

                humanized_count += 1

        return humanized_count

    def transpose_notes(
        self,
        part: str,
        semitones: int,
        selected_indices: Optional[List[int]] = None
    ) -> int:
        """
        Transpose notes by semitones.

        Returns:
            Number of notes transposed
        """
        note_list = self._get_note_list(part)
        if not note_list:
            return 0

        indices = selected_indices if selected_indices is not None else range(len(note_list))
        transposed_count = 0

        for idx in indices:
            if 0 <= idx < len(note_list):
                note = note_list[idx]
                new_pitch = note["pitch"] + semitones
                # Clamp to valid MIDI range
                if 0 <= new_pitch <= 127:
                    note["pitch"] = new_pitch
                    transposed_count += 1

        return transposed_count

    def copy_notes(
        self,
        part: str,
        note_indices: List[int]
    ) -> int:
        """
        Copy notes to clipboard.

        Returns:
            Number of notes copied
        """
        note_list = self._get_note_list(part)
        if not note_list:
            return 0

        self.clipboard = []
        for idx in note_indices:
            if 0 <= idx < len(note_list):
                # Deep copy note
                note = note_list[idx].copy()
                self.clipboard.append(note)

        return len(self.clipboard)

    def paste_notes(
        self,
        part: str,
        start_tick: int,
        offset_pitch: int = 0
    ) -> int:
        """
        Paste notes from clipboard.

        Args:
            part: Part to paste into
            start_tick: Starting position for pasted notes
            offset_pitch: Pitch offset to apply to pasted notes

        Returns:
            Number of notes pasted
        """
        if not self.clipboard:
            return 0

        note_list = self._get_note_list(part)
        if note_list is None:
            return 0

        # Find earliest note in clipboard to calculate offset
        earliest_tick = min(note.get("startTick", 0) for note in self.clipboard) if self.clipboard else 0

        pasted_count = 0
        for note in self.clipboard:
            # Create new note with offset
            new_note = note.copy()
            new_note["startTick"] = start_tick + (note.get("startTick", 0) - earliest_tick)
            new_note["pitch"] = note.get("pitch", 60) + offset_pitch
            # Clamp pitch
            if 0 <= new_note["pitch"] <= 127:
                note_list.append(new_note)
                pasted_count += 1

        return pasted_count

    def cut_notes(
        self,
        part: str,
        note_indices: List[int]
    ) -> int:
        """
        Cut notes (copy and delete).

        Returns:
            Number of notes cut
        """
        copied = self.copy_notes(part, note_indices)
        if copied > 0:
            # Delete in reverse order to maintain indices
            note_list = self._get_note_list(part)
            if note_list:
                for idx in sorted(note_indices, reverse=True):
                    if 0 <= idx < len(note_list):
                        del note_list[idx]
        return copied

    def _get_note_list(self, part: str) -> Optional[List]:
        """Get note list for a part."""
        if not self.midi_data:
            return None

        part_map = {
            "melody": getattr(self.midi_data, "melody", None) or getattr(self.midi_data, "notes", None),
            "bass": getattr(self.midi_data, "bass", None),
            "drums": getattr(self.midi_data, "drumGroove", None),
            "pad": getattr(self.midi_data, "pad", None),
            "strings": getattr(self.midi_data, "strings", None),
            "counter_melody": getattr(self.midi_data, "counterMelody", None),
            "rhythm": getattr(self.midi_data, "rhythm", None),
        }

        return part_map.get(part.lower(), getattr(self.midi_data, "notes", None))

    def undo(self) -> bool:
        """Undo last operation."""
        if not self.undo_stack:
            return False

        # TODO: Implement undo logic
        # This would restore old_state from the command
        return True

    def redo(self) -> bool:
        """Redo last undone operation."""
        if not self.redo_stack:
            return False

        # TODO: Implement redo logic
        return True


def main():
    """Example usage."""
    # Mock MIDI data structure
    class MockMidi:
        def __init__(self):
            self.melody = []
            self.bass = []

    midi = MockMidi()
    editor = MidiEditor(midi)

    # Add some notes
    editor.add_note(60, 0, 480, 100, part="melody")
    editor.add_note(64, 480, 480, 100, part="melody")

    print(f"Melody has {len(midi.melody)} notes")

    # Quantize
    quantized = editor.quantize_notes("melody", 240)
    print(f"Quantized {quantized} notes")


if __name__ == "__main__":
    main()
