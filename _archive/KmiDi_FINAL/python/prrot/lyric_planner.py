"""
Lyric Planner - Lyric-to-Phoneme Planning and Control Data Generation

Generates phoneme sequences, pitch curves, and control data from lyrics and melody.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .voice_profile import VoiceProfile, PhonemeType

logger = logging.getLogger(__name__)


@dataclass
class PhonemeControlData:
    """Control data output structure (Python version)"""

    phoneme_sequence: List[Dict[str, Any]]
    pitch_targets: List[Dict[str, Any]]
    midi_notes: List[Dict[str, Any]]
    automation_envelopes: List[Dict[str, Any]]
    breath_markers: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "phoneme_sequence": self.phoneme_sequence,
            "pitch_targets": self.pitch_targets,
            "midi_notes": self.midi_notes,
            "automation_envelopes": self.automation_envelopes,
            "breath_markers": self.breath_markers,
        }


class LyricPlanner:
    """Lyric-to-phoneme planner and control data generator"""

    def __init__(self, voice_profile: VoiceProfile):
        """Initialize lyric planner with voice profile"""
        self.voice_profile = voice_profile

    def generate_control_data(
        self,
        lyrics: str,
        melody_midi_notes: Optional[List[int]] = None,
        melody_timing: Optional[List[float]] = None,
        tempo_bpm: float = 120.0,
        expression_params: Optional[Dict[str, Any]] = None,
    ) -> PhonemeControlData:
        """
        Generate control data from lyrics and melody

        Args:
            lyrics: Lyrics text
            melody_midi_notes: MIDI note numbers (optional)
            melody_timing: Timing in beats (optional)
            tempo_bpm: Tempo in BPM
            expression_params: Optional expression parameters

        Returns:
            PhonemeControlData structure
        """
        logger.info(f"Generating control data for lyrics: {lyrics[:50]}...")

        # Convert lyrics to phonemes
        phoneme_sequence = self._lyrics_to_phonemes(lyrics)

        # Generate phoneme timings
        phoneme_timings = self._generate_phoneme_timings(phoneme_sequence, tempo_bpm)

        # Generate pitch targets
        pitch_targets = self._generate_pitch_targets(
            phoneme_timings, melody_midi_notes, melody_timing, tempo_bpm
        )

        # Generate MIDI notes
        midi_notes = self._generate_midi_notes(phoneme_timings, pitch_targets)

        # Generate automation envelopes
        automation_envelopes = self._generate_automation_envelopes(
            phoneme_timings, expression_params
        )

        # Generate breath markers
        breath_markers = self._generate_breath_markers(phoneme_timings)

        return PhonemeControlData(
            phoneme_sequence=phoneme_timings,
            pitch_targets=pitch_targets,
            midi_notes=midi_notes,
            automation_envelopes=automation_envelopes,
            breath_markers=breath_markers,
        )

    def _lyrics_to_phonemes(self, lyrics: str) -> List[int]:
        """Convert lyrics to phoneme sequence"""
        # Placeholder: Would use g2p_en or similar for phoneme conversion
        # For now, return placeholder
        phonemes = []

        # Simple placeholder conversion
        # In production, would use:
        # from g2p_en import G2p
        # g2p = G2p()
        # phoneme_symbols = g2p(lyrics)
        # phonemes = [stringToPhonemeType(p) for p in phoneme_symbols]

        return phonemes

    def _generate_phoneme_timings(
        self, phoneme_sequence: List[int], tempo_bpm: float
    ) -> List[Dict[str, Any]]:
        """Generate phoneme timings from sequence"""
        timings = []
        current_time_ms = 0.0

        for phoneme_type in phoneme_sequence:
            # Get duration from voice profile
            duration_stats = self.voice_profile.phoneme_durations.get(phoneme_type, {})
            duration_ms = duration_stats.get("mean_ms", 100.0)  # Default 100ms

            timing = {
                "phoneme": phoneme_type,
                "start_time_ms": current_time_ms,
                "duration_ms": duration_ms,
                "is_stressed": False,
                "is_emphasized": False,
                "stress_level": 0.0,
                "prominence": 0.5,
            }

            timings.append(timing)
            current_time_ms += duration_ms

        return timings

    def _generate_pitch_targets(
        self,
        phoneme_timings: List[Dict[str, Any]],
        melody_midi_notes: Optional[List[int]],
        melody_timing: Optional[List[float]],
        tempo_bpm: float,
    ) -> List[Dict[str, Any]]:
        """Generate pitch targets from melody"""
        targets = []

        if melody_midi_notes and melody_timing:
            # Map melody to phonemes
            note_idx = 0
            for timing in phoneme_timings:
                # Find overlapping note
                time_beats = (timing["start_time_ms"] / 1000.0) * (tempo_bpm / 60.0)

                while (
                    note_idx < len(melody_timing) - 1 and melody_timing[note_idx + 1] < time_beats
                ):
                    note_idx += 1

                if note_idx < len(melody_midi_notes):
                    target = {
                        "time_ms": timing["start_time_ms"],
                        "midi_note": melody_midi_notes[note_idx],
                        "cents_offset": 0.0,
                        "confidence": 1.0,
                    }
                    targets.append(target)
        else:
            # Default: use middle C
            for timing in phoneme_timings:
                target = {
                    "time_ms": timing["start_time_ms"],
                    "midi_note": 60,
                    "cents_offset": 0.0,
                    "confidence": 0.5,
                }
                targets.append(target)

        return targets

    def _generate_midi_notes(
        self, phoneme_timings: List[Dict[str, Any]], pitch_targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate MIDI notes from phonemes and pitch targets"""
        midi_notes = []

        pitch_idx = 0
        for timing in phoneme_timings:
            if pitch_idx < len(pitch_targets):
                pitch = pitch_targets[pitch_idx]

                note = {
                    "note_number": pitch["midi_note"],
                    "velocity": 100,
                    "start_time_ms": timing["start_time_ms"],
                    "duration_ms": timing["duration_ms"],
                    "channel": 0,
                }
                midi_notes.append(note)
                pitch_idx += 1

        return midi_notes

    def _generate_automation_envelopes(
        self, phoneme_timings: List[Dict[str, Any]], expression_params: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate automation envelopes"""
        envelopes = []

        # Dynamics envelope
        dynamics = {"type": "dynamics", "points": []}

        for timing in phoneme_timings:
            value = 0.8 if not timing["is_stressed"] else 1.0
            dynamics["points"].append({"time_ms": timing["start_time_ms"], "value": value})

        envelopes.append(dynamics)

        return envelopes

    def _generate_breath_markers(
        self, phoneme_timings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate breath markers at phrase boundaries"""
        markers = []

        # Place breath markers every 2-3 seconds
        phrase_duration_ms = 2000.0
        current_phrase_end = phrase_duration_ms

        for timing in phoneme_timings:
            if timing["start_time_ms"] >= current_phrase_end:
                marker = {"time_ms": current_phrase_end, "intensity": 0.5, "duration_ms": 200.0}
                markers.append(marker)
                current_phrase_end += phrase_duration_ms

        return markers
