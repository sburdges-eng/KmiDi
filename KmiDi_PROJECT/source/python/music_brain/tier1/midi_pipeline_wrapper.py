import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import mido

from ..session.intent_schema import CompleteSongIntent
from ..session.melody_generator import AdaptiveMelodyGenerator
from ..groove.groove_engine import GrooveSettings, humanize_drums
from ..production.dynamics_engine import DynamicsEngine, EmotionMatch, SongStructure
from ..production.drum_humanizer import DrumHumanizer
from ..structure.comprehensive_engine import TherapySession, render_plan_to_midi
from ..utils.midi_io import save_midi
from ..common import PPQ  # Assuming a common PPQ constant


class MIDIGenerationPipeline:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

        # Initialize Tier-1 components
        self.melody_generator = AdaptiveMelodyGenerator()
        self.dynamics_engine = DynamicsEngine()
        self.drum_humanizer_inst = DrumHumanizer()
        self.therapy_session = TherapySession()  # Used for HarmonyPlan generation

    def generate_midi(self, intent: CompleteSongIntent, output_dir: str = "./") -> Dict[str, Any]:
        # Set random seed for reproducibility if provided
        if self.seed is not None:
            random.seed(self.seed)

        # 1. Harmony Prediction (via TherapySession to leverage existing logic)
        self.therapy_session.process_core_input(
            intent.song_root.core_event or intent.song_intent.mood_primary or "neutral"
        )
        # Use existing logic to map intent values to TherapyState for HarmonyPlan generation
        motivation = 5  # Default, or derive from intent
        chaos_tolerance = 0.3  # Default, or derive from intent

        if intent.song_intent.mood_primary:
            # Attempt to derive motivation/chaos from emotional intent
            # This part can be more sophisticated later, for now, simple mapping
            if intent.song_intent.mood_primary in [
                "Rage",
                "Desperation",
                "Defiance",
            ]:
                motivation = 8
                chaos_tolerance = 0.7
            elif intent.song_intent.mood_primary in [
                "Grief",
                "Melancholy",
                "Dissociation",
            ]:
                motivation = 3
                chaos_tolerance = 0.2

        self.therapy_session.set_scales(motivation, chaos_tolerance)

        # Override with explicit technical constraints if present
        if intent.technical_constraints.technical_key:
            # Directly set the key and mode for the HarmonyPlan if available
            # This requires a small adjustment to TherapySession.generate_plan or a direct HarmonyPlan construction
            # For now, we'll let TherapySession generate and then try to override or influence it.
            pass  # More advanced integration needed here

        # Generate initial HarmonyPlan
        harmony_plan = self.therapy_session.generate_plan()
        harmony_plan.tempo_bpm = (
            random.randint(*intent.technical_constraints.technical_tempo_range)
            if intent.technical_constraints.technical_tempo_range
            else harmony_plan.tempo_bpm
        )
        harmony_plan.root_note = (
            intent.technical_constraints.technical_key or harmony_plan.root_note
        )
        # harmony_plan.mode will be set by TherapySession based on mood, or we can override if a specific 'technical_mode' exists in intent
        # harmony_plan.chord_symbols could be expanded by the LLM later

        # 2. Melody Generation (using AdaptiveMelodyGenerator)
        # Requires an emotion and optional length/profile. Using primary mood.
        melody_notes = self.melody_generator.generate(
            emotion=intent.song_intent.mood_primary or "neutral",
            length=harmony_plan.length_bars * 4,  # Example: 4 notes per bar
            # profile_name=... # Could be derived from intent
        )

        # 3. Drum Humanization (via DrumHumanizer)
        # Need some initial drum events to humanize. For a deterministic path, we'd generate a basic drum pattern first.
        # For now, let's assume a simple, repeating kick-snare pattern.
        drum_events: List[Dict[str, Any]] = []
        bars = harmony_plan.length_bars
        for bar in range(bars):
            # Kick on 1 and 3
            drum_events.append(
                {
                    "start_tick": bar * PPQ * 4,
                    "velocity": 100,
                    "pitch": 36,
                    "duration_ticks": PPQ,
                }
            )
            drum_events.append(
                {
                    "start_tick": bar * PPQ * 4 + PPQ * 2,
                    "velocity": 100,
                    "pitch": 36,
                    "duration_ticks": PPQ,
                }
            )
            # Snare on 2 and 4
            drum_events.append(
                {
                    "start_tick": bar * PPQ * 4 + PPQ,
                    "velocity": 90,
                    "pitch": 38,
                    "duration_ticks": PPQ,
                }
            )
            drum_events.append(
                {
                    "start_tick": bar * PPQ * 4 + PPQ * 3,
                    "velocity": 90,
                    "pitch": 38,
                    "duration_ticks": PPQ,
                }
            )
            # Hi-hats on all 8th notes
            for beat in range(8):
                drum_events.append(
                    {
                        "start_tick": bar * PPQ * 4 + beat * PPQ // 2,
                        "velocity": 70,
                        "pitch": 42,
                        "duration_ticks": PPQ // 4,
                    }
                )

        # Apply humanization based on groove feel and mood tension
        groove_settings = GrooveSettings(
            complexity=chaos_tolerance,  # Directly use chaos tolerance for complexity
            vulnerability=intent.song_intent.mood_secondary_tension,
        )
        humanized_drum_events = humanize_drums(
            events=drum_events,
            complexity=groove_settings.complexity,
            vulnerability=groove_settings.vulnerability,
            ppq=PPQ,
            settings=groove_settings,
            seed=self.seed,
        )

        # 4. Dynamics Application (if a SongStructure can be derived or provided)
        # For now, we'll assume a very simple structure or skip complex dynamics if not specified in intent.
        # If we had a full song structure from the intent, we could apply more nuanced dynamics.
        song_structure = SongStructure(
            sections=["intro", "chorus", "outro"], section_bars=[8, 16, 8]
        )
        emotion_match = EmotionMatch(
            base_emotion=intent.song_intent.mood_primary or "neutral",
            intensity_tier=5,
        )
        _ = self.dynamics_engine.get_arrangement_profile(song_structure, emotion_match)

        # 5. Render to MIDI file
        output_file_name = (
            f"kmidi_generated_{intent.title.replace(' ', '_')}.mid"
            if intent.title
            else "kmidi_generated.mid"
        )
        output_path = Path(output_dir) / output_file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # render_plan_to_midi is designed for HarmonyPlan, so we'll use it for harmony
        # and then manually add other tracks.
        harmony_midi_path = render_plan_to_midi(harmony_plan, str(output_path))
        harmony_mid = mido.MidiFile(harmony_midi_path)

        # Create a new MIDI file to combine everything
        final_mid = mido.MidiFile(ticks_per_beat=PPQ)

        # Add harmony track
        for track in harmony_mid.tracks:
            final_mid.tracks.append(track)

        # Add melody track (simple implementation for now, just note_on/off)
        melody_track = mido.MidiTrack()
        final_mid.tracks.append(melody_track)
        last_event_tick = 0  # absolute tick of the last event appended
        for note_pitch in melody_notes:
            note_on_tick = last_event_tick  # start immediately after the prior note_off
            delta_on = note_on_tick - last_event_tick  # zero except first iteration
            melody_track.append(
                mido.Message("note_on", note=note_pitch, velocity=90, time=delta_on)
            )
            # Quarter note duration; delta is relative to note_on
            melody_track.append(mido.Message("note_off", note=note_pitch, velocity=0, time=PPQ))
            last_event_tick = note_on_tick + PPQ  # advance absolute position

        # Add humanized drum track
        drum_track = mido.MidiTrack()
        final_mid.tracks.append(drum_track)
        current_tick = 0
        for event in humanized_drum_events:
            delta_time = event["start_tick"] - current_tick
            drum_track.append(
                mido.Message(
                    "note_on",
                    note=event["pitch"],
                    velocity=event["velocity"],
                    time=delta_time,
                    channel=9,
                )
            )
            # Assuming a short duration for drums for now
            drum_track.append(
                mido.Message(
                    "note_off",
                    note=event["pitch"],
                    velocity=0,
                    time=PPQ // 4,
                    channel=9,
                )
            )
            current_tick = (
                event["start_tick"] + PPQ // 4
            )  # Advance current_tick by note_on time + duration

        # Save the combined MIDI file
        save_midi(final_mid, str(output_path))

        # Read the file and base64 encode for the response
        with open(output_path, "rb") as f:
            midi_base64_data = base64.b64encode(f.read()).decode("utf-8")

        # Return structured MIDI plan
        midi_plan = {
            "status": "completed",
            "file_path": str(output_path),
            "tempo": harmony_plan.tempo_bpm,
            "key": harmony_plan.root_note + " " + harmony_plan.mode,
            "mood": intent.song_intent.mood_primary,
            "duration_bars": harmony_plan.length_bars,
            "midi_data_base64": midi_base64_data,
            "details": "MIDI generated successfully by KmiDi Tier-1 pipeline.",
        }

        return midi_plan
