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
        # Safely handle technical_tempo_range (may be None or invalid)
        tempo_range = intent.technical_constraints.technical_tempo_range
        if tempo_range and isinstance(tempo_range, (tuple, list)) and len(tempo_range) == 2:
            try:
                tempo_min, tempo_max = int(tempo_range[0]), int(tempo_range[1])
                if tempo_min < tempo_max:
                    harmony_plan.tempo_bpm = random.randint(tempo_min, tempo_max)
            except (ValueError, TypeError):
                # Invalid range, keep default from harmony_plan
                pass
        harmony_plan.root_note = (
            intent.technical_constraints.technical_key or harmony_plan.root_note
        )
        # Override mode if technical_mode is specified in intent
        if intent.technical_constraints.technical_mode:
            harmony_plan.mode = intent.technical_constraints.technical_mode
        # harmony_plan.chord_symbols could be expanded by the LLM later

        # 2. Melody Generation (using AdaptiveMelodyGenerator)
        # Requires an emotion and optional length/profile. Using primary mood.
        melody_length = max(1, harmony_plan.length_bars * 4)  # Ensure at least 1 note
        melody_notes = self.melody_generator.generate(
            emotion=intent.song_intent.mood_primary or "neutral",
            length=melody_length,  # Example: 4 notes per bar
            # profile_name=... # Could be derived from intent
        )
        # Safety check: ensure melody_notes is not empty
        if not melody_notes:
            # Fallback: generate a simple scale
            base_note = 60
            scale = [0, 2, 4, 5, 7, 9, 11, 12]
            melody_notes = [base_note + scale[i % len(scale)] for i in range(melody_length)]

        # 3. Drum Humanization (via DrumHumanizer)
        # Need some initial drum events to humanize. For a deterministic path, we'd generate a basic drum pattern first.
        # For now, let's assume a simple, repeating kick-snare pattern.
        drum_events: List[Dict[str, Any]] = []
        bars = max(1, harmony_plan.length_bars)  # Ensure at least 1 bar
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
        # Safely handle mood_secondary_tension (may not exist or be invalid)
        vulnerability = 0.5  # Default
        if hasattr(intent.song_intent, "mood_secondary_tension"):
            try:
                vulnerability = float(intent.song_intent.mood_secondary_tension)
                # Clamp to valid range [0.0, 1.0]
                vulnerability = max(0.0, min(1.0, vulnerability))
            except (ValueError, TypeError):
                # Invalid value, use default
                pass
        groove_settings = GrooveSettings(
            complexity=chaos_tolerance,  # Directly use chaos tolerance for complexity
            vulnerability=vulnerability,
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
            f"KmiDi_generated_{intent.title.replace(' ', '_')}.mid"
            if intent.title
            else "KmiDi_generated.mid"
        )
        output_path = Path(output_dir) / output_file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # render_plan_to_midi is designed for HarmonyPlan, so we'll use it for harmony
        # and then manually add other tracks.
        try:
            harmony_midi_path = render_plan_to_midi(harmony_plan, str(output_path))
            harmony_mid = mido.MidiFile(harmony_midi_path)
        except Exception as e:
            return {
                "status": "failed",
                "file_path": str(output_path),
                "details": f"Failed to render harmony plan to MIDI: {e}",
            }

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
            # Safety check: ensure valid MIDI note value
            if not isinstance(note_pitch, (int, float)):
                continue
            note_pitch = int(note_pitch)
            note_pitch = max(0, min(127, note_pitch))  # Clamp to valid MIDI range

            note_on_tick = last_event_tick  # start immediately after the prior note_off
            delta_on = note_on_tick - last_event_tick  # zero except first iteration
            melody_track.append(
                mido.Message("note_on", note=note_pitch, velocity=90, time=delta_on)
            )
            # Quarter note duration; delta is relative to note_on
            melody_track.append(
                mido.Message("note_off", note=note_pitch, velocity=0, time=PPQ)
            )
            last_event_tick = note_on_tick + PPQ  # advance absolute position

        # Add humanized drum track
        drum_track = mido.MidiTrack()
        final_mid.tracks.append(drum_track)
        current_tick = 0
        for event in humanized_drum_events:
            # Safety check: ensure event has required fields
            if not isinstance(event, dict):
                continue
            start_tick = event.get("start_tick", 0)
            pitch = event.get("pitch", 36)  # Default to kick drum
            velocity = event.get("velocity", 80)  # Default velocity

            # Ensure valid MIDI values
            pitch = max(0, min(127, int(pitch)))
            velocity = max(1, min(127, int(velocity)))

            delta_time = max(0, start_tick - current_tick)
            drum_track.append(
                mido.Message(
                    "note_on",
                    note=pitch,
                    velocity=velocity,
                    time=delta_time,
                    channel=9,
                )
            )
            # Assuming a short duration for drums for now
            drum_track.append(
                mido.Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    time=PPQ // 4,
                    channel=9,
                )
            )
            current_tick = start_tick + PPQ // 4

        # Save the combined MIDI file
        try:
            save_midi(final_mid, str(output_path))
        except Exception as e:
            return {
                "status": "failed",
                "file_path": str(output_path),
                "details": f"Failed to save MIDI file: {e}",
            }

        # Read the file and base64 encode for the response
        try:
            with open(output_path, "rb") as f:
                midi_base64_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return {
                "status": "failed",
                "file_path": str(output_path),
                "details": f"Failed to read and encode MIDI file: {e}",
            }

        # Return structured MIDI plan
        # Safely construct key string (root_note and mode are strings)
        key_str = f"{harmony_plan.root_note} {harmony_plan.mode}".strip()
        midi_plan = {
            "status": "completed",
            "file_path": str(output_path),
            "tempo": harmony_plan.tempo_bpm,
            "key": key_str,
            "mood": intent.song_intent.mood_primary or "",
            "duration_bars": harmony_plan.length_bars,
            "midi_data_base64": midi_base64_data,
            "details": "MIDI generated successfully by KmiDi Tier-1 pipeline.",
        }

        return midi_plan
