"""MIDI pipeline wrapper — orchestrator-facing; process_intent feeds harmony into MIDI path."""

from pathlib import Path
from typing import Any, Dict

from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.session.intent_processor import process_intent
from music_brain.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony


class MIDIGenerationPipeline:
    """Wires CompleteSongIntent to music_brain: process_intent → harmony → MIDI render."""

    def generate_midi(
        self,
        complete_intent: Any,
        output_dir: str = "",
    ) -> Dict[str, Any]:
        """Generate MIDI from CompleteSongIntent via process_intent → HarmonyResult → generate_midi_from_harmony."""
        if not isinstance(complete_intent, CompleteSongIntent):
            return {
                "status": "failed",
                "details": "midi_pipeline expects CompleteSongIntent",
            }
        intent = complete_intent
        output_path = Path(output_dir or ".").resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        # Default key/mode so process_intent can run when LLM returns minimal intent
        if not getattr(intent.technical_constraints, "technical_key", ""):
            intent.technical_constraints.technical_key = "F"
        if not getattr(intent.technical_constraints, "technical_mode", ""):
            intent.technical_constraints.technical_mode = "major"

        try:
            # Process intent: harmony, groove, arrangement, production, melody, texture, temporal
            processed = process_intent(intent)
            progression = processed["harmony"]
            groove = processed.get("groove")
            # TODO: use groove.timing_offsets_16th / velocity_curve for humanization when tier1 supports it
            tempo_bpm = groove.tempo_bpm if groove else 82
            tempo_range = getattr(
                intent.technical_constraints,
                "technical_tempo_range",
                (80, 120),
            )
            if not groove:
                tempo_bpm = sum(tempo_range) // 2 if tempo_range else 82

            # Feed process_intent harmony into MIDI path: progression → voicings → HarmonyResult
            generator = HarmonyGenerator()
            voicings = generator._chords_to_voicings(progression.chords, progression.key)
            harmony = HarmonyResult(
                chords=progression.chords,
                voicings=voicings,
                key=progression.key,
                mode=progression.mode,
                rule_break_applied=progression.rule_broken or None,
                emotional_justification=progression.rule_effect or None,
            )

            midi_name = "generated_from_intent.mid"
            midi_path = output_path / midi_name
            generate_midi_from_harmony(harmony, str(midi_path), tempo_bpm=tempo_bpm)

            # Integrate arrangement/melody/texture/temporal summary for orchestrator/dashboard
            arrangement = processed.get("arrangement")
            melody = processed.get("melody")
            texture = processed.get("texture")
            temporal = processed.get("temporal")
            out = {
                "status": "completed",
                "midi_path": str(midi_path),
                "key": harmony.key,
                "mode": harmony.mode,
                "chords": harmony.chords,
                "rule_broken": progression.rule_broken,
                "groove_tempo": tempo_bpm,
                "details": f"MIDI written to {midi_path} (process_intent → harmony → MIDI)",
            }
            if arrangement:
                out["arrangement_summary"] = (
                    len(arrangement.sections) if arrangement.sections else 0
                )
                if arrangement.sections:
                    out["arrangement_section_names"] = [
                        s.get("name", "") for s in arrangement.sections[:5]
                    ]
            if melody:
                out["melody_contour"] = getattr(melody, "contour", "") or ""
                out["melody_phrase_structure"] = getattr(melody, "phrase_structure", "") or ""
            if texture:
                out["texture_density"] = getattr(texture, "density_level", "") or ""
            if temporal:
                out["temporal_pacing"] = getattr(temporal, "pacing", "") or ""
            return out
        except Exception as e:
            return {
                "status": "error",
                "details": str(e),
            }
