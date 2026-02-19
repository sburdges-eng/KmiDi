"""
Tier1 convenience wrapper.
"""

from __future__ import annotations

from typing import Dict, Any
from pathlib import Path

from music_brain.structure.comprehensive_engine import HarmonyPlan, render_plan_to_midi
from music_brain.tier1.midi_pipeline import MIDIPipeline


class MIDIGenerationPipeline:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.pipeline = MIDIPipeline()

    def generate_midi(self, intent: Dict[str, Any], output_dir: str = "./") -> Dict[str, Any]:
        result = self.pipeline.run(intent)
        plan = HarmonyPlan(
            root_note="C",
            mode="major",
            tempo_bpm=result["bpm"],
            time_signature="4/4",
            length_bars=result["length_bars"],
            chord_symbols=["C", "Am", "F", "G"],
            harmonic_rhythm="1_chord_per_bar",
            mood_profile="neutral",
            complexity=0.3,
        )
        out_path = str(Path(output_dir) / "tier1_generated.mid")
        midi_path = render_plan_to_midi(plan, out_path)
        return {"status": "ok", "midi_path": midi_path, "data": result}
