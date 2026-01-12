"""
Dee - The Seasoned Musician.

Dee is the no-bullshit technical expert. He takes specific musical instructions
and translates them into concrete production settings. He guides the
"Music Intent Side".
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import re


@dataclass
class DeeTrackSpec:
    """Technical specification for a track produced by Dee."""
    time_signature: str
    tempo: Optional[int]
    instruments: List[str]
    intro_progression: Optional[str]
    effects: List[str]
    structure_notes: str


class Dee:
    """
    The technical wizard who translates jargon into tracks.
    """

    def __init__(self):
        self.known_instruments = ["viola", "synth bass",
                                  "drums", "guitar", "piano", "flute", "tuba"]
        self.known_effects = ["delay", "phaser",
                              "reverb", "distortion", "compression"]

    def produce(self, technical_description: str) -> DeeTrackSpec:
        """
        Parses a technical description and returns a track specification.
        """
        desc = technical_description.lower()

        # 1. Parse Time Signature (e.g., "5/8", "4/4")
        time_sig = "4/4"  # Default
        ts_match = re.search(r"(\d+/\d+)", desc)
        if ts_match:
            time_sig = ts_match.group(1)

        # 2. Parse Tempo (e.g., "120 bpm")
        tempo = None
        bpm_match = re.search(r"(\d+)\s*bpm", desc)
        if bpm_match:
            tempo = int(bpm_match.group(1))

        # 3. Parse Instruments
        instruments = []
        for inst in self.known_instruments:
            if inst in desc:
                instruments.append(inst)

        # 4. Parse Effects
        effects = []
        for fx in self.known_effects:
            if fx in desc:
                effects.append(fx)

        # 5. Extract Intro/Progression logic (Heuristic)
        intro_prog = None
        if "intro" in desc:
            # Try to grab context around "intro"
            # "ascending 2 steps in e minor"
            # This is a simple extraction for now
            intro_match = re.search(r"intro[^\.]*", desc)
            if intro_match:
                intro_prog = intro_match.group(0).strip()

        return DeeTrackSpec(
            time_signature=time_sig,
            tempo=tempo,
            instruments=instruments,
            intro_progression=intro_prog,
            effects=effects,
            structure_notes=f"Generated from: {technical_description[:50]}..."
        )

    def consult(self, spec: DeeTrackSpec) -> str:
        """
        Dee gives his professional opinion/confirmation.
        """
        response = [
            f"Alright, I got you. We're locking in a {spec.time_signature} groove.",
        ]

        if spec.instruments:
            inst_list = ", ".join(spec.instruments)
            response.append(f"Laying down tracks for: {inst_list}.")

        if spec.intro_progression:
            response.append(
                f"For the intro, I'm thinking: '{spec.intro_progression}'.")

        if spec.effects:
            fx_list = ", ".join(spec.effects)
            response.append(f"Dialing in the {fx_list}.")

        response.append(
            "I'll have this printed to stems before you finish your coffee.")
        return " ".join(response)
