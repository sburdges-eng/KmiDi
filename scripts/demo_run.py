#!/usr/bin/env python3
"""
Demo run: full Intent Builder criteria → 1 minute MIDI + WAV.

Uses all interactive features (core_desire, mood, genre, tempo, key_mode,
structure, instruments, groove_feel, narrative_arc, rule_to_break,
rule_justification) and produces matching ~1 minute audio and MIDI in
scripts/output/.

Run from repo root:
  python scripts/demo_run.py
  # or
  micromamba run -n kmidi-core python scripts/demo_run.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 1 minute at 120 BPM, 4/4 → 30 bars
DEMO_TEMPO = 120
DEMO_DURATION_MINUTES = 1.0
DEMO_TOTAL_BARS = int((DEMO_DURATION_MINUTES * DEMO_TEMPO) / 4)  # 30 bars


def build_demo_intent_request():
    """Full CompleteSongIntentRequest matching all UI interactive features."""
    from music_brain.engine_api.schema import (
        CompleteSongIntentRequest,
        StructureSection,
        TrackIntent,
    )

    # Structure: intro 4 + verse 8*2 + chorus 8 = 4+16+8 = 28 bars (~56 s); add 2 bar bridge → 30
    structure = [
        StructureSection(name="intro", bars=4, repetitions=1),
        StructureSection(name="verse", bars=8, repetitions=2),
        StructureSection(name="chorus", bars=8, repetitions=1),
        StructureSection(name="bridge", bars=2, repetitions=1),
    ]
    assert sum(s.bars * s.repetitions for s in structure) == DEMO_TOTAL_BARS

    return CompleteSongIntentRequest(
        core_desire=(
            "A driving synthwave track that evokes a nighttime highway drive; "
            "resolve internal tension with hopeful momentum."
        ),
        mood_primary="Nostalgic",
        genre="Synthwave",
        tempo=DEMO_TEMPO,
        key_mode="A minor",
        structure=structure,
        instruments=[
            TrackIntent(instrument="synth_lead", techniques=["portamento", "sustain"]),
            TrackIntent(instrument="synth_bass", techniques=[]),
            TrackIntent(instrument="piano", techniques=["staccato"]),
        ],
        allow_legacy_fallback=False,
        groove_feel="Straight/Driving",
        narrative_arc="Climb-to-Climax",
        rule_to_break="parallel fifths",
        rule_justification="Intentional tension for emotional payoff in the chorus",
    )


def request_to_complete_intent(validated):
    """Convert validated CompleteSongIntentRequest → CompleteSongIntent (same as API)."""
    from music_brain.session.intent_schema import CompleteSongIntent

    key_parts = validated.key_mode.split()
    technical_key = key_parts[0]
    technical_mode = key_parts[1].lower()
    tempo_range = (max(60, validated.tempo - 20), min(140, validated.tempo + 20))
    return CompleteSongIntent(
        core_event=validated.core_desire,
        core_longing=validated.core_desire,
        mood_primary=validated.mood_primary,
        technical_genre=validated.genre,
        technical_tempo_range=tempo_range,
        technical_key=technical_key,
        technical_mode=technical_mode,
        vulnerability_scale=0.5,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def main():
    out_dir = REPO_ROOT / "scripts" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_path = out_dir / "demo_run.mid"
    wav_path = out_dir / "demo_run.wav"

    from music_brain.api import api
    from music_brain.structure.comprehensive_engine import HarmonyPlan, render_plan_to_midi
    from music_brain.audio import render_midi_to_audio

    logger.info("Building demo intent (all interactive features, 1 min @ %s BPM)...", DEMO_TEMPO)
    validated = build_demo_intent_request()

    complete_intent = request_to_complete_intent(validated)
    logger.info("Processing intent (harmony, groove, arrangement)...")
    result = api.process_song_intent(complete_intent, output_json=None)

    harmony = result.get("harmony")
    if not harmony:
        logger.error("No harmony in result; cannot render MIDI. Result keys: %s", list(result.keys()))
        return 1

    key_parts = validated.key_mode.split()
    root_note = key_parts[0]
    mode = key_parts[1].lower()
    structure = [
        section.model_dump() if hasattr(section, "model_dump") else section.dict()
        for section in validated.structure
    ]
    instruments = [
        track.model_dump() if hasattr(track, "model_dump") else track.dict()
        for track in validated.instruments
    ]

    plan = HarmonyPlan(
        root_note=root_note,
        mode=mode,
        tempo_bpm=validated.tempo,
        time_signature="4/4",
        length_bars=DEMO_TOTAL_BARS,
        chord_symbols=harmony.get("chords", ["Am", "F", "C", "G"]),
        harmonic_rhythm="1_chord_per_bar",
        mood_profile=result.get("intent_summary", {}).get("mood", "neutral"),
        complexity=0.5,
        structure=structure,
        instruments=instruments,
    )
    logger.info("Rendering MIDI to %s...", mid_path)
    render_plan_to_midi(plan, str(mid_path))
    if not mid_path.exists():
        logger.warning(
            "MIDI file was not written (music_brain.daw.logic may be unavailable). "
            "Intent pipeline ran successfully; install full stack for MIDI/WAV output."
        )
        return 0
    logger.info("Rendering WAV to %s...", wav_path)
    try:
        render_midi_to_audio(str(mid_path), str(wav_path))
    except Exception as e:
        logger.warning("Audio render failed: %s", e)
    logger.info("Done. MIDI: %s  WAV: %s", mid_path, wav_path if wav_path.exists() else "(not generated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
