#!/usr/bin/env python3
"""
Generate WASABI-format training manifest from KmiDi emotion thesaurus,
chord progressions, and song intent examples.

Output: data/wasabi/processed/{train,val,test}.jsonl

Usage:
    python scripts/generate_wasabi_manifest.py
    python scripts/generate_wasabi_manifest.py --samples 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = ROOT / "data" / "wasabi" / "processed"

# Base emotions from thesaurus
BASE_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "joy", "calm"]

# Chord progressions: (name, chords, emotion_hint)
PROGRESSIONS = [
    ("I-IV-V-I", ["C", "F", "G", "C"], "happy"),
    ("I-V-vi-IV", ["C", "G", "Am", "F"], "happy"),
    ("vi-IV-I-V", ["Am", "F", "C", "G"], "sad"),
    ("I-vi-IV-V", ["C", "Am", "F", "G"], "calm"),
    ("i-VI-III-VII", ["Am", "F", "C", "G"], "sad"),
    ("i-IV-i", ["Am", "Dm", "Am"], "angry"),
    ("ii-V-I", ["Dm", "G", "C"], "joy"),
    ("I-vi-ii-V", ["C", "Am", "Dm", "G"], "calm"),
    ("I-IV-vi-V", ["C", "F", "Am", "G"], "happy"),
    ("i-bVII-bVI-V", ["Am", "G", "F", "E"], "angry"),
    ("I-bVII-IV-I", ["C", "Bb", "F", "C"], "joy"),
    ("i-V-i", ["Am", "E", "Am"], "fear"),
    ("IV-I-IV-V", ["F", "C", "F", "G"], "happy"),
    ("I-bIII-IV", ["C", "Eb", "F"], "surprise"),
    ("i-iv-i", ["Am", "Dm", "Am"], "disgust"),
]

# Placeholder lyrics snippets by emotion
LYRICS_BY_EMOTION = {
    "happy": ["Feeling alive today", "Everything is bright", "Singing in the sun"],
    "sad": ["When the rain falls", "Lost in the gray", "Quiet empty room"],
    "angry": ["Rage inside", "Breaking through the wall", "No more waiting"],
    "fear": ["Shadows closing in", "Heartbeat rushing", "Don't look back"],
    "surprise": ["Suddenly it changed", "Never saw it coming", "Out of nowhere"],
    "disgust": ["Turn away", "Can't believe it", "Sour taste"],
    "joy": ["Leaping for the sky", "Laughing out loud", "This is the moment"],
    "calm": ["Still water", "Breathing slow", "Peace at last"],
}


def load_song_intents() -> List[Dict[str, Any]]:
    """Load song intent examples for richer samples."""
    path = DATA / "song_intent_examples.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    examples = data.get("examples", {})
    out: List[Dict[str, Any]] = []
    for k, v in examples.items():
        si = v.get("song_intent", {})
        tc = v.get("song_technical_constraints", {})
        exp = v.get("expected_output", {})
        mood = (si.get("mood_primary") or "").lower()
        progression = exp.get("progression", "")
        chords = [s.strip() for s in progression.replace("-", " ").split() if s.strip() and s[0].isalpha()]
        if not chords:
            continue
        out.append({
            "title": v.get("title", k),
            "mood": mood,
            "chords": chords,
            "tempo": tc.get("technical_tempo_range", [100, 120])[0],
            "key": tc.get("technical_key", "C major"),
        })
    return out


def emotion_to_lyrics_emotion(emotion: str, intensity: float = 0.7) -> dict[str, float]:
    """Build lyrics_emotion dict for one primary emotion."""
    d = {e: 0.0 for e in BASE_EMOTIONS}
    d[emotion] = intensity
    # Add slight secondary
    others = [e for e in BASE_EMOTIONS if e != emotion]
    if others:
        d[random.choice(others)] = round(random.uniform(0.05, 0.2), 3)
    return d


def build_sample(
    sample_id: int,
    emotion: str,
    progression: Tuple[str, List[str], str],
    song_intent: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    name, chords, _ = progression
    if song_intent and song_intent.get("chords"):
        chords = song_intent["chords"]
    intensity = random.uniform(0.4, 0.95)
    lyrics_emotion = emotion_to_lyrics_emotion(emotion, intensity)
    lyrics = random.choice(LYRICS_BY_EMOTION.get(emotion, LYRICS_BY_EMOTION["calm"]))
    tempo = int(random.gauss(110, 20))
    tempo = max(60, min(180, tempo))
    year = random.randint(1980, 2024)
    return {
        "id": f"kmidi_{sample_id}",
        "title": f"Sample {sample_id}",
        "artist": "KmiDi",
        "lyrics": lyrics,
        "lyrics_emotion": lyrics_emotion,
        "lyrics_topics": [emotion],
        "chords": chords,
        "tempo": tempo,
        "key": "C major",
        "year": year,
        "genre": ["pop"],
    }


def main():
    ap = argparse.ArgumentParser(description="Generate WASABI-format manifest from KmiDi data")
    ap.add_argument("--samples", type=int, default=8000, help="Total samples to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    intents = load_song_intents()

    samples = []
    for i in range(args.samples):
        emotion = random.choice(BASE_EMOTIONS)
        prog = random.choice(PROGRESSIONS)
        intent = random.choice(intents) if intents else None
        samples.append(build_sample(i, emotion, prog, intent))

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = args.output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in part:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {path} ({len(part)} samples)")

    print("Done. Use data/wasabi/processed/train.jsonl with WasabiDataset / train_midi_generator.")


if __name__ == "__main__":
    main()
