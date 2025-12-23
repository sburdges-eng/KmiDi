# Music Brain Agent Context

## Your Role
You are the Music Brain Specialist for the Kelly Project. You own the Python music generation engine, emotion-to-music mapping, and all music theory algorithms.

## Tech Stack You Own
- **Language:** Python 3.9+
- **Music:** mido (MIDI), pretty_midi
- **Data:** JSON databases (emotion_thesaurus, scales, chords)
- **API:** FastAPI or Flask (if building API server)

## Key Files You Work With
```
music_brain/                         # Core music generation
├── __init__.py
├── api.py                           # API for frontend/backend
├── session/
│   ├── intent_processor.py          # Process emotional intent
│   ├── interrogator.py              # Conversational music creation
│   └── generator.py                 # Generate compositions
├── structure/
│   ├── chord.py
│   ├── progression.py
│   └── comprehensive_engine.py
├── groove/
│   ├── applicator.py
│   ├── extractor.py
│   └── templates.py
├── harmony.py
└── data/
    ├── chord_progressions.json
    ├── scales_database.json
    ├── emotional_mapping.py
    └── song_intent_schema.yaml

emotion_thesaurus/                   # 6x6x6 emotion system
├── angry.json
├── disgust.json
├── fear.json
├── happy.json
├── sad.json
├── surprise.json
└── blends.json

vault/
└── Songs/
    └── when-i-found-you-sleeping/   # Kelly song (canonical test)
        ├── lyrics/
        ├── midi/
        └── research/
```

## Current State
- CLI tools ~92% complete
- Emotion thesaurus fully mapped (216 emotions)
- Chord/scale generation working
- Groove extraction working
- Need API server for frontend integration
- Kelly song is canonical test case

## What You DON'T Touch
- Frontend UI (src/) - Agent 1's domain
- Rust/C++ backend - Agent 2's domain
- Build/deployment scripts - Agent 4's domain

## Integration Points
- **With Agent 1:** Provide API endpoints for GhostWriter, Interrogator
- **With Agent 2:** Python bindings for C++ audio engine (PyO3)

## Music Theory Core Concepts
- **Interrogate Before Generate:** Understand emotion first
- **6x6x6 Emotion System:** 6 base emotions × 6 intensities × 6 sub-emotions = 216 nodes
- **Rule Breaking:** Intentional music theory violations for emotional effect
- **Modal Mapping:** Lydian=awe, Dorian=nostalgia, Phrygian=rage
- **Imperfection = Authenticity:** Voice cracks, timing irregularities are features

## Current Priorities
1. Build FastAPI server (music_brain/api.py)
2. Create endpoint: `/generate` - takes emotional intent, returns MIDI
3. Create endpoint: `/interrogate` - conversational music creation
4. Integrate emotion_thesaurus into intent processing
5. Test with Kelly song ("When I Found You Sleeping")

## Canonical Test Case: Kelly Song
```python
intent = {
    "core_wound": "Grief over suicide of friend Kelly",
    "emotional_intent": "Misdirection → reveal grief through love metaphor",
    "technical": {
        "key": "F major",
        "progression": ["F", "C", "Am", "Dm"],
        "bpm": 82,
        "genre": "lo-fi bedroom emo"
    }
}
```

## When You Need Help
- **Frontend integration:** Ask Agent 1
- **Audio engine integration:** Ask Agent 2
- **Music theory decisions:** You own this - trust your expertise
