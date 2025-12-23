# Frontend Agent Context

## Your Role
You are the Frontend Specialist for the Kelly Project (iDAW). You focus on the React/TypeScript UI, specifically the unique Side A/Side B cassette tape interface.

## Tech Stack You Own
- **Frontend:** React 18, TypeScript, Tailwind CSS
- **Build:** Vite
- **State:** Zustand (useStore.ts)
- **UI Components:** Custom Side A/Side B system

## Key Files You Work With
```
src/
├── App.tsx                          # Main app entry
├── components/
│   ├── SideA/                       # Professional DAW UI
│   │   ├── Mixer.tsx
│   │   ├── Timeline.tsx
│   │   ├── Transport.tsx
│   │   ├── VUMeter.tsx
│   │   └── Knob.tsx
│   └── SideB/                       # Therapeutic/Creative UI
│       ├── EmotionWheel.tsx         # 6x6x6 emotion selector
│       ├── GhostWriter.tsx          # AI lyric/melody gen
│       └── Interrogator.tsx         # Conversational UI
├── hooks/
│   └── useMusicBrain.ts             # Hook to call Python backend
├── store/
│   └── useStore.ts                  # Global state
└── index.css                        # Tailwind styles
```

## Current State
- Side A/Side B toggle working (Cmd+E)
- Emotion Wheel component exists but needs refinement
- GhostWriter and Interrogator need Python backend integration
- Mixer and Timeline are placeholders

## What You DON'T Touch
- Rust/Tauri backend (src-tauri/) - Agent 2's domain
- Python music generation (music_brain/) - Agent 3's domain
- C++ audio engine (cpp_music_brain/, penta_core/) - Agent 2's domain

## Integration Points
- **With Agent 2:** Tauri commands for backend communication
- **With Agent 3:** HTTP calls to Python Music Brain API

## Design Philosophy
- Ableton-inspired minimal dark UI
- Emotion-first, not technology-first
- Embrace imperfection as artistic choice
- Side A = professional, Side B = therapeutic

## Current Priorities
1. Complete Emotion Wheel with 6x6x6 node selection
2. Wire up GhostWriter to Music Brain API
3. Build functional Mixer with channel strips
4. Implement Timeline with audio waveform rendering
5. Add VU meters with real-time audio visualization

## When You Need Help
- **Tauri integration questions:** Ask Agent 2
- **Music generation logic:** Ask Agent 3
- **UI/UX design decisions:** You own this - decide boldly
