# KmiDi Brain — Integration Map

Quick reference for cross-module flows and who consumes what. Contracts live in `docs/CONTRACTS.md`.

## Flow (critical path)

```
User text
    → LLMReasoningEngine.parse_user_intent()   [mcp_workstation.llm_reasoning_engine]
    → CompleteSongIntent

CompleteSongIntent
    → process_intent(intent)                    [music_brain.session.intent_processor]
    → harmony, groove, arrangement, production, melody, texture, temporal, intent_summary

tier1 (MIDI path):
    process_intent → harmony + groove.tempo_bpm → HarmonyResult → generate_midi_from_harmony()
    → MIDI file + return dict (status, midi_path, chords, groove_tempo, arrangement_summary, melody_contour, texture_density, temporal_pacing, …)

Orchestrator:
    complete_intent → midi_pipeline.generate_midi() → complete_intent.midi_plan = result
    (midi_plan carries all keys from tier1 return for dashboard/downstream)

API:
    process_intent() → full serializable dict (harmony, groove, arrangement, production, melody, texture, temporal, intent_summary)
```

## Consumers of process_intent output

| Module | What it uses | Purpose |
|--------|----------------|---------|
| **music_brain.tier1.midi_pipeline_wrapper** | harmony, groove (tempo); arrangement/melody/texture/temporal for summary keys | Render MIDI; attach summary to return for orchestrator. |
| **music_brain.api** (process_song_intent) | Full: harmony, groove, arrangement, production, melody, texture, temporal, intent_summary | Single API surface for desktop/REST; downstream can use melody/texture/temporal. |
| **mcp_workstation.orchestrator** | midi_pipeline return dict → midi_plan | Status, path, chords, groove_tempo, arrangement/melody/texture/temporal summary. |

## Other integrations

| From | To | What |
|------|-----|------|
| **music_brain.harmony** | tier1, API | HarmonyGenerator, HarmonyResult, generate_midi_from_harmony. |
| **music_brain.groove** | API, realtime | extract_groove, apply_groove, humanize_midi_file; realtime midi_processor can use groove analysis. |
| **music_brain.structure** | API | analyze_chords, detect_sections, progression tools, TherapySession. |
| **penta_core.ml.inference** | run_brain penta mode | Model registry / inference entry; no direct tie to intent path. |
| **Spectocloud (visualization)** | Body → API → Brain | Body: `KmiDi_CANON/body/hooks/useMusicBrain.ts` → `POST /spectocloud/render`. Brain: `music_brain/visualization/spectocloud.py` when restored; spine-included per CONTRACTS §5b. |

## Spectocloud path (spine inclusion)

- **Body:** `KmiDi_CANON/body/hooks/useMusicBrain.ts` — `renderSpectocloud(payload)` → `POST /spectocloud/render`.
- **API:** Route `/spectocloud/render` to be implemented/restored where music_brain API is served.
- **Brain:** `KmiDi_CANON/brain/music_brain/visualization/spectocloud.py` (canonical path when restored from forensic or reimplemented).

## Adding integrations

- New consumers of `process_intent` output: document in CONTRACTS §5 (Integration map) and in this file.
- New spine callers: keep single entry (run_brain → orchestrator / penta); no parallel trees.
- Groove humanization: tier1 TODO to use groove.timing_offsets_16th / velocity_curve when supported; realtime layer can align with intent_processor groove later.
