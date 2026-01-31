# KmiDi Brain — Contracts

<!-- STATUS: FINISHED -->
<!-- Completed: 2026-01-31 | Reason: Phase 1 complete; contracts written and stable. Content is reference only. -->

Single source of truth for input/output contracts of the load-bearing spine. Changes to a contract require updating this doc and all callers.

**Spine:** `run_brain.py` → `mcp_workstation.orchestrator` / `penta_core` → `music_brain.session` + `music_brain.tier1` → `music_brain.harmony` / `intent_processor` / structure / groove / **music_brain.visualization (spectocloud)**.

---

## 1. Intent → MIDI

| Field | Value |
|-------|--------|
| **Owner** | `music_brain.tier1.midi_pipeline_wrapper` |
| **Input** | `CompleteSongIntent` (nested: song_root, song_intent, technical_constraints) |
| **Output** | `Dict` with `status`, `midi_path`, `chords`, `rule_broken`, `groove_tempo`, `details`; when completed also `arrangement_summary`, `arrangement_section_names`, `melody_contour`, `melody_phrase_structure`, `texture_density`, `temporal_pacing` (for orchestrator/dashboard). |
| **Path** | `process_intent(intent)` → harmony → `generate_midi_from_harmony(harmony, path, tempo_bpm)` |
| **process_intent output** | harmony, groove, arrangement, production, melody, texture, temporal, intent_summary. MIDI path uses harmony + groove tempo; melody/texture/temporal surface in API and in midi_plan for integration. |

Default key/mode when missing: F / major (in tier1). No silent fallback to empty harmony.

---

## 2. LLM → Intent

| Field | Value |
|-------|--------|
| **Owner** | `mcp_workstation.llm_reasoning_engine` |
| **Input** | User text (str) |
| **Output** | `CompleteSongIntent` (nested); optional `image_prompt`, `audio_texture_prompt` |
| **API** | `parse_user_intent(text)`, `generate_image_prompts(intent)`, `generate_audio_texture_prompt(intent)` |

---

## 3. Orchestrator workflow

| Field | Value |
|-------|--------|
| **Owner** | `mcp_workstation.orchestrator` |
| **Input** | `user_intent_text`, `enable_image_gen`, `enable_audio_gen` |
| **Output** | `CompleteSongIntent` (with `midi_plan`, `generated_image_data`, `generated_audio_data` filled) |
| **Phase order** | LLM intent → from_flat if flat else use forensic intent → MIDI pipeline → optional image → optional audio |
| **Locks** | llm, midi_gen, image_gen, audio_gen (timeout 300s) |

Error handling: phase failure sets corresponding result dict to `status: "error"` or `"failed"` with `details`; workflow still returns the intent.

---

## 4. Boot

| Field | Value |
|-------|--------|
| **Owner** | `run_brain.py` |
| **Modes** | penta, orchestrator, gui (stub), check |
| **Check list** | penta_core/ml, mcp_workstation, music_brain/session, music_brain/tier1, kmidi_gui (see BOOT.md) |

Dependency order: penta_core → music_brain.session → music_brain.tier1 → mcp_workstation.

---

## 5. Integration map (who consumes what)

| Consumer | Consumes | Use |
|---------|----------|-----|
| **music_brain.tier1.midi_pipeline_wrapper** | `process_intent(intent)` | harmony + groove tempo → MIDI; arrangement/melody/texture/temporal → summary keys in return dict. |
| **music_brain.api** | `process_intent(intent)` | Full output: harmony, groove, arrangement, production, melody, texture, temporal, intent_summary (serializable). |
| **mcp_workstation.orchestrator** | `MIDIGenerationPipeline.generate_midi()` result | Stores as `complete_intent.midi_plan` (status, midi_path, chords, groove_tempo, arrangement_summary, melody_contour, texture_density, temporal_pacing, etc.). |
| **LLM → Intent** | User text | `llm_reasoning_engine.parse_user_intent()` → `CompleteSongIntent`; orchestrator passes intent to tier1. |
| **Intent → MIDI** | `CompleteSongIntent` | Single path: orchestrator → tier1 → process_intent → harmony → generate_midi_from_harmony. |
| **Body (spectocloud)** | `useMusicBrain.renderSpectocloud()` | Body calls `POST /spectocloud/render`; contract in `KmiDi_CANON/body/hooks/useMusicBrain.ts` (SpectocloudRenderRequest/Response). |

See `docs/INTEGRATION_MAP.md` for flow diagram and cross-module references.

---

## 5b. Visualization — Spectocloud (spine inclusion)

| Field | Value |
|-------|--------|
| **Path (canonical)** | Body: `KmiDi_CANON/body/hooks/useMusicBrain.ts` → API `POST /spectocloud/render` → Brain: `music_brain/visualization/spectocloud.py` (when restored or reimplemented). |
| **Owner (backend)** | `music_brain.visualization.spectocloud` (module not present in active tree; restore from forensic or reimplement). |
| **Input** | `SpectocloudRenderRequest`: `midi_events` or `midi_file_path`; optional `duration`, `emotion_trajectory`, `mode` (static \| animation), `output_path`, `fps`, `frame_idx`, etc. |
| **Output** | `SpectocloudRenderResponse`: `status`, `mode`, `output_path`, `frames`. |
| **Recovery** | Forensic/archive: `music_brain/visualization/spectocloud.py`, `spectocloud_cli.py`; API route must be registered where music_brain API is served. |

---

## 6. Phase order (orchestrator)

1. LLM reasoning (intent parse, image/audio prompts).
2. Build or use `CompleteSongIntent` (from_flat if flat; else use LLM result).
3. MIDI generation (tier1 pipeline).
4. Image generation (optional).
5. Audio generation (optional).

---

## 7. Stub vs failed vs completed

| Status | Meaning |
|--------|---------|
| **stubbed** | Capability not loaded (e.g. model missing); result is placeholder; orchestrator continues. |
| **failed** | Capability ran but returned error (e.g. invalid input, runtime error). |
| **completed** | Capability ran and produced output. |
| **timeout** | Lock or resource acquisition timed out; no output. |
| **error** | Exception or unrecoverable failure; details in `details` field. |

Orchestrator never blocks the workflow on stub; it records status and continues.

---

## 8. Checkpoints and models (governance)

Per engineering governance: checkpoints live under `~/Models` (e.g. `~/Models/checkpoints`); datasets under `~/Datasets`. No large outputs under the repo. See `docs/DATA_AND_TRAINING.md` for paths, run manifest, and training safety. Document model paths in ENV_AND_TMUX or BOOT; default so `run_brain.py check` still runs without models.

---

## 9. Refactor law

When a spine file grows beyond ~400 lines, split by responsibility and keep a single public API for the orchestrator. Spine = `run_brain.py`, `mcp_workstation.orchestrator`, `music_brain.session.intent_processor`, `music_brain.tier1.midi_pipeline_wrapper`, `penta_core.ml.inference`, and their direct single-file call paths. Do not fragment the load-bearing spine; extract helpers into sibling modules and re-export from the main module if needed.

---

## 10. Recovery rule

If there is "no recoverable code path" for a given feature or module, document the recovery path: reference this roadmap (`docs/PROJECT_ROADMAP.md`, `docs/PROJECT_ROADMAP_REIMPLEMENTATION.md`), `FORENSIC_RECOVERY_REPORT.md` (if present), or main/master and `.cursor/rules/recovery-code-path.mdc`. Do not leave recovery undefined; keep one canonical tree and one documented restore path.

**Standard recovery workflow (before recreating code):**

1. Search `docs/GIT_RESTORE_PATHWAYS.md` for known paths and commits
2. Use `git log -S "symbol_name"` to find when symbol was last present
3. Check `docs/.index/symbol_index_canon.tsv` for indexed symbols
4. Search forensic repo if module was in DAiW-Music-Brain: `~/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain`
5. Consult `docs/INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md` for specific module history

**Example recovery:**

```bash
# Find symbol in git history
cd "/Users/seanburdges/Dev/KmiDi MIDI Companion"
git log --all -S "Spectocloud" --oneline | head -10

# Extract from specific commit
git show 6d4d67c5:music_brain/visualization/spectocloud.py > /tmp/spectocloud_restored.py

# Adapt for current layout (KmiDi_CANON/brain/music_brain/...)
# Update imports, test, integrate
```

This workflow is documented in §4.3 of PROJECT_ROADMAP.md.
