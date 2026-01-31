# KmiDi Brain — Contracts

Single source of truth for input/output contracts of the load-bearing spine. Changes to a contract require updating this doc and all callers.

**Spine:** `run_brain.py` → `mcp_workstation.orchestrator` / `penta_core` → `music_brain.session` + `music_brain.tier1` → `music_brain.harmony` / `intent_processor` / structure / groove.

---

## 1. Intent → MIDI

| Field | Value |
|-------|--------|
| **Owner** | `music_brain.tier1.midi_pipeline_wrapper` |
| **Input** | `CompleteSongIntent` (nested: song_root, song_intent, technical_constraints) |
| **Output** | `Dict` with `status`, `midi_path`, `chords`, `rule_broken`, `groove_tempo`, `details` |
| **Path** | `process_intent(intent)` → harmony → `generate_midi_from_harmony(harmony, path, tempo_bpm)` |

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

## 5. Phase order (orchestrator)

1. LLM reasoning (intent parse, image/audio prompts).
2. Build or use `CompleteSongIntent` (from_flat if flat; else use LLM result).
3. MIDI generation (tier1 pipeline).
4. Image generation (optional).
5. Audio generation (optional).

---

## 6. Stub vs failed vs completed

| Status | Meaning |
|--------|---------|
| **stubbed** | Capability not loaded (e.g. model missing); result is placeholder; orchestrator continues. |
| **failed** | Capability ran but returned error (e.g. invalid input, runtime error). |
| **completed** | Capability ran and produced output. |
| **timeout** | Lock or resource acquisition timed out; no output. |
| **error** | Exception or unrecoverable failure; details in `details` field. |

Orchestrator never blocks the workflow on stub; it records status and continues.

---

## 7. Checkpoints and models (governance)

Per engineering governance: checkpoints live under `~/Models` (e.g. `~/Models/checkpoints`); datasets under `~/Datasets`. No large outputs under the repo. Document model paths in ENV_AND_TMUX or BOOT; default so `run_brain.py check` still runs without models.
