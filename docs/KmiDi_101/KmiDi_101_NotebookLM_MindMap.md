# KmiDi 101 — Mind map source

Upload this document to NotebookLM and ask for a **Mind Map** to get a branching overview. Structure: one root (KmiDi 101), three main branches (Docs, Generate path, Intent contract), then file- and code-level detail under each.

---

## KmiDi 101 — Course module

Plain-language guide to the KmiDi project: what each part does, what depends on it, and how it ties together. Written for non-developers; uses real file and folder names.

### Docs (00, 02–09, 10)

- **00_Overview.md** — What KmiDi is; how the pieces fit (big picture).
- **02_Through_09_By_Area.md** — Brain, Intent, generate path, music_brain folders, Tauri, C++ engine, Web UI, build and run.
- **10_Dependency_Map.md** — Who calls whom (main generate path and intent contract).
- **11_Handoff.md** — How to extend the 101 and pass the baton.
- **DISCOVERY_WORKFLOW.md** — How to find "what depends on X" and update the dependency map.

### Generate path (who calls whom)

End-to-end flow from UI click to engines.

#### Web UI

- **IntentBuilder** — Renders intent form and Generate button; calls Music Brain hook.
  - File: `src/components/IntentBuilder.tsx`
  - Calls: `buildGeneratePayload(intent)`, `generateMusic(apiPayload)` from `useMusicBrain()`.
- **useMusicBrain** — Hook that exposes API calls.
  - File: `src/hooks/useMusicBrain.ts`
  - **buildGeneratePayload** (L121) — Converts `CompleteSongIntentRequest` to `GenerateRequest` (body for `/generate`).
  - **generateMusic** (L153) — Sends `POST` to `API_BASE + '/generate'` with JSON body.
  - **generateFromIntent** (L160–162) — Calls `buildGeneratePayload(intent)` then `generateMusic(payload)`.

#### Music Brain API

- **POST /generate** — HTTP endpoint.
  - File: `music_brain/api.py` L1336.
- **generate_music** — FastAPI route handler (L1337).
  - Builds `strict_payload` from request.
  - Validates with **CompleteSongIntentRequest.model_validate** (`music_brain/engine_api/schema.py`).
  - Converts via **_convert_to_intent** (local helper L1403): `GenerateRequest` + validated request → **CompleteSongIntent**.
  - Calls **api.process_song_intent(complete_intent)**.
- **DAiWAPI.process_song_intent** (L699) — Converts and normalizes result.
  - Calls **process_intent(intent)** (L714).
- **process_intent** — Core logic entry.
  - File: `music_brain/session/intent_processor.py` L719.
  - Creates **IntentProcessor(intent)**, returns **processor.generate_all()**.

#### process_intent expanded

- **IntentProcessor.generate_all** (L702) — Returns dict with:
  - **generate_harmony** — IntentProcessor method.
  - **generate_groove** — IntentProcessor method.
  - **generate_arrangement** — IntentProcessor method.
  - **generate_production** — IntentProcessor method.
  - **intent_summary** — L708–714 (mood, tension, narrative, rule_broken, justification).

### Intent contract (shared shape)

Single source of truth for the request shape; kept in sync across TypeScript, Rust, and Python.

#### Source

- **CompleteSongIntentRequest** — Pydantic model; validates body of `/generate`.
  - File: `music_brain/engine_api/schema.py`.
  - Used by: `api.py` (generate_music), `scripts/sync_entities.py`.
- **CompleteSongIntent** — Internal intent representation used by process_intent.
  - File: `music_brain/session/intent_schema.py`.
  - Built by: `_convert_to_intent` in `api.py`.

#### Sync script

- **sync_entities.py** — Reads Pydantic model; writes JSON schema and generates TypeScript and Rust.
  - File: `scripts/sync_entities.py`.
  - Reads: `music_brain.engine_api.schema.CompleteSongIntentRequest`.
  - Writes: `shared_schemas/CompleteSongIntentRequest.json`, `src/types/Intent.ts`, `src-tauri/src/generated/intent.rs`.

#### Generated outputs

- **CompleteSongIntentRequest.json** — Canonical JSON schema; `shared_schemas/`.
- **Intent.ts** — TypeScript types for Intent; `src/types/`; used by useMusicBrain, IntentBuilder.
- **intent.rs** — Rust struct for Intent; `src-tauri/src/generated/`; used by Tauri commands.

### Relationships

- **Generate path** is documented in **10_Dependency_Map**; **Intent contract** feeds validation at **POST /generate** (schema.py validates body in generate_music).
- **Docs** 00 → 02–09 → 10 flow: overview, then by area, then who-calls-whom.
