# KmiDi 101 — Section 10: Dependency Map (Who Calls Whom)

This section is a compact reference for the main “generate” path: each function or module, what it does, what calls it, and what it calls. Use it when you need to trace a dependency or add a new step.

---

## Generate path (main flow)

| Name | What it does | What calls it | What it calls |
|------|----------------|---------------|----------------|
| **IntentBuilder** (React component) | Renders the intent form and Generate button; calls the Music Brain hook to send the request. | App.tsx (via app layout) | useMusicBrain().generateFromIntent |
| **useMusicBrain** (hook, `src/hooks/useMusicBrain.ts`) | Exposes API calls: generateFromIntent, buildGeneratePayload, generateMusic (POST /generate), interrogate, emotions, etc. | IntentBuilder and other components | fetch(API_BASE + '/generate', …), buildGeneratePayload |
| **buildGeneratePayload** | Converts CompleteSongIntentRequest (UI shape) into GenerateRequest (body for /generate). | useMusicBrain.generateFromIntent | — |
| **POST /generate** (HTTP) | Web endpoint that receives the generate request. | Frontend (useMusicBrain.generateMusic), or any HTTP client | — |
| **generate_music** (handler in `music_brain/api.py`) | FastAPI route handler for POST /generate. Validates body, converts to CompleteSongIntent, calls process_song_intent. | FastAPI (when request hits /generate) | CompleteSongIntentRequest (validate), _convert_to_intent, api.process_song_intent |
| **DAiWAPI.process_song_intent** (`music_brain/api.py`) | Converts CompleteSongIntent and calls session process_intent; normalizes result. | generate_music handler | session.intent_processor.process_intent, intent_schema.CompleteSongIntent |
| **process_intent** (`music_brain/session/intent_processor.py`) | Core logic: chord progressions (with rule-breaking), grooves, arrangement, production. | api.process_song_intent (via DAiWAPI) | intent_schema (CompleteSongIntent, enums), structure, harmony, groove, emotion, kelly_companion, etc. |
| **CompleteSongIntentRequest** (Pydantic, `music_brain/engine_api/schema.py`) | Validates the JSON body of /generate (genre, key, structure, instruments, etc.). | api.py (generate_music) | — |
| **CompleteSongIntent** (dataclass, `music_brain/session/intent_schema.py`) | Internal intent representation used by process_intent. | api.py (after conversion from request), intent_processor | — |
| **sync_entities.py** (`scripts/sync_entities.py`) | Reads Pydantic CompleteSongIntentRequest, writes shared_schemas JSON and generates Intent.ts + intent.rs. | Developer (or CI) after schema change | music_brain.engine_api.schema.CompleteSongIntentRequest |

---

## Intent contract (shared shape)

| Name | What it does | What calls it | What it calls |
|------|----------------|---------------|----------------|
| **shared_schemas/CompleteSongIntentRequest.json** | Canonical JSON schema for the request shape. | sync_entities (output), docs, tests | — |
| **src/types/Intent.ts** | TypeScript types for Intent (generated). | useMusicBrain, IntentBuilder, any React code that touches intent | — |
| **src-tauri/src/generated/intent.rs** | Rust struct for Intent (generated). | Tauri commands that build generate requests | — |
| **music_brain/engine_api/schema.py** | Pydantic model CompleteSongIntentRequest; used for validation and for sync_entities. | api.py, scripts/sync_entities.py | — |

---

## Tauri / C++ bridge

| Name | What it does | What calls it | What it calls |
|------|----------------|---------------|----------------|
| **commands.rs** (`src-tauri/src/commands.rs`) | Tauri commands: kelly_brain_*, generate_music, interrogate, etc. | React via invoke() | bridge::kelly_ffi (KellyBrain, get_kelly_brain_manager, …), HTTP for generate_music fallback |
| **kelly_ffi.rs** (`src-tauri/src/bridge/kelly_ffi.rs`) | Rust FFI bindings to KellyFFI C ABI. | commands.rs | libKellyFFI (C functions) |
| **kelly_ffi.cpp / kelly_ffi.h** (`src/bridge/`) | C ABI surface; implements the functions the Rust side calls; talks to KellyCore (C++). | Rust (via dynamic link to libKellyFFI) | C++ engine (src_penta-core, include/penta) |
| **build.rs** (`src-tauri/build.rs`) | Tells Cargo where to find and link libKellyFFI. | cargo (when building Tauri) | — |

---

## Other callers of process_intent / session

| Name | What it does | What calls it | What it calls |
|------|----------------|---------------|----------------|
| **intent_bridge.py** (`music_brain/session/intent_bridge.py`) | process_intent(intent_json: str) → str; JSON in/out bridge. | Other code that needs JSON interface to process_intent | session intent_processor.process_intent |
| **intent.py** (`music_brain/session/intent.py`) | process_intent_tool(intent_json); async tool. | Agents/tools that need to run intent from JSON | api.process_song_intent |

---

You can extend this map as you deepen other sections: add one row per important function or module, and keep “what calls it” / “what it calls” up to date so the next person can follow the chain.
