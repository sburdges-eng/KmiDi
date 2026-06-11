# Whole-Repository Code Review — KmiDi / iDAW

Historical note
- This review preserves a Tauri-era repo interpretation and is not a current architecture authority document.
- Use it as historical analysis only.
- Treat commands and shell classifications below, including `npm run dev:tauri`, as historical/legacy unless revalidated against the current repo scripts.
- When it conflicts with the current repo architecture, follow the 2026 authority set headed by `docs/ARCHITECTURE.md` and `docs/REPO_MODULE_MAP.md`.

Evidence-bound code review. Zero tolerance for hallucinated findings. Every finding has exact file, line(s), scenario, impact, and minimal fix.

**Scope:** React (Vite) frontend, Music Brain FastAPI, Tauri/Rust shell, C++ (KellyCore/KellyFFI), schemas, tests, CI, scripts, config. KmiDi-specific: local-service boundaries, DAW/plugin stability, realtime safety, FFI containment, human-approval boundaries.

---

## A. REPO MAP

### Apps / packages / modules

| Area | Path | Purpose |
|------|------|---------|
| Frontend | `src/` | React (Vite) — AppConsole (default entry in main.tsx), IntentBuilder, SideA (Transport, Mixer, Timeline, VUMeter), SideB (EmotionWheel, GhostWriter, Interrogator), LyricPanel, SpectoCloudPanel, MusicCustomizer |
| Tauri shell | `engine/intent_ir/` | Rust app: commands (Kelly FFI + Music Brain HTTP fallback), bridge (kelly_ffi.rs, musicbrain.rs), state, events, generated intent, intent_ir |
| Music Brain API | `music_brain/` | FastAPI app in `api.py`; engine_api/schema (CompleteSongIntentRequest); session, structure, voice, groove, visualization, etc. |
| C++ engine | `src/` (C++), `engine/`, `include/` | KellyCore, KellyBrain, IntentPipeline, Wound/IntentResult; plugin (VST3/CLAP) in `src/plugin/` |
| FFI bridge | `src/bridge/kelly_ffi.cpp`, `kelly_ffi.h` | C ABI for Rust; KellyBrainWrapper, serialize_intent_result, parse_wound_json |
| Shared contract | `shared_schemas/` | CompleteSongIntentRequest.json, CompleteSongIntent.json |
| Sync | `scripts/sync_entities.py` | Generates `src/types/Intent.ts`, `engine/intent_ir/src/generated/intent.rs` from schema + Pydantic |
| Tests | `tests/` | unit/ (api_schema, api_audit_fixes, intent_processor, …), integration/, cpp/, rust/, e2e/, performance/ |
| Scripts | `scripts/` | load-env.sh, sync_entities.py, dev-setup.sh, build-full-stack.sh, acquire/, mcp/, training/ |

### Runtimes / languages

- **Node 20+**: Vite, React, npm scripts  
- **Python 3.9+** (3.11 in CI): FastAPI, uvicorn, music_brain  
- **Rust (stable)**: Tauri 2, idaw_lib  
- **C++20**: KellyCore, KellyFFI, plugins (JUCE, Qt6 when enabled)  
- **CMake 3.27+**: Root CMakeLists.txt, engine, rt_harness, src_penta-core, bindings  

### Entrypoints

| Entrypoint | File / command |
|------------|----------------|
| Frontend (default) | `src/main.tsx` → AppConsole |
| Frontend (legacy) | App.tsx (not mounted per main.tsx comment) |
| Music Brain API | `python -m music_brain.api` or `uvicorn music_brain.api:app` |
| Historical Tauri desktop shell | historical/legacy review-era shell assumption; `npm run dev:tauri` is not present in the current `package.json` |
| C++ KellyFFI | Built by CMake; loaded by Rust via libloading/dylib |

### Important data flows

1. **Generate flow:** React (IntentBuilder/AppConsole) → either `invoke()` Tauri commands (generate_music, etc.) or `fetch(API_BASE + '/generate')` when VITE_KMIDI_USE_API=true → Tauri uses Kelly FFI if initialized else Music Brain HTTP → FastAPI `/generate` → CompleteSongIntentRequest validation → process_song_intent → response with result/lyrics/midi_path.  
2. **Contract flow:** shared_schemas/CompleteSongIntentRequest.json + music_brain/engine_api/schema.py → sync_entities.py → Intent.ts + intent.rs; CI verifies git diff on these.  
3. **Audio classify:** POST /audio/classify with audio_path → _resolve_audio_path_sandbox(audio_path) → only paths under KMIDI_AUDIO_SERVE_ROOT allowed.

### External integrations

- CORS allowlist: localhost:1420, 127.0.0.1:1420, tauri://localhost, localhost:5173.  
- Music Brain HTTP: Tauri musicbrain.rs uses `MUSIC_BRAIN_API_URL` (default 127.0.0.1:8000).  
- Frontend API base: `VITE_API_BASE` (default 127.0.0.1:8000).  
- Optional: Freesound, Ollama, ONNX, JEPA/SageMaker (env-configured).

### Highest-risk areas

- **FFI boundary:** C++ kelly_ffi.cpp (JSON serialization without escaping, parse_wound_json missing intensity).  
- **Path handling:** /audio/* sandboxed; spectocloud midi_file_path and output_path not sandboxed.  
- **Config write:** PUT /config/humanizer and spectocloud output_path can write to disk.  
- **Tauri:** All invoke handlers registered without capability files in repo (Tauri 2 default permissions).  
- **Realtime/audio:** KellyCore/plugin code not fully traced for heap/locks on audio thread in this pass.

---

## B. REVIEW COVERAGE

### Inspected paths

| Path | Status |
|------|--------|
| package.json, vite.config.ts, src/main.tsx, src/AppConsole.tsx, src/hooks/useMusicBrain.ts, src/components/IntentBuilder.tsx | Inspected |
| music_brain/api.py (full file) | Inspected |
| music_brain/engine_api/schema.py | Inspected |
| shared_schemas/*.json, scripts/sync_entities.py | Inspected |
| engine/intent_ir/Cargo.toml, build.rs, lib.rs, main.rs, commands.rs, bridge/musicbrain.rs, bridge/kelly_ffi.rs | Inspected |
| engine/intent_ir/tauri.conf.json | Inspected |
| src/bridge/kelly_ffi.cpp, kelly_ffi.h (repo root C++ bridge) | Inspected |
| CMakeLists.txt, pyproject.toml | Inspected |
| .github/workflows/ci.yml, ci-python.yml | Inspected |
| .env.example, scripts/load-env.sh, docs/ENVIRONMENT.md | Inspected |
| tests/unit/test_api_schema.py | Inspected |
| src/engine/IntentProcessor.h (Wound struct), IntentPipeline usage | Inspected |

### Skipped paths + reason

| Path | Reason |
|------|--------|
| `external/JUCE`, `build/`, `dist/`, `node_modules/` | Generated or third-party; not reviewed for app logic |
| `music_brain/**` (full tree) | Only api.py and engine_api/schema.py inspected in depth; rest partial/semantic search |
| `scripts/**` (full tree) | Only sync_entities.py, load-env.sh inspected in depth |
| `src/engine/*` (C++ beyond Wound/IntentResult) | No full line-by-line; FFI and headers only |
| `src/plugin/*`, rt_harness, src_penta-core | Not inspected for audio-thread/realtime in this pass |
| KmiDi_PROJECT, frontend/, tools/, experiments/ | Alternate or experimental; not part of main app entrypoints |
| All remaining tests (e2e, integration, cpp, rust) | Not executed or traced in this review |

### Unresolved unknowns

- Whether Tauri 2 capability/permission files exist outside repo or are default-allow.  
- Full audio callback path in KellyCore/plugin for heap/lock usage.  
- Exact list of files written by process_song_intent (temp MIDI/audio under tempdir).

---

## C. CHECKED / NO EVIDENCE FOUND

- **Auth bypass:** No auth layer; API and Tauri are local-only by binding (127.0.0.1 / Tauri origin). No evidence of auth bypass.  
- **Secret leakage:** No hardcoded API keys; .env.example placeholders only; VITE_* are build-time.  
- **Path traversal on /audio/:** `/audio/{file_path:path}` and `/audio/classify` use _audio_serve_root and _resolve_audio_path_sandbox; checked and no evidence of bypass.  
- **Client/server contract drift:** CI runs sync_entities.py and `git diff --exit-code` on Intent.ts and intent.rs; schema tests in test_api_schema.py.  
- **InterrogateRequest.message / LyricsRequest.lyrics:** Both have max_length (4096 and 32768) in api.py; no unbounded body.  
- **load-env.sh = in value:** Script uses key="${line%%=*}" and val="${line#*=}" and export "$key=$val"; first = is separator. No evidence of E2 from prior audit.  
- **Duplicate pydantic:** pyproject.toml lists pydantic once in dependencies (line 25); no duplicate observed in current file.

---

## D. FINDINGS

### F1 — FFI: journey intensity from JSON ignored

- **ID:** F1  
- **Severity:** Medium  
- **Category:** Correctness / contract  
- **File:** `src/bridge/kelly_ffi.cpp`  
- **Lines:** 197–248 (parse_wound_json), 318–334 (kelly_brain_from_journey)  
- **Problem:** `parse_wound_json` sets only `wound.urgency` from JSON ("urgency" or "intensity"); it never sets `wound.intensity`. `kelly::Wound` has `float intensity = 0.7f` (IntentProcessor.h:28). In `kelly_brain_from_journey`, `current.intensity = w.intensity` therefore always uses the default 0.7, not the client-supplied value.  
- **Evidence:** parse_wound_json extracts urgency/intensity into wound.urgency only; no assignment to wound.intensity. from_journey uses w.intensity for SideA/SideB.  
- **Realistic scenario:** Client sends journey JSON with "intensity": 0.2; engine uses 0.7 for blending.  
- **Root cause:** Missing assignment from parsed urgency/intensity to wound.intensity.  
- **Runtime/product impact:** Journey-based intent always uses default intensity; emotional scaling from UI/API is ignored.  
- **Minimal safe fix:** In parse_wound_json, after setting wound.urgency, set wound.intensity = wound.urgency (or parse an explicit "intensity" field and use it for wound.intensity).  
- **Tests to add/update:** Unit test that from_journey (or equivalent) returns different results for different intensity in input JSON.  
- **Confidence:** High  

### F2 — FFI: JSON string escaping in serialization

- **ID:** F2  
- **Severity:** Medium  
- **Category:** Correctness / security  
- **File:** `src/bridge/kelly_ffi.cpp`  
- **Lines:** 68–136 (serialize_intent_result), 141–193 (serialize_generated_midi), 204–248 (parse_wound_json output fields)  
- **Problem:** JSON is built via ostringstream with raw string values (e.g. `json << "  \"core_wound\": \"" << result.sourceWound.description << "\""`). If description, desire, expression, or rule-break text contains `"` or `\`, the output is invalid JSON and can break Rust/client parsing or inject structure.  
- **Evidence:** No escaping of description, desire, expression, productionNotes, ruleBreaks[].description/.justification in serialize_intent_result.  
- **Realistic scenario:** User enters emotion text with a quote; FFI returns malformed JSON; Rust or frontend fails to parse.  
- **Root cause:** Manual JSON construction without escaping.  
- **Runtime/product impact:** Parse errors or possible JSON injection.  
- **Minimal safe fix:** Use a proper JSON library (e.g. nlohmann/json) for serialization, or implement a small escape helper for string values (escape `\` and `"`).  
- **Tests to add/update:** Test that from_text with description containing `"` and `\` returns valid JSON and round-trips.  
- **Confidence:** High  

### F3 — Spectocloud: path traversal on midi_file_path and output_path

- **ID:** F3  
- **Severity:** High  
- **Category:** I/O and trust boundaries  
- **File:** `music_brain/api.py`  
- **Lines:** 1330–1333 (midi_file_path), 1371–1375, 1389–1393 (output_path)  
- **Status:** **Already mitigated in current code.** `render_spectocloud` uses `_resolve_audio_path_sandbox(payload.midi_file_path)` before `_parse_midi_file` and `_resolve_audio_path_sandbox(payload.output_path)` for both static and animation output paths. Paths outside `_audio_serve_root` are rejected with HTTP 400.  
- **Evidence:** Inspected api.py lines 1330–1394; both midi_file_path and output_path are passed through _resolve_audio_path_sandbox.  
- **No code change required.**  
- **Confidence:** High  

### F4 — Tauri Music Brain fallback: wrong env var name

- **ID:** F4  
- **Severity:** Low  
- **Category:** Config / env drift  
- **File:** `engine/intent_ir/src/bridge/musicbrain.rs`  
- **Lines:** 9–14  
- **Problem:** API base URL is read from `MUSIC_BRAIN_API_URL`. `.env.example` and docs use `KMIDI_API_URL`. If only KMIDI_API_URL is set, Tauri’s HTTP fallback to Music Brain will use default 127.0.0.1:8000, not the user’s URL.  
- **Evidence:** musicbrain.rs: `env::var("MUSIC_BRAIN_API_URL")`; .env.example: `KMIDI_API_URL=http://127.0.0.1:8000`.  
- **Realistic scenario:** User sets KMIDI_API_URL to a different host/port; Tauri fallback still uses 127.0.0.1:8000.  
- **Root cause:** Naming inconsistency between Rust and env docs.  
- **Runtime/product impact:** Fallback requests go to wrong host unless user sets MUSIC_BRAIN_API_URL.  
- **Minimal safe fix:** In musicbrain.rs, check both MUSIC_BRAIN_API_URL and KMIDI_API_URL (e.g. prefer MUSIC_BRAIN_API_URL, fall back to KMIDI_API_URL), and document both in .env.example and ENVIRONMENT.md.  
- **Tests to add/update:** None required for env var name.  
- **Confidence:** High  

---

## E. RISK MATRIX

| Subsystem | Failure mode | Likelihood | Impact | Why |
|-----------|---------------|------------|--------|-----|
| FFI (kelly_ffi.cpp) | Invalid JSON / wrong intensity | Medium | Medium | Manual JSON and missing intensity field (F1, F2) |
| API (spectocloud) | Path traversal read/write | Medium | High | No sandbox on midi_file_path/output_path (F3) |
| Tauri (musicbrain) | Wrong API URL in fallback | Low | Low | Env var name mismatch (F4) |
| API (/generate) | Contract mismatch | Low | Medium | Mitigated by schema validation and sync_entities CI |
| C++ plugin/realtime | Heap/lock on audio thread | Unknown | High | Not fully audited in this pass |

---

## F. CONTRACT DRIFT TABLE

| Producer | Consumer | Contract | Drift found | Fix |
|----------|----------|----------|-------------|-----|
| shared_schemas/CompleteSongIntentRequest.json + engine_api/schema | sync_entities.py → Intent.ts, intent.rs | CompleteSongIntentRequest fields | CI verifies; no drift observed | — |
| FastAPI GenerateRequest / EmotionalIntent | api.py generate_music → CompleteSongIntentRequest | structure, instruments, key_mode, tempo | Mapped explicitly; validation on strict_payload | — |
| C++ IntentResult (FFI) | Rust kelly_ffi.rs | JSON shape from serialize_intent_result | F2: invalid JSON if strings contain " or \ | Escape or use JSON lib |
| Journey JSON (client) | kelly_ffi parse_wound_json → Wound | intensity/urgency | F1: intensity not parsed | Set wound.intensity from JSON |

---

## G. CONFIG / ENV MATRIX

| Variable / config | Defined where | Used where | Required? | Default behavior | Risk |
|-------------------|---------------|------------|-----------|------------------|------|
| VITE_KMIDI_USE_API | Build-time (Vite) | src/hooks/useMusicBrain.ts | No | Unset → external API disabled | Doc in ENVIRONMENT.md; .env.example could add |
| VITE_API_BASE | Build-time | useMusicBrain.ts | No | http://127.0.0.1:8000 | Safe |
| KMIDI_API_URL | .env.example | Docs / scripts | No | 127.0.0.1:8000 | — |
| MUSIC_BRAIN_API_URL | Not in .env.example | engine/intent_ir/bridge/musicbrain.rs | No | 127.0.0.1:8000 | F4: name mismatch with KMIDI_API_URL |
| KMIDI_AUDIO_SERVE_ROOT | Not in .env.example | music_brain/api.py | No | tempfile.gettempdir() | Document; narrow dir recommended |
| TAURI_DEV_HOST, TAURI_PLATFORM | .env.example | vite.config.ts | No | localhost / macos | Safe |

---

## H. SECURITY SURFACES

| Entrypoint | Trust boundary | Protection present | Gap | Evidence |
|------------|----------------|--------------------|-----|----------|
| FastAPI (127.0.0.1:8000) | Network (local) | CORS allowlist, no auth | Spectocloud path traversal (F3) | api.py render_spectocloud |
| /audio/* | Path input | _audio_serve_root, relative_to | None in reviewed code | _resolve_audio_path_sandbox, serve_audio |
| Tauri invoke | Frontend → Rust | Tauri origin | No capability files in repo; default allowlist | main.rs invoke_handler |
| Tauri → Music Brain HTTP | Rust → local API | MUSIC_BRAIN_API_URL | F4 env name | musicbrain.rs |
| Kelly FFI (dylib) | Rust ↔ C++ | Mutex on wrapper, null checks | F1, F2 (logic/JSON) | kelly_ffi.cpp |

---

## I. TEST GAPS

| Subsystem | Missing scenario | Likely failure | Test to add |
|-----------|-------------------|----------------|-------------|
| FFI journey | Journey JSON with intensity != 0.7 | Intensity ignored (F1) | Test from_journey intensity in C++ or Rust |
| FFI serialization | description/desire with " and \ | Invalid JSON (F2) | Test from_text round-trip with special chars |
| API spectocloud | midi_file_path outside sandbox | Path traversal (F3) | Test 400 for path outside root; output under root only |
| Tauri commands | generate_music when Kelly not initialized | Fallback to HTTP | Integration test with API URL env |

---

## J. FIX PLAN

### Fix now (correctness / security)

- **F3:** Already fixed: spectocloud uses _resolve_audio_path_sandbox for midi_file_path and output_path. No change needed.

### Safe automated fixes

- **F4:** In engine/intent_ir/src/bridge/musicbrain.rs, read KMIDI_API_URL if MUSIC_BRAIN_API_URL unset; update .env.example and ENVIRONMENT.md to document both.  
  - Affected files: musicbrain.rs, .env.example, docs/ENVIRONMENT.md.  
  - Risk: Low.  
  - Validation: Run Tauri with only KMIDI_API_URL set; confirm fallback hits correct host.

### Needs architecture/product decision

- **F1:** Add wound.intensity from JSON in parse_wound_json (and possibly align Wound.urgency vs .intensity semantics across C++).  
- **F2:** Introduce a JSON library or escape helper in kelly_ffi.cpp; ensure all string fields are escaped.

### Nice-to-have cleanup

- Document KMIDI_AUDIO_SERVE_ROOT and VITE_KMIDI_USE_API in .env.example and ENVIRONMENT.md.  
- Add capability/permission manifest for Tauri if moving to explicit allowlist.

---

## K. PATCH ORDER

1. **Correctness / security blockers:** F3 already mitigated (spectocloud sandbox in place).  
2. **Contract mismatches:** F1 (intensity), F2 (JSON escaping).  
3. **Data / state / config safety:** F4 (env var).  
4. **CI / test protection:** Add tests for F1–F3.  
5. **Cleanup:** Env/docs.  
6. **Architecture-decision items:** Tauri capabilities, FFI JSON library.

---

## L. HONESTY NOTES

- **Not reviewed deeply enough to claim safe:** Full music_brain tree (only api.py and engine_api/schema.py in depth); all C++ engine and plugin code (only FFI bridge and Wound struct); runtime behavior of penta_core, rt_harness; all e2e and integration tests (not run); scripts beyond sync_entities and load-env.  
- **Native surfaces needing direct inspection:** Audio callback path in KellyCore and plugin (heap, locks, logging on realtime thread); full plugin host interaction and buffer/sample-rate assumptions.  
- **Full-repo review:** Not achieved. Review is systematic but selective: entrypoints, contract paths, API and FFI boundaries, and high-risk path handling were prioritized; large subtrees (e.g. music_brain subpackages, C++ engine internals, all scripts) were not fully file-by-file.

---

*End of report. Apply only high-confidence, low-risk fixes after this review; re-run tests and typecheck after each batch.*
