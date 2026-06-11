# Whole-Repository Code Review — KmiDi / iDAW

Historical note
- This review preserves a Tauri-era architecture reading and is not a current authority document.
- Use it as historical analysis only.
- Treat commands and shell classifications below, including `npm run dev:tauri`, as historical/legacy unless revalidated against the current repo scripts.
- When it conflicts with the current repo architecture, follow the 2026 authority set beginning with `docs/ARCHITECTURE.md`.

Evidence-bound code review: React/Vite frontend, FastAPI Music Brain API, Tauri/Rust shell, C++ (KellyCore/KellyFFI). Zero tolerance for hallucinated findings; every finding has file path, line numbers, scenario, impact, and minimal fix.

---

## A. REPO MAP

### Apps / packages / modules
| Area | Path | Purpose |
|------|------|---------|
| Frontend | `src/` | React app (AppConsole entry, IntentBuilder, SideA/SideB, hooks, types) |
| Tauri shell | `engine/intent_ir/` | Rust app (commands, bridge/kelly_ffi, main.rs worker, capabilities) |
| Music Brain API | `music_brain/` | FastAPI app (api.py, engine_api, session, harmony, groove, etc.) |
| Shared contract | `shared_schemas/` | CompleteSongIntentRequest.json → sync to TS/Rust |
| Scripts | `scripts/` | sync_entities.py, dev-setup.sh, load-env.sh, acquire/, build-full-stack.sh |
| Tests | `tests/` | unit (test_api_schema.py, test_api_audit_fixes.py), integration, e2e, fixtures |
| C++ (root) | `engine/`, `include/`, `cmake/` | Kelly core, FFI; build via root CMakeLists.txt |
| Legacy/other | `KmiDi_FINAL/`, `KmiDi_PROJECT/`, `frontend/` | Alternate or legacy trees; not primary entrypoints |

### Runtimes / languages
- **Node 20**: Vite 7, React 19, TypeScript ~5.8
- **Python 3.9+** (3.11 in CI): FastAPI, uvicorn, pydantic
- **Rust (stable)**: Tauri 2, tokio
- **C++20**: KellyCore, KellyFFI (CMake 3.27+, Ninja)

### Entrypoints
| Entrypoint | File | How |
|------------|------|-----|
| Frontend | `src/main.tsx` | Mounts AppConsole (not App.tsx) |
| API | `music_brain/api.py` | `uvicorn music_brain.api:app` or `python -m music_brain.api` (host 127.0.0.1 in _main()) |
| Historical Tauri shell | `engine/intent_ir/` | historical/legacy review-era shell path; `npm run dev:tauri` and `npm run tauri build` are not present in the current `package.json` |
| Schema sync | `scripts/sync_entities.py` | Updates Intent.ts, intent.rs, Python from shared_schemas |

### Important data flows
- **Generate (Tauri)**: React → invoke `start_generation` → main.rs worker → parsed CompleteSongIntentRequest → kelly_ffi manager → from_text + generate_midi → gen-result event.
- **Generate (HTTP)**: React (useMusicBrain when VITE_KMIDI_USE_API=true) → POST /generate → GenerateRequest → CompleteSongIntentRequest.model_validate → api.process_song_intent → response.
- **Audio classify/voice**: POST body `audio_path` → _resolve_audio_path_sandbox (KMIDI_AUDIO_SERVE_ROOT) → file read.
- **Spectocloud**: POST body `midi_file_path`, `output_path` → _parse_midi_file(midi_file_path), render to output_path (no sandbox).

### External integrations
- CORS: localhost:1420, 127.0.0.1:1420, tauri://localhost, localhost:5173.
- API keys via env: OPENAI, ANTHROPIC, GOOGLE, XAI, GITHUB, FREESOUND (see .env.example).
- JUCE submodule (external/JUCE) for C++/plugins.

### Highest-risk areas
1. **music_brain/api.py** — path handling (audio serve, spectocloud, humanizer config), request validation, 500 detail.
2. **engine/intent_ir** — Tauri commands (kelly_brain_initialize(data_path)), fs/dialog/opener capabilities, invoke surface.
3. **Rust ↔ C++ FFI** — engine/intent_ir/bridge/kelly_ffi.rs, C strings, error propagation.
4. **Config writes** — PUT /config/humanizer (config/humanizer.json), atomic write via tempfile+replace.

---

## B. REVIEW COVERAGE

### Inspected paths
- `package.json`, `pyproject.toml`, `vite.config.ts`, `tsconfig` (refs), `.env.example`, `AGENTS.md`
- `.github/workflows/ci.yml`, `.github/workflows/ci-python.yml`
- `music_brain/api.py` (full read)
- `music_brain/engine_api/schema.py`, `shared_schemas/CompleteSongIntentRequest.json`
- `scripts/sync_entities.py` (partial)
- `src/main.tsx`, `src/AppConsole.tsx`, `src/components/IntentBuilder.tsx`, `src/hooks/useMusicBrain.ts`
- `engine/intent_ir/src/lib.rs`, `engine/intent_ir/src/main.rs`, `engine/intent_ir/src/commands.rs`, `engine/intent_ir/src/bridge/kelly_ffi.rs` (partial)
- `engine/intent_ir/capabilities/default.json`, `engine/intent_ir/tauri.conf.json`
- `CMakeLists.txt` (root, limit 80)
- `tests/unit/test_api_schema.py`, `tests/unit/test_api_audit_fixes.py`
- `docs/CROSS_CUTTING_AUDIT_REPORT.md` (reference), `docs/ENVIRONMENT.md` (limit)

### Skipped paths + reason
- **engine/, include/ (root C++)** — Realtime/audio-thread and DAW plugin code not traced line-by-line; RTMemoryPool/GrooveEngine etc. live in KmiDi_FINAL/KmiDi_PROJECT in this repo; root build uses BUILD_KELLY_CORE/KELLY_FFI. Marked as “not reviewed deeply” for realtime safety.
- **KmiDi_FINAL/, KmiDi_PROJECT/, frontend/** — Alternate/legacy trees; not the primary stack per AGENTS.md.
- **music_brain/** (except api.py, engine_api/schema.py) — Large package; only API surface and schema reviewed; internal modules (harmony, groove, session, etc.) not fully audited.
- **Full C++ KellyFFI implementation** — FFI declarations and Rust wrapper inspected; C++ implementation under external/KmiDi_FINAL not inspected.
- **Tauri plugin fs scope** — “fs:default” permission present; exact scope (allowlist) not resolved (plugin docs / node_modules not read).
- **bootstrap.sh** — Root and scripts/bootstrap.sh both exist; CI runs `./bootstrap.sh` from repo root; root script read (limit).

### Unresolved unknowns
- Whether root `engine/` or another tree is the actual KellyFFI C++ source used in v1_shell_build (CI builds KellyFFI from root CMake).
- Exact Tauri fs:default scope (which directories are allowlisted).
- Full coverage of music_brain internal imports used by api.py (e.g. render_plan_to_midi, TherapySession) for path/validation behavior.

---

## C. CHECKED / NO EVIDENCE FOUND

- **Auth bypass** — No auth layer; API and Tauri are documented/local-only; CORS limits origins; no evidence of auth bypass (no auth to bypass).
- **Secret leakage** — No hardcoded API keys in inspected code; keys via env; .env.example placeholders only.
- **Path traversal on /audio/** — GET /audio/{file_path:path} and POST /audio/classify, /voice/classify use _resolve_audio_path_sandbox; Path.resolve() and relative_to(_audio_serve_root) enforce sandbox; tests in test_api_audit_fixes.py confirm reject outside sandbox.
- **Client/server contract (Generate)** — GenerateRequest (API) maps to CompleteSongIntentRequest (engine); sync_entities.py keeps TS/Rust in sync; CI runs sync and git diff on Intent.ts and intent.rs; no drift observed.
- **InterrogateRequest / LyricsRequest max length** — InterrogateRequest.message max_length=4096, LyricsRequest.lyrics max_length=32768 in api.py (lines 979, 984); no missing validation.
- **Humanizer config write** — Atomic write via tempfile.mkstemp + os.replace; cleanup on exception; no evidence of partial state or race.
- **IntentBuilder Tauri commands** — Only start_generation and cancel_generation invoked with fixed names; no user-controlled command name.
- **start_generation intent size** — main.rs caps intent_json.len() to 64_000 and validates JSON; no unbounded payload.

---

## D. FINDINGS

### F1 — Spectocloud path traversal (arbitrary file read and write)

| Field | Value |
|-------|--------|
| **ID** | F1 |
| **Severity** | High |
| **Category** | I/O and trust boundaries / path handling |
| **File** | `music_brain/api.py` |
| **Lines** | 833–834 (read), 868–877 (write) |
| **Problem** | POST /spectocloud/render accepts `midi_file_path` and `output_path` from the request body and uses them without sandboxing. `_parse_midi_file(Path(payload.midi_file_path))` reads any server-accessible file; `output_path` (or default under tempfile.gettempdir()) is used for render output; if client supplies `output_path`, the server can write to an arbitrary path. |
| **Evidence** | In render_spectocloud: `if payload.midi_file_path: parsed_events, parsed_duration = _parse_midi_file(Path(payload.midi_file_path))` (833–834). For static mode: `out_path = payload.output_path or str(Path(tempfile.gettempdir()) / "spectocloud_frame.png")` and `specto.render_static_frame(..., output_path=out_path)` (868–875). Same pattern for animation (876–877). No resolve/relative_to check against a dedicated root. |
| **Realistic scenario** | Attacker with access to API (e.g. same network or misconfiguration) sends midi_file_path="/etc/passwd" (or another sensitive path) to probe; or output_path="/tmp/overwrite_cron" to write a file. |
| **Root cause** | Spectocloud endpoint was not given the same path-restriction pattern as /audio/ and classify. |
| **Runtime/product impact** | Information disclosure (read arbitrary files), potential overwrite of server files (arbitrary write). |
| **Minimal safe fix** | Introduce a single spectocloud root (e.g. same as KMIDI_AUDIO_SERVE_ROOT or a new env KMIDI_SPECTOCLOUD_ROOT). For midi_file_path: resolve Path and require relative_to(spectocloud_root); for output_path: if provided, resolve and require relative_to(spectocloud_root); else keep default under tempdir. Reject with 400 if outside root. |
| **Tests to add/update** | Add tests in tests/unit (or test_api_audit_fixes.py): reject midi_file_path outside root; reject output_path outside root; accept paths under root. |
| **Confidence** | High |

---

## E. RISK MATRIX

| Subsystem | Failure mode | Likelihood | Impact | Why |
|-----------|--------------|------------|--------|-----|
| Music Brain API | Spectocloud path traversal | Medium | High | User-controlled paths; no sandbox (F1). |
| Music Brain API | Humanizer config overwrite | Low | Medium | Atomic write and reload; only local config. |
| Music Brain API | Audio classify/voice | Low | High | Sandboxed; tests confirm. |
| Tauri | kelly_brain_initialize(data_path) | Low | Medium | Trusted desktop; data_path passed to C++; no path validation in Rust. |
| Tauri | fs:default scope | Unknown | Medium | Scope not verified; if broad, could allow broad file access. |
| FFI (Rust/C++) | C string / panic | Low | High | CString used; null checks; panic not fully traced in C++ side. |
| Frontend | API disabled (VITE_KMIDI_USE_API) | N/A | Low | By design; no finding. |
| CI | bootstrap.sh path | Low | Low | Root ./bootstrap.sh exists and runs; scripts/bootstrap.sh also exists. |

---

## F. CONTRACT DRIFT TABLE

| Producer | Consumer | Contract | Drift found | Fix |
|----------|----------|----------|-------------|-----|
| shared_schemas/CompleteSongIntentRequest.json | sync_entities.py → Intent.ts, intent.rs, Python schema | JSON schema → TS interface, Rust structs, Pydantic | CI verifies git diff; none observed | — |
| GenerateRequest (API) | process_song_intent | technical/structure/instruments mapped to CompleteSongIntentRequest | None | — |
| EmotionalIntent (API) | CompleteSongIntent | core_wound, core_desire, emotional_intent, technical.* | Mapped in handler; optional fields handled | — |

---

## G. CONFIG / ENV MATRIX

| Variable/config | Defined where | Used where | Required? | Default behavior | Risk |
|-----------------|----------------|------------|-----------|------------------|------|
| KMIDI_AUDIO_SERVE_ROOT | Env (not in .env.example) | api.py _audio_serve_root, _resolve_audio_path_sandbox | No | tempfile.gettempdir() | If set broad, larger serve surface; path traversal still blocked by relative_to. |
| KMIDI_API_URL | .env.example | Frontend/docs | No | http://127.0.0.1:8000 | — |
| VITE_KMIDI_USE_API | Build-time | useMusicBrain.ts | No | Unset → API disabled | Document in ENVIRONMENT.md. |
| VITE_API_BASE | Build-time | useMusicBrain.ts | No | http://127.0.0.1:8000 | — |
| KELLY_MODELS_PATH | .env.example | C++/docs | No | ./models | — |

---

## H. SECURITY SURFACES

| Entrypoint | Trust boundary | Protection present | Gap | Evidence |
|------------|-----------------|--------------------|-----|----------|
| POST /generate | Network → API | CORS, Pydantic, structure/instrument validation | None | CORS middleware; CompleteSongIntentRequest.model_validate. |
| GET /audio/{path} | Network → API | Path resolve + relative_to(_audio_serve_root) | None | api.py 656–664, 668–674. |
| POST /audio/classify, /voice/classify | Network → API | _resolve_audio_path_sandbox | None | test_api_audit_fixes.py. |
| POST /spectocloud/render | Network → API | None for paths | Arbitrary read/write (F1) | api.py 833–834, 868–877. |
| Tauri invoke | WebView → Rust | Commands allowlisted in generate_handler | kelly_brain_initialize(data_path) unvalidated path | main.rs, commands.rs. |
| PUT /config/humanizer | Network → API | Atomic write, normalized payload | Local-only assumed (no auth) | api.py 806–829. |

---

## I. TEST GAPS

| Subsystem | Missing scenario | Likely failure | Test to add |
|-----------|------------------|----------------|-------------|
| API | Spectocloud midi_file_path outside sandbox | 400 with clear detail | test_spectocloud_rejects_midi_path_outside_root |
| API | Spectocloud output_path outside sandbox | 400 | test_spectocloud_rejects_output_path_outside_root |
| API | Spectocloud accepts paths under root | 200 or 400 for other reason | test_spectocloud_accepts_paths_under_root |
| Tauri | kelly_brain_initialize with path validation | N/A (trusted desktop) | Optional: reject path outside app dir or env-configured root. |
| Contract | GenerateResponse shape vs frontend | Frontend expects result.*, midi_path, etc. | Contract test or e2e that asserts response shape. |

---

## J. FIX PLAN

### Fix now
- **F1 (Spectocloud path sandbox)** — Affected file: `music_brain/api.py`. Change: add spectocloud root (reuse KMIDI_AUDIO_SERVE_ROOT or new env); resolve midi_file_path and output_path and require relative_to(root); return 400 if outside. Risk: low. Validation: run pytest tests/unit/test_api_audit_fixes.py and new spectocloud tests.

### Safe automated fixes
- Add KMIDI_AUDIO_SERVE_ROOT (and if added, KMIDI_SPECTOCLOUD_ROOT) to .env.example and ENVIRONMENT.md.
- Document VITE_KMIDI_USE_API in ENVIRONMENT.md (and .env.example if missing).

### Needs architecture/product decision
- Tauri kelly_brain_initialize(data_path): whether to restrict data_path to an app-configured or env-configured root.
- Tauri fs:default scope: confirm or narrow scope in capability config.

### Nice-to-have cleanup
- Remove duplicate pydantic in pyproject.toml if still present (see CROSS_CUTTING_AUDIT_REPORT D1).
- CI: ensure bootstrap.sh path is unambiguous (root vs scripts/).

---

## K. PATCH ORDER

1. **Correctness / security blockers** — F1 (Spectocloud path sandbox).
2. **Contract mismatches** — None identified.
3. **Data / state / config safety** — Document env vars; humanizer already atomic.
4. **CI / test protection** — Add spectocloud path tests after F1 fix.
5. **Cleanup** — pyproject, env docs, bootstrap reference.
6. **Architecture-decision items** — Tauri data_path and fs scope.

---

## L. HONESTY NOTES

- **Not reviewed deeply enough to claim safe**
  - Root C++ engine/ and KellyFFI implementation (realtime/audio-thread behavior not traced).
  - KmiDi_FINAL / KmiDi_PROJECT C++ (RTMemoryPool, PluginProcessor, etc.).
  - Full music_brain dependency tree (all modules that api.py imports).
  - Tauri plugin fs:default exact scope (allowlist paths).
  - Native plugin/DAW host-facing code (buffer size, sample rate, filesystem during render).

- **Native surfaces still needing direct inspection**
  - C++ code that implements kelly_brain_initialize and file I/O in KellyFFI.
  - Any code path that runs in audio callback or realtime thread (heap, locks, I/O).

- **Full-repo review**
  - Full *directory* listing and contract/entrypoint review was done for the primary stack (src/, engine/intent_ir/, music_brain/api.py, shared_schemas, scripts/sync_entities, CI, config). Not every file in music_brain, engine, or legacy trees was read line-by-line. Findings are evidence-based from inspected code; “no evidence found” is stated where checked; “skipped” and “not reviewed” are explicit above.

---

*End of report. Apply only high-confidence, low-risk fixes first; re-run tests after changes.*
