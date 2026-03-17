# Cross-Cutting Audit Report — KmiDi / iDAW

Audit procedures: Dependency + supply-chain; Env + configuration; I/O boundaries; Authorization; Data integrity; Concurrency + idempotency; Error-handling + recovery; Observability; Test quality; Build/CI/Release; Performance structure; Security. Plus required trace methods, finding admission rule, mandatory negative checks, and output sections (Risk Matrix, Contract Drift, Config/ENV Matrix, Security Surfaces, Test Gaps).

**Scope:** React (Vite) frontend, Music Brain FastAPI API, Tauri/Rust shell, C++ (KellyCore/KellyFFI) — development and CI as represented in repo.

---

## A. DEPENDENCY + SUPPLY-CHAIN AUDIT

### Direct and critical transitive dependencies

- **Frontend (package.json):** React 19, react-dom, @tauri-apps/* (api, plugin-dialog, plugin-fs, plugin-opener), react-markdown, remark-gfm. Dev: Vite 7, TypeScript ~5.8.3, Tailwind/PostCSS, concurrently.
- **Python (pyproject.toml):** numpy, torch, librosa, pyyaml, scipy, pydantic, pybind11, fastapi, uvicorn. Optional: soundfile, pydub, lhotse (jepa), pytest, black, flake8, ruff, mypy.

### Findings (admission rule: code evidence, path, scenario, impact, minimal fix)

| # | Finding | Evidence | Path:Line | Scenario | Impact | Minimal fix |
|---|---------|----------|--------|----------|--------|-------------|
| D1 | **Duplicate `pydantic` in pyproject.toml** — listed twice (bare and `>=2.0`). | `pyproject.toml` dependencies list. | `pyproject.toml`:25–30 | Any install. | Possible version ambiguity; no functional bug observed. | Remove duplicate; keep single `pydantic>=2.0`. |
| D2 | **No postinstall/prebuild scripts** in package.json. | `package.json` has no `postinstall`, `preinstall`, `prepare`, `prebuild`. | `package.json` | N/A | None. | — |
| D3 | **Declared vs actual imports (frontend):** All declared deps are used (react, react-dom, Tauri plugins, react-markdown, remark-gfm). No unused prod deps; no missing prod deps masked by hoisting observed. | Grep of `src` imports vs package.json. | — | Build/runtime. | None. | — |
| D4 | **Lockfile and package-manager:** `package-lock.json` present, lockfileVersion 3; CI uses `npm ci`. Single package manager (npm). | `.github/workflows/ci.yml` (v1_shell_build), `package-lock.json`. | — | CI/local. | Reproducible installs when using `npm ci`. | — |

### Not found / no evidence

- Abandoned or obviously risky packages: not flagged (React 19, Vite 7, FastAPI, Tauri 2 are current).
- Multiple versions of core runtime (e.g. two Reacts) in lockfile: not observed in reviewed slice.
- Python: optional deps (jepa, audio) may pull heavy stacks (lhotse, etc.); not classified as “risky” without CVE check.

---

## B. ENV + CONFIGURATION AUDIT

### Config matrix

- **Represented:** local/dev (load-env.sh: .env → .env.development → feature envs → .env.local), production (.env.production). No separate test/staging env files in repo; CI does not load .env files for test job (uses defaults/implicit).

### Trace: definition → usage

| Variable | Defined where | Used where | Required? | Default / behavior | Risk |
|----------|----------------|------------|-----------|--------------------|------|
| `VITE_KMIDI_USE_API` | Build-time (Vite); not in .env.example | `src/hooks/useMusicBrain.ts` | No | Unset → external API disabled | Doc in ENVIRONMENT.md for frontend build; .env.example could add VITE_* for clarity. |
| `VITE_API_BASE` | Build-time | `src/hooks/useMusicBrain.ts` | No | `http://127.0.0.1:8000` | Safe. |
| `KMIDI_AUDIO_SERVE_ROOT` | Not in .env.example / ENVIRONMENT.md | `music_brain/api.py:1039` | No | `tempfile.gettempdir()` | Silent fallback; if set to wrong path, could expose dirs. Document and restrict to intended dir. |
| `FREESOUND_API_KEY` | .env.example, ENVIRONMENT.md (API Keys) | `music_brain/misc_code/freesound_downloader.py`, `music_brain/emotion_scale/freesound_downloader.py`, `music_brain/penta_core/ml/audio_downloader.py` | No | Empty string / None | Empty default; if code paths log or echo key, leakage. Verify no log of key. |
| `LLM_ONNX_*`, `OLLAMA_*` | Not in .env.example | `music_brain/intelligence/onnx_llm.py`, `ollama_bridge.py` | No | Various in-code defaults | Document in ENVIRONMENT.md. |
| `TAURI_DEV_HOST`, `TAURI_PLATFORM` | .env.example, vite.config.ts | `vite.config.ts:6` (host) | No | localhost / macos | Safe. |

### Findings

| # | Finding | Evidence | Path:Line | Scenario | Impact | Minimal fix |
|---|---------|----------|--------|----------|--------|-------------|
| E1 | **`KMIDI_AUDIO_SERVE_ROOT` used but not documented** in ENVIRONMENT.md / .env.example. | `os.environ.get("KMIDI_AUDIO_SERVE_ROOT", ...)` | `music_brain/api.py:1039` | Operator sets to broad path. | Larger file-serving surface; path traversal already mitigated by `relative_to`. | Add to ENVIRONMENT.md and .env.example; recommend narrow dir. |
| E2 | **Load-env.sh exports unvalidated:** `export "$line"` can fail or mis-parse for values with `=` in value. | `scripts/load-env.sh` | `scripts/load-env.sh:33` | Env value contains `=`. | Variable not set or wrong value. | Use `export` only for simple KEY=VAL or source with set -a. |
| E3 | **ENVIRONMENT.md documents `KMIDI_USE_API`** (backend/Streamlit); frontend uses **`VITE_KMIDI_USE_API`**. Different names. | ENVIRONMENT.md Feature Flags vs `useMusicBrain.ts` | `docs/ENVIRONMENT.md`, `src/hooks/useMusicBrain.ts:4` | Confusion when enabling API from frontend. | Developer sets wrong var; API stays disabled. | Document VITE_KMIDI_USE_API in ENVIRONMENT.md and .env.example. |

### Secrets

- No hardcoded API keys in committed code (only `os.getenv`/`os.environ.get` with empty or example defaults).
- .env.example contains placeholders; .env.local git-ignored. No evidence secrets are logged or exposed to client bundles (VITE_* are build-time and only control API base and feature flag).

---

## C. INPUT/OUTPUT BOUNDARY AUDIT

### Ingress

| Ingress | Location | Validation / normalization |
|---------|----------|----------------------------|
| HTTP POST /generate | `music_brain/api.py` | Pydantic `GenerateRequest` → `CompleteSongIntentRequest.model_validate(strict_payload)`; BPM/tempo clamped 40–300; key_mode regex; structure total bars ≤1000. |
| HTTP POST /interrogate | Same | `InterrogateRequest` (message, session_id, context). No length/sanitization limit on `message`. |
| HTTP POST /lyrics | Same | `LyricsRequest` (lyrics, source). No max length on `lyrics`. |
| HTTP GET /audio/{file_path:path} | Same | Path decoded, `Path.resolve()`; must be `relative_to(_audio_serve_root)`; 404 if outside. |
| HTTP PUT /config/humanizer | Same | JSON body normalized via `_normalize_humanizer_config`. Writes to `config/humanizer.json`. |
| HTTP POST /spectocloud/render | Same | `SpectocloudRenderRequest`; midi_file_path / midi_events / duration validated; audio_file_path currently rejected (not implemented). |
| HTTP POST /audio/classify, /voice/classify | Same | `audio_path` / path from body — used as file path; no path traversal check if path is user-controlled. |
| CLI / cron / queue / webhooks | — | None in scope. |

### Egress

| Egress | Location | Notes |
|--------|----------|--------|
| File system (write) | config/humanizer.json, temp MIDI/audio under tempdir | Humanizer config and generated artifacts. |
| File system (read) | MIDI/audio paths from request, config/humanizer.json, models dir | /audio/ sandboxed; spectocloud midi_file_path is server-local. |
| External API | Freesound, Ollama, ONNX (local) when used | Via env-configured keys/hosts. |
| Logs | logging.info/exception/warning | See Observability. |
| HTTP response | All handlers | JSON or FileResponse; 500 details sometimes `str(exc)`. |

### Findings

| # | Finding | Evidence | Path:Line | Scenario | Impact | Minimal fix |
|---|---------|----------|--------|----------|--------|-------------|
| I1 | **POST /audio/classify and /voice/classify accept `audio_path`** from body. If that path is passed through to filesystem without sandboxing, path traversal or read of arbitrary files. | `classify_audio(request: AudioClassifyRequest)` uses `request.audio_path`; `api.classify_voice_file(request.audio_path)`. | `music_brain/api.py` (e.g. 1196, 1243) | Attacker sends `audio_path: "/etc/passwd"` or path under KMIDI_AUDIO_SERVE_ROOT to probe. | Information disclosure or denial if path is used unsafely. | Restrict to a configured allowlist dir (e.g. same as KMIDI_AUDIO_SERVE_ROOT or dedicated upload dir); resolve and check `relative_to` that root before open. |
| I2 | **/interrogate and /lyrics** do not enforce max length on text. | `InterrogateRequest.message`, `LyricsRequest.lyrics` unbounded. | `music_brain/api.py` | Very large body. | DoS or memory pressure. | Add Pydantic `max_length` (e.g. 32k for lyrics, 4k for message). |
| I3 | **Internal type vs runtime validation:** UI → API uses `GenerateRequest` / `EmotionalIntent`; engine boundary uses `CompleteSongIntentRequest`. Conversion in handler with `model_validate(strict_payload)`. Contract is explicit; sync_entities.py keeps TS/Rust in sync. No finding that internal types are mistaken for validation. | api.py generate_music, engine_api/schema.py | — | — | — | — |

---

## D. AUTHORIZATION MATRIX AUDIT

### Route/action matrix

| Path / action | Authn | Authz | Ownership/tenant | Admin/internal restriction |
|---------------|-------|-------|------------------|----------------------------|
| GET / | No | — | — | No |
| GET /health | No | — | — | No |
| GET /docs | No | — | — | No (OpenAPI UI) |
| GET /audio/{path} | No | — | — | No (sandboxed to KMIDI_AUDIO_SERVE_ROOT) |
| GET /emotions | No | — | — | No |
| GET/PUT /config/humanizer | No | — | — | No (writes to config/) |
| POST /config/humanizer/reload | No | — | — | No |
| POST /spectocloud/render | No | — | — | No |
| POST /generate | No | — | — | No |
| POST /interrogate | No | — | — | No |
| POST /lyrics, GET /lyrics | No | — | — | No (in-memory per process) |
| POST /audio/classify, /audio/valence-arousal | No | — | — | No |
| GET /audio/models | No | — | — | No |
| POST /voice/classify | No | — | — | No |
| GET /ai/jepa/status | No | — | — | No |

**Conclusion:** No authentication or authorization on any endpoint. No admin-only or tenant checks. By design for local/dev “Brain” usage; if API is ever exposed to untrusted networks, authn/authz must be added. No confused-deputy or IDOR in current design (no multi-tenant resources).

---

## E. DATA INTEGRITY + MIGRATION AUDIT

- **Schema:** No SQL database in scope. “Schema” is Pydantic (`CompleteSongIntentRequest`, `GenerateRequest`, etc.) and JSON (shared_schemas/CompleteSongIntentRequest.json) with sync to TS and Rust.
- **Migrations:** No DB migrations. Scripts that touch data: sync_entities.py (generates code from JSON); prepare_datasets; acquire scripts. No destructive migrations or rollback logic to audit.
- **Lyrics/session state:** Stored in singleton `api` (DAiWAPI) in memory. No persistence; no migration or backup.
- **Config writes:** PUT /config/humanizer writes JSON to config/humanizer.json; no versioning or rollback.

**Findings:** None that meet the admission rule for “destructive migration” or “nullable/enum drift.” Application assumes no DB; schema evolution is handled by sync_entities and CI drift check (git diff on generated files).

---

## F. CONCURRENCY + IDEMPOTENCY AUDIT

- **Concurrent surfaces:** Multiple requests to same FastAPI process (async); single global `api` instance; lyrics state shared.
- **Background jobs / retries / webhooks:** None in scope.
- **Optimistic UI:** Frontend may send multiple generate requests; no idempotency keys.

**Findings:**

| # | Finding | Evidence | Path:Line | Scenario | Impact | Minimal fix |
|---|---------|----------|--------|----------|--------|-------------|
| C1 | **Lyrics and “last generated” state are process-global.** Concurrent POST /generate and POST /lyrics can interleave. | `api.user_lyrics`, `api.last_generated_lyrics` on single `DAiWAPI()` instance. | `music_brain/api.py` (e.g. 606, 267–268) | Two users or tabs: one sets lyrics, other generates; first gets wrong lyrics in result. | Cross-request state mix-up. | Document as single-user/local only; or add per-session/store keys (e.g. session_id) and scope state. |
| C2 | **POST /generate is not idempotent.** No idempotency key; duplicate submissions create duplicate work and files (temp MIDI/audio). | generate_music creates files under tempdir with timestamp. | `music_brain/api.py` (e.g. 958–968) | Client retries or double-click; multiple generations. | Wasted CPU, duplicate files. | Optional idempotency key (header/body) and skip or return cached result. |

---

## G. ERROR-HANDLING + RECOVERY AUDIT

- **Failure paths:** Handlers generally `try/except`; re-raise `HTTPException`; catch `Exception` and `logging.exception()` then `raise HTTPException(status_code=500, detail=str(exc))`.
- **Cleanup/rollback:** No DB; file writes (humanizer config, temp MIDI/audio) are not rolled back on failure. Partial write of config/humanizer.json possible if process dies mid-write (no atomic rename).
- **Exposure:** Several 500 responses use `detail=str(exc)` (e.g. list_emotions, reload_humanizer, spectocloud render, generate, set_lyrics, get_lyrics, classify_audio, etc.). So exception messages (sometimes with internal paths or stack context) can reach the client.

**Findings:**

| # | Finding | Evidence | Path:Line | Scenario | Impact | Minimal fix |
|---|---------|----------|--------|----------|--------|-------------|
| G1 | **500 responses expose exception message to client.** `str(exc)` may contain paths, import names, or traceback snippets. | Multiple `raise HTTPException(status_code=500, detail=str(exc))`. | e.g. `music_brain/api.py:1124,1249,1278,1374,1158,1736` | Any unhandled exception in those handlers. | Information disclosure; possible leakage of internal structure. | Log full exc; return generic message in production (e.g. "Internal server error") or sanitized message. |
| G2 | **Humanizer config write not atomic.** Direct write to config/humanizer.json. | `with open(cfg_path, "w", ...) json.dump(...)`. | `music_brain/api.py:1233–1235` | Process killed during write. | Corrupted or empty config file. | Write to temp file then os.replace to cfg_path. |

---

## H. OBSERVABILITY AUDIT

- **Logging:** Python `logging` used in api.py (info, warning, error, exception). No correlation IDs or request IDs. No structured (JSON) logs.
- **Metrics/tracing:** No Prometheus/OpenTelemetry or similar in repo.
- **Sensitive data:** Some `logging.info` include `decoded_path` (user-controlled path); could log path under allowed root. Not logging API keys or lyrics content in reviewed paths.
- **Risky operations:** Generate path has exception logging; no explicit “request started”/“request completed” log with status. Retries and external calls are not explicitly traced.

**Findings:**

| # | Finding | Evidence | Path:Line | Scenario | Impact | Minimal fix |
|---|---------|----------|--------|----------|--------|-------------|
| O1 | **User-controlled path in logs.** `logging.info(f"Serving audio file: {decoded_path}")` and similar. | api.py | `music_brain/api.py:1060–1061,1104` | Attacker sends path with sensitive filename. | Path may appear in log aggregation. | Log only basename or hash; or redact. |
| O2 | **No correlation/request ID.** Hard to trace a single request across logs. | No middleware or context. | — | Debugging production. | Hard to correlate logs with request. | Add middleware that sets/reads request_id and logs it. |

---

## I. TEST QUALITY AUDIT

- **Schema tests:** `tests/unit/test_api_schema.py` — valid payload, rejected BPM, key_mode, missing required, oversized structure, UI constraint fields. Good boundary tests.
- **CI:** Python tests (pytest unit + schema); C++ build and tests; Tauri build and `cargo test`; schema drift check (git diff on generated files). No dedicated frontend unit test job in main ci.yml (Vitest not observed in scripts).
- **Snapshot/mock:** No snapshot-heavy tests reviewed; test_api_schema uses real Pydantic validation.

**Gaps (semantic):**

| Subsystem | Missing scenario | Likely failure | Test to add |
|-----------|------------------|----------------|-------------|
| API /generate | Invalid structure (e.g. wrong section name) | 422 with clear message | Parametrized test for structure name regex. |
| API /audio/{path} | Path traversal (e.g. `..`) | 404, path not served | Integration test: request with path outside root. |
| API error handling | Handler raises before response | 500, no leak of stack | Test that 500 detail is sanitized when configured. |
| Frontend | TypeScript contract vs API | Runtime type error or wrong payload | Contract test or e2e that builds request from TS types and hits API. |

---

## J. BUILD / CI / RELEASE AUDIT

- **Local vs CI:** CI runs: Python install + pytest; sync_entities + drift check; C++ configure + build + test; Valgrind; Tauri build (npm ci, cmake KellyFFI, npm run tauri build); quality (black, flake8, mypy). Local dev: `npm run dev`, `npm run dev:python`, `npm run dev:tauri`, `./scripts/dev-setup.sh`.
- **Typecheck/lint:** Python: flake8, black, mypy in CI (quality job; mypy continue-on-error in ci-python.yml). Frontend: `npm run build` runs `tsc && vite build`; that runs only in v1_shell_build (Tauri), not in a standalone frontend job. So **frontend typecheck is only guaranteed when Tauri build runs.**
- **Build-time vs runtime env:** VITE_* are build-time for frontend; backend uses runtime env. No mismatch observed.
- **Artifact:** Tauri produces app artifact; C++ produces library/plugins. No artifact mismatch reported.

**Findings:**

| # | Finding | Evidence | Path:Line | Scenario | Impact | Minimal fix |
|---|---------|----------|--------|----------|--------|-------------|
| J1 | **No dedicated frontend typecheck/lint job.** TypeScript and Vite build run only as part of Tauri build. | .github/workflows/ci.yml has no job that runs `npx tsc --noEmit` or `npm run build` for frontend only. | `.github/workflows/ci.yml` | PR changes only src/; Python and C++ pass; frontend broken. | Broken main. | Add job: checkout, npm ci, npx tsc --noEmit (and optionally npm run build). |

---

## K. PERFORMANCE STRUCTURE AUDIT

- **N+1:** No DB in API; no N+1.
- **Rerenders:** Not audited at component level; React 19 and single app shell.
- **Unbounded loops/maps:** generate path builds structure/instruments from validated payload (bounded by schema). Spectocloud uses payload.n_particles and events; validated.
- **Sync on hot path:** process_song_intent and harmony/groove are CPU-bound; no async offload. Acceptable for “Brain” service.
- **Pagination/streaming:** /generate returns full result; no streaming. Large responses possible for big structure.

No finding meeting “code clearly indicates real cost risk” beyond the above (no pagination on generate response).

---

## L. SECURITY REVIEW PROCEDURE

- **Injection:** Request bodies parsed as JSON and Pydantic; no raw SQL. Path parameters (e.g. file_path) decoded and checked with `relative_to`. No command or template injection observed.
- **SSRF/open redirect:** No server-side fetch of user URL in reviewed code. N/A.
- **Path traversal:** /audio/{file_path:path} — `Path(decoded_path).resolve().relative_to(_audio_serve_root)`; rejected if outside. **Mitigated.**
- **Session/cookie:** No session or cookie auth; CORS allowlist (localhost, 127.0.0.1, tauri://localhost, 5173). Credentials allowed.
- **CSRF:** No session-based auth; no CSRF tokens. If API is used from browser on same origin, risk is low; document if API is ever opened to other origins.
- **Deserialization:** JSON + Pydantic; no pickle or unsafe deserialization.
- **File handling:** Humanizer config and temp files; audio paths restricted on /audio/. /audio/classify and /voice/classify take path from body — see I1.
- **CORS:** allow_origins fixed list; not overbroad.
- **Client-provided fields:** Structure, instruments, keys validated by schema. BPM/duration clamped.
- **Stack traces/internal identifiers:** 500 detail=str(exc) can expose internal info — see G1.

**Tied finding:** See I1 (audio_path on classify endpoints) and G1 (500 detail).

---

## 8) REQUIRED TRACE METHODS

### Subsystem: POST /generate (full intent pipeline)

- **Happy-path trace:** Client sends valid GenerateRequest with technical.structure and instruments → model_validate(strict_payload) → CompleteSongIntent → process_song_intent → result with harmony/groove/arrangement/production → optional MIDI/audio render → response with result, lyrics, structure, instruments, midi_path/audio_path.
- **Failure-path trace:** Invalid structure (e.g. bars > 128) → ValidationError → 422 with safe_errors. Exception in process_song_intent → logging.exception → 500 with str(exc).
- **State-transition trace:** Lyrics state: _select_lyric_payload uses api.user_lyrics or api.generate_structured_lyrics; last_generated_lyrics set by generate_structured_lyrics. No DB; state is in-memory api singleton.
- **Auth/permission trace:** No auth; any client can call. N/A.
- **Data-contract trace:** UI (Intent.ts / useMusicBrain) → buildGeneratePayload → HTTP body (GenerateRequest) → API (EmotionalIntent, TechnicalIntent) → strict_payload → CompleteSongIntentRequest (engine_api/schema) → CompleteSongIntent (session) → process_intent. Contract enforced at CompleteSongIntentRequest.model_validate; sync_entities keeps TS/Rust aligned with JSON.

### Subsystem: GET /audio/{file_path}

- **Happy-path trace:** file_path decoded → Path.resolve() → relative_to(_audio_serve_root) → exists(), is_file() → FileResponse.
- **Failure-path trace:** Path outside root → ValueError → 404. File not found → 404. Not a file → 400.
- **Auth:** None. **Data-contract:** Path is string; no schema beyond path.

---

## 9) FINDING ADMISSION RULE

All findings in this report include: code reference (file/path), line or area, realistic scenario, impact, and minimal fix. No finding is included without these.

---

## 10) MANDATORY NEGATIVE CHECKS

| Check | Result |
|-------|--------|
| Auth bypass | **Reviewed.** No auth implemented; no bypass. If API is exposed, auth must be added. |
| Data loss migration risk | **Reviewed.** No DB migrations. Config write (humanizer) not atomic — see G2. |
| Retry/idempotency bugs | **Reviewed.** No idempotency on /generate; duplicate requests duplicate work — see C2. Lyrics state shared — see C1. |
| Secret leakage | **Reviewed.** No hardcoded secrets. 500 detail could leak internals — see G1. Env vars not logged in reviewed code; FREESOUND empty default — verify no log. |
| Client/server contract drift | **Reviewed.** CI runs sync_entities and git diff on generated TS/Rust; schema tests validate CompleteSongIntentRequest. Risk: frontend builds payload that omits new required fields — covered by schema tests and sync. |
| Timezone/date bugs | **Not deeply checked.** Created timestamps use `time.strftime`; no user timezone in scope. |
| CI/release drift | **Reviewed.** Frontend typecheck only in Tauri job — see J1. |

---

## F. RISK MATRIX

| Subsystem | Failure mode | Likelihood | Impact | Why |
|-----------|--------------|------------|--------|-----|
| API | No auth on any route | High (if exposed) | High | Any client can generate, read lyrics, write config. |
| API | 500 exposes str(exc) | Medium | Medium | Internal paths/messages in client response. |
| API | Classify endpoints accept arbitrary audio_path | Medium | Medium | Path traversal or read outside intended dir if not sandboxed. |
| API | Lyrics/state shared across requests | Medium | Low | Wrong lyrics in response under concurrency. |
| Config | Humanizer config write not atomic | Low | Low | Corrupted config if kill during write. |
| CI | Frontend typecheck only in Tauri job | Medium | Medium | TS errors can land on main if only src/ changed. |
| Env | KMIDI_AUDIO_SERVE_ROOT misconfigured | Low | Medium | Larger file serve surface. |

---

## G. CONTRACT DRIFT TABLE

| Producer | Consumer | Contract | Drift found | Fix |
|----------|-----------|----------|-------------|-----|
| shared_schemas/CompleteSongIntentRequest.json | src/types/Intent.ts, src-tauri generated intent.rs | JSON schema | CI enforces git diff; none. | Keep sync_entities + drift check. |
| FastAPI GenerateRequest/EmotionalIntent | React useMusicBrain, buildGeneratePayload | Request body shape | API expects technical.structure (list of {name, bars, repetitions}); UI sends structure. Optional groove_feel, narrative_arc. | Document; add test that TS types match API. |
| CompleteSongIntentRequest (engine) | Generate handler | strict_payload dict | Handler builds strict_payload from tech + intent; model_validate. | None. |

---

## H. CONFIG/ENV MATRIX

| variable/config | defined where | used where | required? | default behavior | risk |
|-----------------|----------------|------------|-----------|------------------|------|
| VITE_KMIDI_USE_API | (build env) | src/hooks/useMusicBrain.ts | No | false (API disabled) | Doc gap — see E3. |
| VITE_API_BASE | (build env) | src/hooks/useMusicBrain.ts | No | http://127.0.0.1:8000 | Low. |
| KMIDI_AUDIO_SERVE_ROOT | (runtime env) | music_brain/api.py | No | tempfile.gettempdir() | Doc gap; restrict to intended dir — see E1. |
| FREESOUND_API_KEY | .env.example | freesound_downloader, audio_downloader | No | "" | Verify not logged. |
| LLM_ONNX_* / OLLAMA_* | (runtime env) | onnx_llm.py, ollama_bridge.py | No | In-code defaults | Document. |
| KELLY_MODELS_PATH | .env.example | (C++/config) | Yes (for C++) | ./models | Documented. |
| TAURI_DEV_HOST | .env.example, vite | vite.config.ts | No | localhost | Safe. |

---

## I. SECURITY SURFACES

| Entrypoint | Trust boundary | Protection present | Gap | Evidence |
|------------|----------------|--------------------|-----|----------|
| POST /generate | Network → API | Pydantic validation; BPM/structure clamped | None | api.py generate_music, engine_api/schema. |
| GET /audio/{file_path} | Network → API → FS | Path resolve + relative_to root | None | api.py serve_audio. |
| POST /lyrics, GET /lyrics | Network → API | None | No auth; no max length (DoS) | api.py set_lyrics, get_lyrics; I2. |
| PUT /config/humanizer | Network → API → FS | Normalization | No auth; write not atomic | api.py update_humanizer_config; G2. |
| POST /audio/classify, /voice/classify | Network → API → FS | None | audio_path could be arbitrary path | api.py classify_audio, classify_voice; I1. |
| All endpoints | — | CORS allowlist | No auth | CORS middleware; no auth middleware. |

---

## J. TEST GAPS

| Subsystem | Missing scenario | Likely failure | Test to add |
|-----------|------------------|----------------|-------------|
| API | Path traversal on /audio/ | Serve file outside root | Integration: GET /audio/../../../etc/passwd → 404. |
| API | Max length on /lyrics, /interrogate | DoS or error | Unit: Pydantic max_length or body size. |
| API | 500 sanitization | Leak in production | Config-driven test: 500 detail does not contain stack/path. |
| API | Classify with path outside allowed dir | Read arbitrary file | Integration: sandbox check for classify. |
| Frontend | TS vs API contract | Wrong payload or runtime error | Contract or e2e: build request from TS, POST, assert 2xx or expected 4xx. |
| CI | Frontend-only change | TS error not caught | Job: npx tsc --noEmit (and optionally npm run build). |

---

**End of report.** All findings satisfy the admission rule (evidence, path, scenario, impact, minimal fix). Mandatory negative checks are stated; config/env matrix, security surfaces, and test gaps are appended as required.
