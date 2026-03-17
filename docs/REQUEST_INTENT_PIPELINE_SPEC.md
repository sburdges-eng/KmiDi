# Request → Intent Pipeline — Single Canonical Path

Enforcement of a single Request → Intent pipeline and elimination of logic bypass.  
File paths and code references as of the refactor.

---

## SECTION A — Entry Path

**POST /generate handler**

- **File:** `music_brain/api.py`
- **Line:** ~1328
- **Function:** `async def generate_music(request: GenerateRequest)`

**Exact call chain (strict path):**

1. `generate_music(request)` receives `GenerateRequest` (Pydantic; API boundary only).
2. `request.intent.technical` must be present; else HTTP 422.
3. `IntentPipeline()` is constructed (from `music_brain.pipeline`).
4. `normalized = pipeline.normalize(request)` — Stage 1: emotion_map, fallbacks, no override of explicit values.
5. `validated = pipeline.validate(normalized)` — Stage 2: `CompleteSongIntentRequest` (Pydantic); on `ValidationError` → HTTP 422.
6. `complete_intent = pipeline.expand(request, validated, normalized)` — Stage 3: `CompleteSongIntent`; on exception → HTTP 500 (fail-closed, no partial intent).
7. `result = api.process_song_intent(complete_intent, output_json=None)` — internal API accepts **only** `CompleteSongIntent`.
8. `process_song_intent` calls `process_intent(intent)` from `music_brain.session.intent_processor`.

**No other code path:** The legacy therapy_session fallback is only reachable when `use_full_pipeline` is False (currently hard-coded True), so the only active path is the pipeline above.

**Conversion entry point used by tests / other callers:**

- `DAiWAPI._convert_request_to_complete_intent(request)` in `music_brain/api.py` (lines ~848–854) delegates to `IntentPipeline().run(request)` (normalize → validate → expand). Single conversion source.

---

## SECTION B — Conversion Refactor

**Status:** Already implemented. No `_convert_to_intent` in codebase.

**Canonical conversion:**

- **Base:** `_convert_request_to_complete_intent` → `IntentPipeline().run(request)`.
- **Pipeline:** `music_brain/pipeline/intent_pipeline.py` — class `IntentPipeline`.

**3-stage pipeline:**

| Stage | Method | Input | Output |
|-------|--------|--------|--------|
| 1 | `normalize(request)` | Raw request (duck-typed `.intent`) | `Dict` (mood_primary via emotion_map, key_mode, tempo, structure, instruments, narrative_arc, groove_feel, etc.) |
| 2 | `validate(normalized)` | Normalized dict | `CompleteSongIntentRequest` (Pydantic; schema in `music_brain/engine_api/schema.py`) |
| 3 | `expand(request, validated, normalized)` | Request + validated + normalized | `CompleteSongIntent` (dataclass; `music_brain/session/intent_schema.py`) |

**Expansion guarantees (Stage 3):**

- **emotion_map:** Applied in Stage 1 (`_infer_mood_primary` + `EMOTION_MAP`); mood_primary flows into validated and then into `CompleteSongIntent`.
- **imagery_texture:** Set only in Stage 3 from `request.intent.imagery_texture`.
- **BPM/tempo:** Stage 1 clamps to 40–300 (`_infer_tempo`); Stage 3 builds `technical_tempo_range` from validated tempo.
- **core_event:** Stage 3: `core_wound` → `validated.core_desire` → `emotional_intent`.

**Fail-closed (Phase 6):**

- In `music_brain/api.py`, `pipeline.expand(...)` is wrapped in try/except: on any exception, log and raise `HTTPException(status_code=500, detail="intent expansion failed; no partial intent fallback")`. No fallback to partial intent.

**Code diff (fail-closed addition in api.py):**

```diff
                 try:
                     validated = pipeline.validate(normalized)
                 except ValidationError as validation_error:
                     ...
-                complete_intent = pipeline.expand(request, validated, normalized)
-
-                # Process full intent
+                try:
+                    complete_intent = pipeline.expand(request, validated, normalized)
+                except Exception as expand_exc:
+                    logging.exception("Intent expansion failed")
+                    raise HTTPException(
+                        status_code=500,
+                        detail="intent expansion failed; no partial intent fallback",
+                    ) from expand_exc
+
+                # Process full intent (CompleteSongIntent only; no Request past this point)
                 result = api.process_song_intent(complete_intent, output_json=None)
```

---

## SECTION C — process_intent Unification

**Canonical implementation:**

- **File:** `music_brain/session/intent_processor.py`
- **Function:** `process_intent(intent: CompleteSongIntent) -> Dict` (line ~719)
- **Behavior:** Builds `IntentProcessor(intent)` and returns `processor.generate_all()`.

**Other surfaces (all delegate or distinct):**

| Location | Role |
|----------|------|
| `music_brain/session/intent_bridge.py` | `process_intent(intent_json: str) -> str` — C++ bridge: parses JSON → `CompleteSongIntent.from_dict` → calls **canonical** `process_intent(intent)` (imported as `process_intent_canonical`), then converts result to C++ format. No longer uses a singleton `IntentProcessor()` with no args. |
| `music_brain/session/intent.py` | `process_intent_tool(intent_json: str)` — async tool; calls `api.process_song_intent(intent)` where intent is already `CompleteSongIntent`. |
| `music_brain/api.py` | Imports `process_intent` from `music_brain.session.intent_processor`; `process_song_intent` calls it with `CompleteSongIntent`. |
| `music_brain/orchestrator/bridge_api.py` | Uses `IntentProcessor` from `music_brain.orchestrator.processors` (different class: pipeline stage). Not the session intent processor. |

**Change made:** `intent_bridge.py` now imports and calls the canonical `process_intent(CompleteSongIntent)` from `music_brain.session.intent_processor` instead of maintaining a singleton `IntentProcessor()` and calling a non-existent instance method. No duplicate logic.

---

## SECTION D — Schema Enforcement

**Rule:** Request schema (`GenerateRequest` / Pydantic API models) is **only** allowed at the API boundary. All code past conversion must use `CompleteSongIntent` (or `CompleteSongIntentRequest` only inside the pipeline up to and including validate).

**Enforcement:**

1. **Conversion boundary:** The only place that turns a Request into an intent is `IntentPipeline().run(request)` (or the three steps normalize → validate → expand). Called from `generate_music` and from `_convert_request_to_complete_intent`.
2. **Downstream signatures:**  
   - `DAiWAPI.process_song_intent(self, intent: CompleteSongIntent, ...)`  
   - `process_intent(intent: CompleteSongIntent)` in `music_brain.session.intent_processor`  
   No function past conversion accepts `GenerateRequest`.
3. **Regression test:** `tests/unit/test_schema_boundary.py` asserts that `process_song_intent` and `process_intent` declare a parameter typed as `CompleteSongIntent` (no Request).

**Files:**

- Request models: `music_brain/api.py` (FastAPI block) — `GenerateRequest`, `EmotionalIntent`, `TechnicalIntent`.
- Validated request schema: `music_brain/engine_api/schema.py` — `CompleteSongIntentRequest`.
- Internal intent: `music_brain/session/intent_schema.py` — `CompleteSongIntent`.

---

## SECTION E — Tests Added

**Pipeline regression (emotion, imagery, tempo, core_event):**

- **File:** `tests/unit/test_intent_pipeline_regression.py`
- **Cases:**
  - **Emotion map:** joy → tenderness, grief preserved, anger → rage (via `mood_primary` through pipeline).
  - **Imagery texture:** non-empty and empty preserved in `CompleteSongIntent.song_intent.imagery_texture`.
  - **Tempo/BPM:** BPM 400 clamped to 300 in normalized/validated; valid BPM 100 preserved; `technical_tempo_range` present.
  - **Core event:** from `core_wound`; fallback to `core_desire` when `core_wound` is empty.

**Schema boundary:**

- **File:** `tests/unit/test_schema_boundary.py`
- **Cases:** `process_song_intent` and `process_intent` accept only `CompleteSongIntent` (signature assertion).

**Run:**

```bash
python3 -m pytest tests/unit/test_intent_pipeline_regression.py tests/unit/test_schema_boundary.py tests/unit/test_ui_mapping.py -v
```

---

## SECTION F — Risk Notes

1. **Legacy fallback:** The block “Legacy fallback retained for safety” in `generate_music` is currently unreachable (`use_full_pipeline = True`). If that flag is ever made conditional, a second path would reappear; consider removing or gating behind a strict feature flag.
2. **Tempo range for extreme BPM:** `expand()` builds `tempo_range = (max(60, tempo - 20), min(140, tempo + 20))`. For validated tempo 300 this yields (280, 140) (lo > hi). Downstream use may assume lo ≤ hi; consider clamping or normalizing the tuple in the pipeline.
3. **intent_bridge on error:** On any exception, `intent_bridge.process_intent(intent_json)` returns a default C++ result instead of failing. That is intentional for the C++ bridge but differs from the fail-closed rule for the HTTP API.
4. **Orchestrator IntentProcessor:** `music_brain/orchestrator/processors.py` (and bridge_api) use a different `IntentProcessor`; naming is overloaded but behavior is pipeline-stage specific, not the session intent processor.

---

**Summary:** Single entry path (POST /generate → IntentPipeline → process_song_intent → process_intent). Single conversion path (normalize → validate → expand). Canonical process_intent in `session.intent_processor`; intent_bridge and API call it. Schema boundary enforced and tested. Fail-closed on expansion failure. Regression tests cover emotion_map, imagery_texture, tempo, and core_event through the pipeline.
