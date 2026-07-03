<!-- markdownlint-disable MD011 MD056 -->  <!-- frozen point-in-time report; tables intentionally unrestructured -->
# Repository Archaeology & Static Analysis Audit

**Scope:** Full historical and structural analysis of the KmiDi repository to identify lost logic, duplicate implementations, and unification opportunities.  
**Rule:** Zero hallucination. Every finding cites file path and commit reference.  
**HEAD:** `6781720b` (docs: remove outdated documentation files).  
**Total commits (all refs):** ~897.

---

## SECTION A — Function Lineage Map

**Method:** Current HEAD file enumeration; deleted source files from `git log --diff-filter=D`; renames from `git log --name-status -M`; function/symbol extraction via grep for `def ` / `class ` (Python) and key API surface. A full per-commit AST index across 897 commits was not run; lineage below is for audited clusters and key deleted paths.

### A.1 Current HEAD source-file counts (representative)

| Extension | Count (approx.) | Notes |
|-----------|-----------------|--------|
| Tracked files (all) | 7999 | `git ls-files` |
| `.py` (excl. .tools, KmiDi/external) | ~250+ | music_brain, scripts, tests |
| `.cpp` / `.h` (excl. JUCE, .tools) | 400+ | engine, KmiDi_FINAL, src, include |
| `.ts` / `.tsx` | — | src/ (React) |
| `.rs` | — | engine/intent_ir |

### A.2 Deleted source files (from history; project code only)

Files below were deleted in repo history. “Deleted in” = one representative commit where the file no longer appears (or path removed). Excludes JUCE examples and KmiDi_BACKUP/external bulk deletes.

| File path (deleted) | Deleted in / Notes |
|--------------------|--------------------|
| `KmiDi_BACKUP/project/Logger.py` | D (backup tree) |
| `KmiDi_BACKUP/project/api/main.py` | D (backup tree) |
| `KmiDi_BACKUP/project/data/harmony_generator.py` | D (backup tree) |
| `KmiDi_BACKUP/project/data/groove_applicator.py` | D (backup tree) |
| `KmiDi_BACKUP/project/data/groove_extractor.py` | D (backup tree) |
| `KmiDi_BACKUP/project/data/emotional_mapping.py` | D (backup tree) |
| `KmiDi_BACKUP/project/data/emotion_thesaurus/emotion_thesaurus.py` | D (backup tree); current equivalent: `music_brain/data/emotion_thesaurus/emotion_thesaurus.py` |
| `KmiDi_BACKUP/project/data/scales/scale_generator.py` | D (backup tree) |
| `scripts/emotion_scale_sampler.py` | Renamed to `music_brain/samples/emotion_scale_sampler.py` (R092 in history) |

**Commit reference for recent doc/file cleanup:** `6781720b` (docs: remove outdated documentation files).

### A.3 Renamed / moved source files (representative)

| Old path | New path | Similarity |
|----------|----------|------------|
| `BridgeClient.cpp` | `src/BridgeClient.cpp` | R100 |
| `BridgeClient.h` | `include/BridgeClient.h` | R100 |
| `scripts/emotion_scale_sampler.py` | `music_brain/samples/emotion_scale_sampler.py` | R092 |
| `DEVELOPMENT_ROADMAP_music-brain.md` | `docs/DEVELOPMENT_ROADMAP_music-brain.md` | R077 |

### A.4 Function lineage table (audited clusters)

| function_name | file_path | first_seen_commit | last_seen_commit | status |
|---------------|-----------|-------------------|------------------|--------|
| `_convert_request_to_complete_intent` | music_brain/api.py | (in history) | 6781720b | active; not on POST /generate path |
| `_convert_to_intent` (inner) | music_brain/api.py ~L1482 | (in history) | 6781720b | active; used by generate_music |
| `process_intent` | music_brain/session/intent_processor.py | 15d4ed68 / 5e34200e | 6781720b | active |
| `process_intent` | music_brain/session/intent_bridge.py | — | 6781720b | active (wrapper) |
| `process_intent` | music_brain/kelly_companion/session/intent_processor.py | — | 6781720b | active (duplicate surface) |
| `process_intent` | music_brain/kelly/core/intent_processor.py | — | 6781720b | active (different type: Wound) |
| `floatToUmp32Value` / `buildUmpAffectCC` | src/midi/AffectUMP.cpp | — | 6781720b | active |
| `float_to_ump32` / `build_ump_cc_v2` | scripts/ump_affect_utils.py | — | 6781720b | active |
| `IntentPipeline` | music_brain/pipeline/intent_pipeline.py | — | 6781720b | active |
| `DAiWAPI` | music_brain/api.py | — | 6781720b | active |

**Reasoning:** first_seen/last_seen for specific functions not computed over full history; “active” = present at HEAD; “not on POST /generate path” = only used by tests (see Section D).

---

## SECTION B — Function Clusters

Clusters of “same intent” functions (by name, signature, or behavior). From `docs/CODEBASE_AUDIT_FUNCTION_CLUSTERS.md` plus verification.

### Cluster 1: Request → CompleteSongIntent (Generate path)

| Variant | File:line | Signature / role |
|---------|-----------|-------------------|
| A | music_brain/api.py:847 | `_convert_request_to_complete_intent(self, request) -> CompleteSongIntent` |
| B | music_brain/api.py:1482 | `_convert_to_intent(request, strict_intent)` (inner in generate_music) |

**Runtime:** NON_REALTIME (Python, HTTP).

### Cluster 2: Float → UMP32 (Affect / MIDI 2.0 CC)

| Variant | File | Role |
|---------|------|------|
| A | src/midi/AffectUMP.cpp | `floatToUmp32Value`, `buildUmpAffectCC` (REALTIME_SAFE) |
| B | scripts/ump_affect_utils.py | `float_to_ump32`, `build_ump_cc_v2` (NON_REALTIME) |

### Cluster 3: process_intent (Intent execution)

| Variant | File | Role |
|---------|------|------|
| A | music_brain/session/intent_processor.py:719 | `process_intent(intent: CompleteSongIntent) -> Dict` |
| B | music_brain/session/intent_bridge.py:28 | `process_intent(intent_json: str) -> str` (JSON wrapper) |
| C | music_brain/kelly_companion/session/intent_processor.py | Same name/semantics as A (duplicate module) |
| D | music_brain/kelly/core/intent_processor.py | `process_intent(self, wound: Wound)` — different domain |

### Cluster 4: Intent schema (CompleteSongIntent vs CompleteSongIntentRequest)

| Shape | File | Role |
|-------|------|------|
| CompleteSongIntentRequest | music_brain/engine_api/schema.py, src/types/Intent.ts, engine/intent_ir/.../intent.rs | Boundary contract |
| CompleteSongIntent | music_brain/session/intent_schema.py | Internal processing shape |

---

## SECTION C — Lost / Unmerged Logic

Per-cluster differential; code snippets are exact from current tree unless stated.

### C.1 Request → CompleteSongIntent (Cluster 1)

**Logic present in `_convert_request_to_complete_intent` (music_brain/api.py:847–937) but NOT in `_convert_to_intent` (L1482–1512):**

1. **Emotion mapping**  
   - **File:** music_brain/api.py, lines 538–553.  
   - **Snippet:**
   ```python
   emotion_map = {
       "grief": "grief", "sadness": "grief", "joy": "tenderness",
       "happiness": "tenderness", "anger": "rage", "rage": "rage",
       "fear": "fear", "love": "tenderness", "nostalgia": "nostalgia", "awe": "awe",
   }
   for key, value in emotion_map.items():
       if key.lower() in emotional.lower():
           mood_primary = value
           break
   ```
   - **Impact:** Strict path does not normalize free-text `emotional_intent` to canonical mood.

2. **imagery_texture**  
   - **File:** music_brain/api.py, lines 569–571 (legacy path).  
   - **Snippet:** `imagery_texture=getattr(request.intent, "imagery_texture", "") or ""`  
   - **Impact:** Production generate path never sets `complete_intent.song_intent.imagery_texture`; IntentProcessor.generate_production() uses `self.imagery`, so this input is dropped on strict path.

3. **core_event from core_wound or emotional; mood from parentheses**  
   - **File:** music_brain/api.py, 534–536 (legacy): `core_event=request.intent.core_wound or emotional`; 863–865: mood from `emotional.split("(")[0].strip()`.  
   - **Impact:** Strict path uses `core_event=req.intent.core_wound or validated.core_desire` (L919) and does not apply parenthesis stripping for mood.

4. **BPM clamping and tempo_range**  
   - Legacy: BPM clamped 40–300; tempo_range (bpm±20, clamped 60–140). Strict path derives from validated tempo ± 20; ensure same bounds are documented and applied.

**Strict-path snippet (for comparison)** — music_brain/api.py L1482–1510:

```python
def _convert_to_intent(req, validated):
    key_parts = validated.key_mode.split()
    technical_key = key_parts[0] if key_parts else "C"
    technical_mode = key_parts[1].lower() if len(key_parts) > 1 else "major"
    tempo_range = (max(60, validated.tempo - 20), min(140, validated.tempo + 20))
    return CompleteSongIntent(
        core_event=req.intent.core_wound or validated.core_desire,
        core_longing=validated.core_desire, mood_primary=validated.mood_primary,
        narrative_arc=validated.narrative_arc, ...
        # no imagery_texture, no emotion_map, no mood-from-parentheses
    )
```

**Commit:** Logic exists at HEAD 6781720b; “lost” = not used on the code path that serves POST /generate.

### C.2 Float → UMP32 (Cluster 2)

**Lost features:** None identified. Python comment in scripts/ump_affect_utils.py states alignment with AffectUMP.h/cpp.

### C.3 process_intent (Cluster 3)

**Lost features:** No confirmed logic unique to one implementation. **Risk:** kelly_companion and session intent_processor copies may have diverged; not diffed line-by-line here.

### C.4 Intent schema (Cluster 4)

**Lost on strict path:** Same as C.1 — imagery_texture, emotion_map, and richer Phase 0/1 fields (CompleteSongIntent) not populated when using CompleteSongIntentRequest-only path.

---

## SECTION D — Orphaned Features

Functions or modules that are not used on the main production path or are only referenced in tests/docs.

| Item | File path | Callers / references | Reasoning |
|------|-----------|----------------------|-----------|
| `DAiWAPI._convert_request_to_complete_intent` | music_brain/api.py:847 | tests/unit/test_ui_mapping.py:35 | Only used by test; never called from generate_music or POST /generate. |
| `music_brain/kelly_companion/*` (IntentProcessor, emotion_thesaurus, engines, etc.) | music_brain/kelly_companion/ | QUICK_START.md (docs only) | No import from music_brain.api or main API; QUICK_START references kelly_companion.session.IntentProcessor and related. |
| `music_brain/misc_code/api.py` `process_intent` | music_brain/misc_code/api.py:48 | Import from music_brain.session.intent_processor | Call site exists; not orphaned. Verify whether misc_code API is mounted or used. |
| `process_intent_tool` | music_brain/session/intent.py:62 | Registered as tool | Used by session tooling; not orphaned. |

**Orphaned capability summary:**  
- **Salvageable:** `_convert_request_to_complete_intent` contains emotion_map, imagery_texture, key/mode/tempo logic; either use on a legacy/debug endpoint or merge its behavior into the strict-path conversion.  
- **kelly_companion:** Orphaned from main API; only QUICK_START.md references it. Either re-export from music_brain.session.intent_processor or document as alternate/legacy surface.

---

## SECTION E — Unified Refactor Proposals

### E.1 Request → CompleteSongIntent (Cluster 1)

**Proposal (from CODEBASE_AUDIT_FUNCTION_CLUSTERS.md):**

- Do not merge the two implementations into one; they serve unvalidated vs validated entry points.
- **Canonical spec:** Document in one place: (1) allowed modes (VALID_MUSICAL_MODES), (2) BPM and tempo_range rules, (3) when emotion_map applies, (4) mapping CompleteSongIntentRequest → CompleteSongIntent field list.
- **Restore on strict path:**  
  - After building `complete_intent` in `_convert_to_intent`, set `complete_intent.song_intent.imagery_texture` from `request.intent.imagery_texture` when present.  
  - Optionally apply the same `emotion_map` to `request.intent.emotional_intent` and use it to default/override `mood_primary` when the UI sends free text.
- **Single conversion helper:** Refactor so both paths call one shared helper with signature `(request.intent, validated: Optional[CompleteSongIntentRequest]) -> CompleteSongIntent`, with branches for “from validated” vs “from raw request” (emotion_map, imagery_texture, key parsing). Preserves behavior and recovers lost fields.

**Risk:** Low if helper is additive and strict validation remains the gate.

### E.2 Float → UMP32 (Cluster 2)

- Do not merge across REALTIME vs non-REALTIME.
- **Canonical spec:** One short doc (e.g. src/midi/README.md or docs/) defining: (1) formula float → UMP32, (2) CC indices and ranges, (3) that Python and C++ must stay in sync for harness/bridge scripts.
- **Tests:** Add Python test for `float_to_ump32` with fixed (value, min, max) triples and expected 32-bit outputs to catch drift.

### E.3 process_intent (Cluster 3)

- **Unify call sites:** All API entry points that need “full song intent → harmony/groove/arrangement/production” should call `music_brain.session.intent_processor.process_intent` (or the same module). Confirm music_brain/misc_code/api.py and any api_misc import from session.
- **Duplicate modules:** Treat music_brain/kelly_companion/session/intent_processor.py (and kellymidicompanion_session if present) as salvageable only if another product depends on them; otherwise remove or re-export from music_brain.session.intent_processor.
- **intent_bridge:** Keep as thin wrapper: JSON → CompleteSongIntent → session process_intent → C++-format JSON.

### E.4 Intent schema (Cluster 4)

- Keep two shapes: CompleteSongIntentRequest = boundary; CompleteSongIntent = internal. Document the single intended conversion path and which fields are preserved (and which are legacy-only).

---

## SECTION F — Risk Report

| Priority | Issue | Location | Reasoning |
|----------|--------|----------|-----------|
| **HIGH** | Emotion mapping and imagery_texture never applied on POST /generate | music_brain/api.py: _convert_to_intent vs _convert_request_to_complete_intent | Production path ignores free-text emotion normalization and imagery_texture; user intent may be under-expressed. |
| **HIGH** | Duplicate intent_processor implementations may diverge | music_brain/session/intent_processor.py vs music_brain/kelly_companion/session/intent_processor.py | Same line numbers (e.g. 719) suggest copy; fixes in one may not be applied to the other. |
| **MEDIUM** | _convert_request_to_complete_intent only used by tests | music_brain/api.py, tests/unit/test_ui_mapping.py | Valuable logic (emotion_map, imagery_texture, key/mode/tempo) not on production path; tests may pass while production behavior differs. |
| **MEDIUM** | kelly_companion only referenced from QUICK_START.md | music_brain/kelly_companion/, QUICK_START.md | If QUICK_START is outdated or unused, kelly_companion is dead weight; if it is supported, it should be wired or re-exported clearly. |
| **LOW** | Float → UMP32 Python/C++ parity | scripts/ump_affect_utils.py, src/midi/AffectUMP.cpp | No automated parity test; comment says “must match”; drift could cause harness/bridge bugs. |
| **LOW** | Deleted backup data modules (harmony_generator, groove_*, scale_generator) | KmiDi_BACKUP/project/data/*.py | Logic may have been reimplemented elsewhere (e.g. music_brain, penta-core); confirm no unique behavior lost in backup tree. |

---

## SECTION G — PHASE 8: TEST COVERAGE GAP

Map audited logic to existing tests; identify logic that existed historically but is not tested; suggest tests for recovered logic. All file paths and test names are from the current tree.

### G.1 Logic vs existing tests

| Logic (cluster / function) | Source file | Existing test(s) | What is tested | What is not tested |
|----------------------------|-------------|------------------|----------------|--------------------|
| **Request → CompleteSongIntent (legacy)** | music_brain/api.py:847 `_convert_request_to_complete_intent` | tests/unit/test_ui_mapping.py | Single test: Build GenerateRequest with imagery_texture, core_wound, key, bpm, narrative_arc, etc.; call `api._convert_request_to_complete_intent(req)`; assert CompleteSongIntent shape, imagery_texture, key/mode, vulnerability_scale mapping, groove, rule_to_break. | **emotion_map:** No test that "sadness" → mood_primary "grief", "happiness" → "tenderness", etc. test_ui_mapping uses "melancholy" (not in map) so mood stays "melancholy". **BPM clamp** (40–300) and **tempo_range** (60–140) not asserted. **core_event** from core_wound vs emotional; **mood from parentheses** (e.g. "joy (bright)" → tenderness) not tested. |
| **Request → CompleteSongIntent (strict path)** | music_brain/api.py:1482 `_convert_to_intent` (inner in generate_music) | tests/test_input_validation.py:381 `test_full_convert_roundtrip` | Builds CompleteSongIntent **manually** with same shape as _convert_to_intent (key_mode split, tempo_range, validated fields). Does **not** call the real _convert_to_intent or the API. Asserts narrative_arc, vulnerability_scale, groove, rule_to_break, genre, key, mode. | **Strict-path conversion itself** is untested: no test that calls generate_music path or POST /generate and asserts on the resulting CompleteSongIntent. **Regression for recovered logic:** No test that imagery_texture is present when restored on strict path; no test that emotion_map is applied when added. |
| **POST /generate endpoint** | music_brain/api.py (FastAPI) | — | test_api_audit_fixes.py uses TestClient for /audio/classify, /interrogate, /lyrics, etc.; **no test for POST /generate**. e2e: tests/e2e/frontend-backend.test.ts invokes `generate_music` (Tauri). | **No Python unit or integration test** that POSTs to /generate with a valid body and asserts response shape or that intent conversion (strict path) ran. |
| **process_intent** | music_brain/session/intent_processor.py:719 | tests/unit/test_intent_processor.py | test_intent_processor.py: TestIntentProcessor.test_processor_creation — only creates IntentProcessor(sample_intent) and asserts intent, key, mode. **No call to process_intent(intent)**. | **process_intent()** return value (harmony, groove, arrangement, production, intent_summary) is **not tested**. No test for empty or edge intents. |
| **intent_bridge process_intent** | music_brain/session/intent_bridge.py:28 `process_intent(intent_json: str)` | — | None. | JSON in → CompleteSongIntent.from_dict → session process_intent → C++-format JSON out is **untested**. |
| **Float → UMP32** | scripts/ump_affect_utils.py: float_to_ump32, build_ump_cc_v2 | — | None. Scripts (ump_affect_harness, morph_affect_bridge) use these at runtime. | **No unit test** for float_to_ump32(value, min, max) → expected 32-bit value, or for build_ump_cc_v2 byte layout. |
| **Intent schema (CompleteSongIntent)** | music_brain/session/intent_schema.py | tests/test_input_validation.py (CompleteSongIntent.from_dict, mood_secondary_tension); conftest sample_intent | from_dict, mood_secondary_tension coercion. | Full field list and defaulting not exhaustively tested. |
| **kelly_companion intent_processor** | music_brain/kelly_companion/session/intent_processor.py | conftest.py: sample_emotion_thesaurus uses kelly_companion.core.emotion_thesaurus | No test calls kelly_companion session process_intent. | Parity or divergence vs music_brain.session.intent_processor **untested**. |

### G.2 Logic that existed historically but is not tested

- **emotion_map** (music_brain/api.py L538–553): Exists in legacy conversion; never exercised by tests with inputs that trigger the map (e.g. emotional_intent "sadness" or "happiness"). test_ui_mapping uses "melancholy", which does not match any key, so the loop never overwrites mood_primary.
- **BPM clamping and tempo_range formula** (L898–904): bpm clamped to 40–300; tempo_range (max(60, bpm-20), min(140, bpm+20)). No test asserts these bounds or that invalid BPM is clamped.
- **Mode validation** (L891–895): technical_mode must be in VALID_MUSICAL_MODES else "major". Not tested.
- **Mood from parentheses** (L863–865): mood_primary = emotional.split("(")[0].strip(). Not tested.
- **core_event = core_wound or emotional** (L918): Legacy path; strict path uses core_wound or validated.core_desire. No test that explicitly checks core_wound taking precedence when present.
- **Float → UMP32 formula**: Historically present in both C++ and Python; no automated test that Python output matches a fixed expected value (or C++ if harness available) to prevent drift.

### G.3 Suggested tests for recovered logic

After implementing the unified conversion helper or restoring emotion_map/imagery_texture on the strict path (see Section E), add the following. File paths are suggested locations; reasoning ties to Section C lost logic.

| Suggested test | File (suggested) | Logic covered | Rationale |
|----------------|------------------|---------------|-----------|
| **test_emotion_map_maps_sadness_to_grief** | tests/unit/test_ui_mapping.py or tests/unit/test_api_convert.py | emotion_map in _convert_request_to_complete_intent | Build GenerateRequest with emotional_intent="sadness"; call _convert_request_to_complete_intent; assert intent.song_intent.mood_primary == "grief". Similarly for "happiness" → "tenderness", "anger" → "rage". | Ensures recovered emotion_map (if applied on strict path) or legacy path is correct. |
| **test_emotion_map_parentheses_stripped** | Same | mood from parentheses | emotional_intent="joy (bright)"; assert mood_primary becomes "tenderness" (from map) or at least "joy" (strip). | Covers legacy mood extraction. |
| **test_bpm_clamped_and_tempo_range** | Same | BPM 40–300, tempo_range 60–140 | tech with bpm=20 → assert clamped to 40 and tempo_range; bpm=400 → 300; bpm=100 → tempo_range (80, 120). | Prevents regressions when unifying conversion. |
| **test_strict_path_imagery_texture_when_restored** | tests/unit/test_api_convert.py or test_input_validation.py | _convert_to_intent (strict) | When unified helper or strict-path fix is in place: build valid GenerateRequest + CompleteSongIntentRequest with request.intent.imagery_texture="foggy"; call the conversion used by generate_music; assert complete_intent.song_intent.imagery_texture == "foggy". | Regression test for recovered imagery_texture on production path. |
| **test_post_generate_returns_200_and_shape** | tests/unit/test_api_generate.py (new) or test_api_audit_fixes.py | POST /generate, strict path | TestClient.post("/generate", json=minimal_valid_body) with required intent fields; assert status 200 and response has expected keys (e.g. harmony, groove or error handled). Optional: assert that when request.intent.imagery_texture is sent, downstream intent carries it (after restore). | Ensures the live generate path is exercised and conversion is integrated. |
| **test_process_intent_returns_structure** | tests/unit/test_intent_processor.py | process_intent(intent) | Call process_intent(sample_intent); assert result is dict with keys e.g. harmony, groove, arrangement, production, intent_summary (match actual return shape from intent_processor.py). | Covers the main session entry point used by API. |
| **test_intent_bridge_json_roundtrip** | tests/unit/test_intent_bridge.py (new) | intent_bridge.process_intent(str) | Build minimal CompleteSongIntent, serialize to JSON (e.g. via from_dict/to_dict or equivalent), call process_intent(intent_json); assert result is str and parseable; optional assert on structure. | Covers C++-oriented JSON bridge. |
| **test_float_to_ump32_known_values** | tests/unit/test_ump_affect_utils.py (new) | scripts/ump_affect_utils.py | Call float_to_ump32(0.0, -1.0, 1.0) → expect 0x7FFFFFFF or equivalent; float_to_ump32(1.0, 0.0, 1.0) → 0xFFFFFFFF; float_to_ump32(-1.0, -1.0, 1.0) → 0. Document expected values so C++ can be compared later. | Catches Python/C++ drift (Section E.2). |
| **test_build_ump_cc_v2_layout** | Same | build_ump_cc_v2 | For group=0, channel=0, controller=0x28, data32=0xFFFFFFFF; assert returned tuple has 8 bytes and first nibbles match MIDI 2.0 CC (e.g. 0x40, 0x0B, ...). | Documents and locks byte layout. |

**Note:** For tests that require the unified conversion helper or strict-path imagery_texture/emotion_map restore, implement the code change first (Section E.1), then add the corresponding test so the recovered logic is covered.

---

## Tools / methods used

- `git rev-parse HEAD`, `git log --oneline`, `git ls-files`
- `git log --diff-filter=D --name-only`, `git log --name-status -M` (renames)
- `grep` for `def `, `class `, `process_intent`, `_convert_request_to_complete_intent`, `_convert_to_intent`, and imports
- Existing audit: `docs/CODEBASE_AUDIT_FUNCTION_CLUSTERS.md`
- Merge/restoration context: `docs/KMIDI_FINAL_MERGE_PLAN.md`

---

## Summary

- **Function lineage:** Mapped for audited clusters and key renames/deletes; full per-commit index not produced.
- **Clusters:** Request→Intent, Float→UMP32, process_intent, Intent schema (four clusters).
- **Lost logic:** Emotion mapping, imagery_texture, core_event/mood parsing from legacy conversion not applied on strict generate path.
- **Orphaned:** _convert_request_to_complete_intent (test-only); kelly_companion (doc-only reference).
- **Unified proposals:** Single conversion helper for Request→Intent with optional emotion_map/imagery_texture on strict path; doc and tests for UMP32; single source of truth for process_intent; document schema boundary.
- **Risks:** HIGH = production path missing emotion/imagery and duplicate intent_processor drift; MEDIUM = test-only conversion and kelly_companion wiring; LOW = UMP32 parity and backup-tree data modules.
- **Test coverage (Phase 8):** Legacy conversion is partially tested (imagery_texture, key/mode, vulnerability); emotion_map, BPM/tempo_range, strict-path conversion, POST /generate, process_intent() return value, intent_bridge, and float_to_ump32 have no or minimal tests. Suggested tests for recovered logic are listed in Section G.
