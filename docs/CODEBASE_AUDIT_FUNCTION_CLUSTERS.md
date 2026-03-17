# Codebase Audit: Function Clusters & Reconstruction Plan

**Scope:** Duplicate implementations, lost features, and runtime-class boundaries across the merged KmiDi repository.  
**Rule:** Every claim cites actual file paths and code. No merging across REALTIME_SAFE / NON_REALTIME.

---

## Reality Check

This audit is not “clean code.” It aims to:

- Recover **lost intent** (logic present in one implementation but not the dominant path)
- Preserve **edge-case intelligence** (validation, clamping, emotion mapping)
- Eliminate **parallel evolution bugs** (same name, different behavior)

---

## FUNCTION CLUSTER: Request → CompleteSongIntent (Generate Path)

### Files Found

- `music_brain/api.py` — `DAiWAPI._convert_request_to_complete_intent` (lines 519–593), and inline `_convert_to_intent` inside `generate_music` (lines 1482–1512)

### Runtime Class

- **NON_REALTIME** (Python, HTTP request handling)

### Key Differences

- **`_convert_request_to_complete_intent(request)`**  
  - Input: raw `GenerateRequest` (no prior validation).  
  - Uses `emotion_map` to map free-text `emotional_intent` to canonical mood (grief, tenderness, rage, fear, nostalgia, awe).  
  - Sets `core_event` from `core_wound or emotional`; `imagery_texture` from `request.intent.imagery_texture`; `narrative_arc` from `request.intent.narrative_arc` or tech.  
  - Parses `technical.key` ("F major" → key + mode), validates mode against `VALID_MUSICAL_MODES`, clamps BPM 40–300 and derives `tempo_range` (bpm±20, clamped 60–140).  
  - Normalizes `vulnerability_scale` (float 0–1 or string Low/Medium/High).  
  - **Not used on the current POST /generate path** (only by tests, e.g. `tests/unit/test_ui_mapping.py`).

- **`_convert_to_intent(request, strict_intent)`** (inside `generate_music`)  
  - Input: `GenerateRequest` + **already-validated** `CompleteSongIntentRequest` from `music_brain.engine_api.schema`.  
  - No emotion_map; uses `validated.mood_primary` and `validated.core_desire` directly.  
  - No `imagery_texture` (not in `CompleteSongIntentRequest`).  
  - Key/mode from `validated.key_mode` split; tempo_range from `validated.tempo ± 20`.  
  - **This is the only conversion used when the UI calls POST /generate.**

### Lost Features (LOST_FEATURE_CANDIDATE)

- **Emotion mapping:** The explicit `emotion_map` in `_convert_request_to_complete_intent` (e.g. "sadness"→"grief", "happiness"→"tenderness") is never applied on the live generate path.  
  - Code reference: `music_brain/api.py` lines 538–553 (`emotion_map` and loop over `emotional.lower()`).
- **imagery_texture:** Set only in `_convert_request_to_complete_intent` via `getattr(request.intent, "imagery_texture", "")`. The strict pipeline does not pass it; `IntentProcessor.generate_production()` uses `self.imagery` (from intent), so production guidelines lose that input when using the strict path.  
  - Code reference: `music_brain/api.py` 569–571 (assign to intent) vs 1482–1512 (no imagery_texture).
- **core_event from core_wound or emotional:** The strict path sets `core_event=req.intent.core_wound or validated.core_desire`; the legacy path uses `core_wound or emotional` and mood extraction from parentheses in emotional string (e.g. "joy (bright)" → mood_primary "tenderness").  
  - Code reference: `music_brain/api.py` 534–536 vs 919 (`core_event=req.intent.core_wound or validated.core_desire`).

### Recommended Unified Approach (Same Runtime)

- **Do not merge** the two into one implementation; they serve different entry points (unvalidated vs validated).
- **Canonical behavior spec:** Document in one place: (1) allowed modes, (2) BPM and tempo_range rules, (3) emotion_map and when it applies, (4) mapping of `CompleteSongIntentRequest` → `CompleteSongIntent` field list.
- **Restore lost behavior on the strict path:**  
  - After building `complete_intent` in `_convert_to_intent`, optionally set `complete_intent.song_intent.imagery_texture` from `request.intent.imagery_texture` if present.  
  - If you want emotion normalization on the strict path, apply the same `emotion_map` to `request.intent.emotional_intent` and use it to override or default `mood_primary` when the UI sends free text.
- **Single conversion helper:** Refactor so both paths call one shared helper that takes (request.intent, validated: Optional[CompleteSongIntentRequest]) and returns CompleteSongIntent, with branches for “from validated” vs “from raw request” (emotion_map, imagery_texture, key parsing). That preserves all current behavior and recovers the lost fields without breaking the strict validation path.

### Missing Features (Summary)

- C++/Rust: N/A (this cluster is Python-only).
- TS: `useMusicBrain.ts` builds `GenerateRequest` and calls POST /generate; it does not perform conversion. No duplication in frontend.

---

## FUNCTION CLUSTER: Float to UMP32 (Affect / MIDI 2.0 CC)

### Runtime Classes

- **REALTIME_SAFE:** `src/midi/AffectUMP.cpp` — `floatToUmp32Value`, `buildUmpAffectCC`
- **NON_REALTIME:** `scripts/ump_affect_utils.py` — `float_to_ump32`, `build_ump_cc_v2`

### Shared Intent

- Map a float in `[min, max]` to 32-bit UMP value `[0, 0xFFFFFFFF]`: clamp then linear normalize, round.
- Build MIDI 2.0 Channel Voice Control Change (32-bit) packet for affect (valence, arousal, dynamics).

### Files Found

- `src/midi/AffectUMP.h` — declarations
- `src/midi/AffectUMP.cpp` — C++ implementation (uses `std::clamp`, JUCE Factory when available)
- `scripts/ump_affect_utils.py` — Python implementation (used by `ump_affect_harness.py`, `morph_affect_bridge.py`)

### Key Differences

- **Clamp/normalize:** Identical logic (minVal >= maxVal → 0; clamp; norm = (clamped - min) / (max - min); round to 32-bit). C++ uses `std::clamp` and explicit cast; Python uses `max(min_val, min(max_val, value))`.
- **C++:** `buildUmpAffectCC` uses JUCE `Factory::makeControlChangeV2` when `JUCE_MODULE_AVAILABLE_juce_audio_basics`; else writes zeros. Python always builds the 2-word packet manually (byte layout documented in script).
- **Constants:** CC indices and control rate match (`kCCValence` 0x28, etc. in header; `CC_VALENCE = 0x28` in Python).

### Lost Features

- None identified. Python comment states “Must match src/midi/AffectUMP.h and AffectUMP.cpp.”

### Recommendation

- **Do NOT merge** implementations across runtime boundary.
- **Canonical spec:** One short doc (e.g. in `src/midi/README.md` or `docs/`) defining: (1) formula for float → UMP32, (2) CC indices and ranges (valence [-1,1], arousal/dynamics [0,1]), (3) that Python and C++ must stay in sync for harness/bridge scripts.
- **Tests:** Add a small Python test that runs `float_to_ump32` for a few (value, min, max) triples and compares outputs to expected 32-bit values; optionally document one C++ and one Python output for the same inputs to catch drift.

---

## FUNCTION CLUSTER: process_intent (Intent Execution)

### Files Found

- `music_brain/session/intent_processor.py` — `process_intent(intent: CompleteSongIntent) -> Dict` (line 719), `IntentProcessor(intent).generate_all()`
- `music_brain/session/intent_bridge.py` — `process_intent(intent_json: str) -> str` (JSON in → JSON out in C++-oriented format)
- `music_brain/kelly_companion/session/intent_processor.py` — same `process_intent(intent: CompleteSongIntent)` at line 719 (duplicate)
- `music_brain/kelly_companion/kellymidicompanion_session/kellymidicompanion_intent_processor.py` — same at line 719 (duplicate)
- `music_brain/kelly/core/intent_processor.py` — `process_intent(self, wound: Wound) -> Dict` (different type: Wound, not CompleteSongIntent)
- `music_brain/api.py` — calls `process_intent(intent)` (line 715) after conversion
- `music_brain/misc_code/api.py` — calls `process_intent(intent)` (line 480)
- `music_brain/api_misc.py` — calls `process_intent(intent)` (line 484)

### Runtime Class

- **NON_REALTIME** (Python only for this cluster)

### Key Differences

- **Session intent_processor:** Takes `CompleteSongIntent`, uses `intent.song_root`, `intent.song_intent`, `intent.technical_constraints` (e.g. `technical_groove_feel`, `rule_breaking_justification`), and `IntentProcessor` attributes derived from intent (key, mode, tempo, rule_to_break, narrative_arc, vulnerability, imagery). Returns dict with harmony, groove, arrangement, production, intent_summary.
- **intent_bridge:** Wraps session: parses JSON → `CompleteSongIntent.from_dict` → `_intent_processor.process_intent(intent)` → `_convert_to_cpp_format(result)`. Different I/O (string JSON), same core logic.
- **Kelly intent_processor:** Different domain (Wound → Dict). Not the same intent; different cluster by semantics.
- **kelly_companion copies:** Appear to be duplicate copies of `music_brain/session/intent_processor.py` (same line numbers and function name). Risk of divergence if only one is updated.

### Lost Features

- None confirmed as unique to one implementation; the duplicate copies in kelly_companion could have drifted (not diffed here).

### Recommendation

- **Unify call sites:** Ensure all API entry points that need “full song intent → harmony/groove/arrangement/production” call `music_brain.session.intent_processor.process_intent` (or the same module). Verify `misc_code/api.py` and `api_misc.py` import from `music_brain.session.intent_processor`.
- **Duplicate modules:** Treat `music_brain/kelly_companion/session/intent_processor.py` and `kellymidicompanion_session/kellymidicompanion_intent_processor.py` as **SALVAGEABLE_MODULE** only if they are intentionally used by another product; otherwise consider removing or re-exporting from `music_brain.session.intent_processor` to avoid parallel evolution.
- **intent_bridge:** Keep as thin wrapper: JSON → CompleteSongIntent → session `process_intent` → C++-format JSON. No need to merge with session logic.

---

## FUNCTION CLUSTER: Intent Schema (CompleteSongIntent vs CompleteSongIntentRequest)

### Files Found

- `music_brain/session/intent_schema.py` — `CompleteSongIntent` (dataclass with song_root, song_intent, technical_constraints, flat __init__ kwargs)
- `music_brain/engine_api/schema.py` — `CompleteSongIntentRequest` (Pydantic: core_desire, mood_primary, genre, tempo, key_mode, structure, instruments, allow_legacy_fallback, groove_feel, narrative_arc, rule_to_break, rule_justification)
- `src/types/Intent.ts` — `CompleteSongIntentRequest` (synced from shared_schemas via `scripts/sync_entities.py`)
- `src-tauri/src/generated/intent.rs` — generated from same schema

### Runtime Classes

- **NON_REALTIME:** Python (session + API), TypeScript (UI), Rust (Tauri bridge)

### Key Differences

- **CompleteSongIntentRequest (engine_api + TS + Rust):** Flat, UI/engine boundary; strict validation (structure pattern, key_mode pattern, min_lengths). Used for HTTP and frontend.
- **CompleteSongIntent (session):** Nested (song_root, song_intent, technical_constraints); used by intent_processor and DAiWAPI. Has more Phase 0/1 fields (core_resistance, core_stakes, core_transformation, imagery_texture, mood_secondary_tension) and validation in `validate_intent()`.
- **Mapping:** API converts Request → Intent via the two conversion paths above. `CompleteSongIntent` is never sent over the wire; it is the internal engine shape.

### Lost Features

- Already covered under “Request → CompleteSongIntent”: imagery_texture, emotion_map, and richer Phase 0/1 fields are not populated when using the strict Request path.

### Recommendation

- Keep two shapes: (1) **CompleteSongIntentRequest** = boundary contract (shared_schemas, sync_entities, engine_api); (2) **CompleteSongIntent** = internal processing shape. Document the single intended conversion path and which fields are preserved (and which are only in the legacy conversion).

---

## DEAD CODE / SALVAGEABLE

### SALVAGEABLE_MODULE

- **`DAiWAPI._convert_request_to_complete_intent`** — Contains emotion_map, imagery_texture, and key/mode/tempo validation used by tests. Either: (a) use it on a code path that accepts unvalidated request (e.g. a “legacy” or “debug” generate endpoint), or (b) extract emotion_map and imagery_texture handling into a shared helper and call it from the strict-path conversion so production generation gets the same behavior.
- **`music_brain/kelly_companion/`** and **kellymidicompanion_session** intent_processor copies — Salvageable only if a separate product depends on them; otherwise re-export from `music_brain.session.intent_processor` or delete to avoid drift.

### SAFE_TO_DELETE (after verification)

- No file recommended for deletion without verifying no imports or runtime entry points. The duplicate intent_processor files should be checked for imports before removal.

---

## Optional: Multi-Language Note

The repo includes Python (music_brain, scripts), TypeScript (src), Rust (src-tauri), and C++ (engine, src/midi, plugins). Clusters that span languages (e.g. Affect UMP) are classified by runtime and not merged across REALTIME_SAFE / NON_REALTIME. Shared intent is documented so each language keeps behavioral parity without violating constraints.

---

## Summary Table

| Cluster                         | Runtime boundary | Lost feature candidates                          | Action |
|---------------------------------|------------------|---------------------------------------------------|--------|
| Request → CompleteSongIntent    | NON_REALTIME     | emotion_map, imagery_texture, core_event/mood     | Restore in strict path or unify conversion |
| Float → UMP32 / Affect CC       | REALTIME vs non  | None                                              | Doc spec; keep separate impls |
| process_intent                  | NON_REALTIME     | Possible drift in kelly_companion copies          | Single source of truth; re-export or remove copies |
| Intent schema (Request vs Intent) | All non-RT     | Field loss in strict path                         | Document; fix conversion as above |
