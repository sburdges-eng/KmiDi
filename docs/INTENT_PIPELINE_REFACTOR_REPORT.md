# Intent Pipeline Integration — Refactor Report (Phases 2–6)

## Files created

- `scripts/tests/__init__.py` — package marker for scripts tests
- `scripts/tests/test_ump_consistency.py` — cross-language float→UMP32 consistency tests
- `src/midi/README.md` — canonical float→UMP32 mappings and Python/C++ contract
- `docs/INTENT_PIPELINE_REFACTOR_REPORT.md` — this report

## Files modified

- `music_brain/api.py` — Replaced `_convert_request_to_complete_intent` body with `IntentPipeline().run(request)`; replaced generate_music strict path with pipeline (normalize → validate → expand); all `strict_intent` usages replaced with `validated`; added `IntentPipeline` import
- `music_brain/pipeline/intent_pipeline.py` — Added optional `normalized` parameter to `expand()`; implemented explicit precedence (validated > normalized > raw request) for all expand fields; `run()` now passes `normalized` into `expand()`
- `music_brain/kelly_companion/session/intent_processor.py` — Added SAFE_TO_DELETE comment at top
- `music_brain/kelly_companion/kellymidicompanion_session/kellymidicompanion_intent_processor.py` — Added SAFE_TO_DELETE comment at top

## Duplicate modules “removed”

- **Marked SAFE_TO_DELETE** (not physically removed):
  - `music_brain/kelly_companion/session/intent_processor.py`
  - `music_brain/kelly_companion/kellymidicompanion_session/kellymidicompanion_intent_processor.py`
- Canonical processor: `music_brain/session/intent_processor.py`. No other modules import the two duplicates.

## Features restored

- Single deterministic path: all request→intent conversion goes through `IntentPipeline` (run or normalize→validate→expand).
- `imagery_texture` and other Stage 3 fields preserved; no silent feature loss between test and production paths.
- Validated fields (tempo, key_mode, structure, instruments, allow_legacy_fallback) still drive MIDI generation and response; source is now `validated` from the pipeline.

## Behavior changes

- **None intended.** Precedence rule (validated > normalized > raw) is enforced in `expand()`; validated values are never overwritten by inferred or raw request values. MIDI and response logic unchanged except variable name (`validated` instead of `strict_intent`).

## Success criteria

| Criterion | Status |
|-----------|--------|
| Single deterministic pipeline | All conversion via IntentPipeline |
| No feature loss | test_ui_mapping passes; imagery_texture and validated fields preserved |
| No duplicate intent processors in use | Canonical only; duplicates marked SAFE_TO_DELETE |
| Verified cross-language parity | test_ump_consistency passes; README documents mappings |
| No runtime boundary violations | Validated never overwritten; strict path removed in favor of pipeline |
