# Emotion Schema Lock — Design Spec

**Date:** 2026-03-31
**Phase:** 1 (Contracts) of the 90-Day Demo Roadmap

## Goal

Establish a single source-of-truth emotion contract (`shared_schemas/emotion_schema.json`) and enforce it across Python, Rust, C++, and TypeScript via codegen and parity tests.

## Schema Shape

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kmidi.dev/schemas/emotion/v1",
  "title": "EmotionState",
  "type": "object",
  "properties": {
    "valence":    { "type": "number", "minimum": -1.0, "maximum": 1.0 },
    "arousal":    { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "dominance":  { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "tags": {
      "type": "array",
      "items": {
        "enum": ["tension", "release", "warm", "cold", "bright", "dark", "drive", "float"]
      },
      "maxItems": 3,
      "uniqueItems": true,
      "default": []
    },
    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
  },
  "required": ["valence", "arousal", "dominance", "confidence"],
  "additionalProperties": false
}
```

**Fields:**

| Field | Type | Range | Default | Required |
|-------|------|-------|---------|----------|
| valence | float | [-1.0, 1.0] | 0.0 | yes |
| arousal | float | [0.0, 1.0] | 0.5 | yes |
| dominance | float | [0.0, 1.0] | 0.5 | yes |
| tags | string[] | max 3, controlled vocab | [] | no |
| confidence | float | [0.0, 1.0] | 0.0 | yes |

**Allowed tags:** tension, release, warm, cold, bright, dark, drive, float

**Excluded from schema:** DSP suggestions (filter_cutoff, reverb_wet, drive_amount) — these are derived, not part of the contract.

**Deprecated fields:** `discrete_id` and `intensity` from existing EmotionState structs are dropped. One-release deprecation cycle in Rust/Python before removal.

## Codegen Pipeline

### Source of truth

Pydantic model `EmotionState` in `music_brain/engine_api/schema.py`. Running `sync_entities.py` generates:

1. `shared_schemas/emotion_schema.json` — JSON Schema (committed, reviewable)
2. `src/types/EmotionState.ts` — TypeScript interface
3. `engine/intent_ir/src/generated/emotion.rs` — Rust struct with serde + validate()
4. `music_brain/intent_ir/generated/emotion.py` — Python dataclass (re-export from Pydantic model)

### sync_entities.py changes

- Add `EmotionState` import alongside existing `CompleteSongIntentRequest`
- Add output paths: `EMOTION_SCHEMA_PATH`, `EMOTION_TS_OUT`, `EMOTION_RUST_OUT`
- Add `sync_emotion()` function following `sync_boundaries()` pattern
- Call from `__main__`

### Existing code migration

| File | Change |
|------|--------|
| `engine/intent_ir/src/types.rs` | `EmotionState` re-exports from `generated/emotion.rs`. `discrete_id`, `intensity` get `#[deprecated]` |
| `music_brain/intent_ir/__init__.py` | `EmotionState` re-exports from generated module. Old fields get `DeprecationWarning` |
| `include/penta/ml/AudioEmotionRunner.h` | `EmotionResult` unchanged (RT-safe struct, not the schema type). No tags in RT path yet |
| `include/penta/common/RTState.h` | Unchanged. `discreteEmotionId` and `emotionIntensity` stay for now (C++ deprecation is lighter) |

### C++ note

No C++ codegen. Instead, a compile-time or test-time check reads `emotion_schema.json` and verifies that RTState/EmotionResult field names and ranges match. This avoids adding a JSON Schema → C++ codegen dependency.

## Golden Fixtures

Location: `tests/fixtures/intent/`

### Valid fixtures

| File | Content |
|------|---------|
| `emotion_valid_neutral.json` | `{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "confidence": 0.5}` |
| `emotion_valid_excited.json` | `{"valence": 0.8, "arousal": 0.9, "dominance": 0.7, "tags": ["bright", "drive"], "confidence": 0.9}` |
| `emotion_valid_sad.json` | `{"valence": -0.7, "arousal": 0.2, "dominance": 0.3, "tags": ["cold", "dark"], "confidence": 0.8}` |
| `emotion_valid_max_tags.json` | `{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "tags": ["tension", "warm", "drive"], "confidence": 0.6}` |
| `emotion_valid_no_tags.json` | `{"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "confidence": 0.5}` (tags omitted entirely) |

### Invalid fixtures

| File | Expected error |
|------|---------------|
| `emotion_invalid_valence_oob.json` | valence=2.0 — out of range |
| `emotion_invalid_tag_unknown.json` | tags=["angry"] — not in vocabulary |
| `emotion_invalid_too_many_tags.json` | 4 tags — exceeds maxItems=3 |
| `emotion_invalid_extra_field.json` | has `intensity: 0.5` — additionalProperties=false |

## Parity Tests

All three language test suites load the same fixtures and must agree on pass/fail:

| Test file | Framework | What it checks |
|-----------|-----------|---------------|
| `tests/unit/test_emotion_schema.py` | pytest | Pydantic model validates/rejects each fixture correctly |
| `engine/intent_ir/src/tests/emotion_tests.rs` | cargo test | Rust struct deserialize + validate agrees with Python |
| `tests/cpp/test_emotion_schema.cpp` | Catch2 | Parse JSON fixtures, verify field names/ranges match RTState |

**CI gate:** All three must pass. Any disagreement fails the build.

## What This Does NOT Include

- DSP suggestion schema (internal to AudioEmotionRunner)
- IntentFrame schema (separate Phase 1 task, builds on this)
- Tag inference from JEPA (future — tags will be empty until a classifier is trained)
- Core ML or ONNX schema changes (inference produces latent, not EmotionState directly)

## Acceptance Criteria

1. `shared_schemas/emotion_schema.json` exists and is valid JSON Schema
2. `python3 scripts/sync_entities.py` regenerates TS + Rust from the Pydantic model without error
3. All 9 golden fixtures committed to `tests/fixtures/intent/`
4. Python tests pass: 5 valid accepted, 4 invalid rejected
5. Rust tests pass: same 9 fixtures, same outcomes
6. C++ tests pass: field/range parity check against schema
7. `discrete_id` and `intensity` marked deprecated in Rust and Python
