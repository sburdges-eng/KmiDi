# IntentFrame Schema Lock — Design Spec

**Date:** 2026-03-31
**Phase:** 1 (Contracts) of the 90-Day Demo Roadmap

## Goal

Establish `shared_schemas/intent_frame_schema.json` as the single source-of-truth IntentFrame contract, with codegen to TS/Rust and parity tests across Python and Rust. Mirrors the emotion schema lock pattern.

## Schema Shape

IntentFrame is a nested struct with 6 sub-structs. All field names, types, and ranges match the existing Rust types in `engine/intent_ir/src/types.rs`.

### IntentMetaSchema

| Field | Type | Constraint | Default |
|-------|------|-----------|---------|
| ir_version | int | == 1 | 1 |
| intent_id | int | >= 0 | 0 |
| session_id | int | >= 0 | 0 |

### EmotionStateSchema

Already locked in `music_brain/engine_api/schema.py`. Referenced by `IntentFrameSchema` via composition. The JSON Schema uses `$ref`.

### MusicalIntentSchema

| Field | Type | Range | Default |
|-------|------|-------|---------|
| tempo_bias | float | [-1.0, 1.0] | 0.0 |
| rhythmic_density | float | [0.0, 1.0] | 0.5 |
| groove_strength | float | [0.0, 1.0] | 0.5 |
| harmonic_tension | float | [0.0, 1.0] | 0.5 |
| harmonic_motion | float | [0.0, 1.0] | 0.5 |
| mode_preference | int | {-1, 0, 1} | 0 |
| melodic_activity | float | [0.0, 1.0] | 0.5 |
| contour_variance | float | [0.0, 1.0] | 0.5 |
| dynamic_range | float | [0.0, 1.0] | 0.5 |
| texture_density | float | [0.0, 1.0] | 0.5 |

### TimeScopeSchema

| Field | Type | Constraint | Default |
|-------|------|-----------|---------|
| start_bar | int | -1 = immediate | -1 |
| end_bar | int | -1 = open-ended; if both set, end > start | -1 |
| fade_in_beats | float | >= 0.0 | 0.0 |
| fade_out_beats | float | >= 0.0 | 0.0 |

### IntentConstraintsSchema

| Field | Type | Constraint | Default |
|-------|------|-----------|---------|
| allowed_engines_mask | int | >= 0 | 0xFFFFFFFF |
| forbidden_engines_mask | int | >= 0 | 0 |
| max_cpu_cost | float | >= 0.0 | 1.0 |
| max_event_rate | float | >= 0.0 | 1000.0 |

### IntentProvenanceSchema

| Field | Type | Constraint | Default |
|-------|------|-----------|---------|
| source | int | 0-5 (enum: UiDirect, UiEdit, MlText, MlAudio, Preset, Automation) | 0 |
| user_override_weight | float | [0.0, 1.0] | 0.5 |

### IntentFrameSchema (top-level)

| Field | Type | Required |
|-------|------|----------|
| meta | IntentMetaSchema | yes (has defaults) |
| emotion | EmotionStateSchema | yes (has defaults) |
| music | MusicalIntentSchema | yes (has defaults) |
| time | TimeScopeSchema | yes (has defaults) |
| constraints | IntentConstraintsSchema | yes (has defaults) |
| provenance | IntentProvenanceSchema | yes (has defaults) |

`additionalProperties: false` on all sub-structs and the top-level.

## Codegen Pipeline

### Source of truth

Pydantic models in `music_brain/engine_api/schema.py`. Running `sync_entities.py` generates:

1. `shared_schemas/intent_frame_schema.json` — JSON Schema
2. `src/types/IntentFrame.ts` — TypeScript interfaces for all 7 types
3. `src-tauri/src/generated/intent_frame.rs` — Rust serde structs with `validate()` + `deny_unknown_fields`

### sync_entities.py changes

- Add Pydantic model imports for all IntentFrame sub-structs
- Add `sync_intent_frame()` function
- Reuse `EmotionStateSchema` via composition (Pydantic `$ref`)
- Call from `__main__`

### Existing code unchanged

| File | Status |
|------|--------|
| `engine/intent_ir/src/types.rs` | Untouched — `#[repr(C)]` FFI types |
| `music_brain/intent_ir/__init__.py` | Untouched — runtime dataclasses with `to_json()`/`from_json()` |
| `engine/intent_ir/src/validator.rs` | Untouched — FFI validation |

## Golden Fixtures

Location: `tests/fixtures/intent/`

### Valid

| File | Content |
|------|---------|
| `frame_valid_default.json` | All sub-structs with defaults |
| `frame_valid_full.json` | All fields non-default, 2 emotion tags |
| `frame_valid_ml_audio.json` | source=3 (MlAudio), low confidence, high arousal |

### Invalid

| File | Expected rejection |
|------|-------------------|
| `frame_invalid_version.json` | ir_version=99 |
| `frame_invalid_tempo_oob.json` | tempo_bias=5.0 |
| `frame_invalid_time_scope.json` | end_bar < start_bar (both set) |
| `frame_invalid_extra_field.json` | extra top-level field |

## Parity Tests

| Test file | Framework | Validates |
|-----------|-----------|-----------|
| `tests/unit/test_intent_frame_schema.py` | pytest | Pydantic validates/rejects all 7 fixtures |
| `src-tauri/tests/test_intent_frame_schema.rs` | cargo test | Rust serde + validate() agrees on all 7 fixtures |

## What This Does NOT Include

- Changes to existing `intent_ir` FFI types or dataclasses
- DSP suggestion schema (internal to AudioEmotionRunner)
- C++ parity tests (C++ uses FFI types, not schema types)
- Runtime serialization changes (`to_json`/`from_json` on the dataclass)

## Acceptance Criteria

1. `shared_schemas/intent_frame_schema.json` exists and is valid JSON Schema with all 6 sub-structs
2. `python3 scripts/sync_entities.py` regenerates TS + Rust without error
3. All 7 golden fixtures committed
4. Python tests: 3 valid accepted, 4 invalid rejected
5. Rust tests: same 7 fixtures, same outcomes
6. Existing `intent_ir` code untouched — no regressions
