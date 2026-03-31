# IntentFrame Schema Lock — Design Spec (v2)

**Date:** 2026-03-31
**Phase:** 1 (Contracts) of the 90-Day Demo Roadmap

## Goal

Establish `shared_schemas/intent_frame_schema.json` as the single source-of-truth IntentFrame contract with all roadmap-required fields, codegen to TS/Rust, and parity tests across Python and Rust.

## Schema Shape

IntentFrame has 8 sub-structs: the 6 existing ones from `types.rs` plus 2 new ones (DSPTargets, MusicHints). New top-level fields: `timestamp_ms` and `latency_budget_ms`.

### IntentMetaSchema

| Field | Type | Constraint | Default |
|-------|------|-----------|---------|
| schema_version | int | == 1 | 1 |
| intent_id | int | >= 0 | 0 |
| session_id | int | >= 0 | 0 |

Note: renamed `ir_version` → `schema_version` for clarity in the contract.

### EmotionStateSchema

Already locked. Referenced via composition. Fields: valence [-1,1], arousal [0,1], dominance [0,1], tags (max 3), confidence [0,1].

### MusicalIntentSchema (existing fields)

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

### MusicHintsSchema (NEW — roadmap required)

| Field | Type | Constraint | Default |
|-------|------|-----------|---------|
| key | string | e.g. "C", "F#", "" = unspecified | "" |
| tempo_bpm | float | >= 0.0 (0 = unspecified) | 0.0 |
| chord_bias | string | e.g. "minor7", "" = none | "" |
| section_role | string | enum: "intro", "verse", "chorus", "bridge", "outro", "build", "drop", "" | "" |

### DSPTargetsSchema (NEW — roadmap required)

Each target has a value + confidence + stale flag for confidence propagation.

| Field | Type | Range | Default (safe) |
|-------|------|-------|----------------|
| filter_cutoff | float | [0.0, 1.0] | 0.5 |
| filter_cutoff_confidence | float | [0.0, 1.0] | 0.0 |
| reverb_send | float | [0.0, 1.0] | 0.2 |
| reverb_send_confidence | float | [0.0, 1.0] | 0.0 |
| drive | float | [0.0, 1.0] | 0.0 |
| drive_confidence | float | [0.0, 1.0] | 0.0 |
| stale | bool | — | true |

**Safety fallbacks:** All defaults are deterministic safe values (filter mid-open, reverb subtle, drive off, stale=true). When `stale=true` or all confidences are 0, consumers must hold their last-known-good or use these defaults.

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
| source | int | 0-5 (UiDirect..Automation) | 0 |
| user_override_weight | float | [0.0, 1.0] | 0.5 |

### IntentFrameSchema (top-level)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| meta | IntentMetaSchema | yes | Version, IDs |
| timestamp_ms | int | yes | Monotonic ms since session start, default=0 |
| emotion | EmotionStateSchema | yes | VAD + tags + confidence |
| music | MusicalIntentSchema | yes | Biases and tendencies |
| music_hints | MusicHintsSchema | yes | Key, tempo, chord, section (NEW) |
| dsp_targets | DSPTargetsSchema | yes | Filter, reverb, drive + confidence (NEW) |
| time | TimeScopeSchema | yes | Bar scope, fades |
| constraints | IntentConstraintsSchema | yes | Engine limits |
| provenance | IntentProvenanceSchema | yes | Source, override weight |
| latency_budget_ms | float | yes | Max ms for RT engine, default=10.0 |

`additionalProperties: false` on all sub-structs and top-level.

## Serialization Note

This JSON Schema defines the **contract** — what fields exist, their types, ranges, and defaults. On the RT hot path (AU plugin ↔ engine), the C `IntentFrame` struct in `intent_ir` is used directly (flat memory, no serialization). JSON is for validation, codegen, fixtures, and non-RT interchange. No JSON parsing on the audio thread.

## Codegen Pipeline

Same pattern as EmotionState:

1. Pydantic models in `music_brain/engine_api/schema.py`
2. `sync_entities.py` generates:
   - `shared_schemas/intent_frame_schema.json`
   - `src/types/IntentFrame.ts`
   - `src-tauri/src/generated/intent_frame.rs`

Existing `intent_ir` FFI types untouched. New fields (music_hints, dsp_targets, timestamp_ms, latency_budget_ms) exist only in the schema contract for now — they'll be added to the C struct when the AU plugin needs them.

## Golden Fixtures

| File | Purpose |
|------|---------|
| `frame_valid_default.json` | All defaults — safe fallback state |
| `frame_valid_full.json` | All fields non-default, DSP confident, stale=false |
| `frame_valid_ml_audio.json` | source=MlAudio, stale DSP, low confidence |
| `frame_invalid_version.json` | schema_version=99 |
| `frame_invalid_tempo_oob.json` | tempo_bias=5.0 |
| `frame_invalid_time_scope.json` | end_bar < start_bar |
| `frame_invalid_extra_field.json` | extra top-level field |

## Parity Tests

- Python: `tests/unit/test_intent_frame_schema.py` — Pydantic validates all 7
- Rust: `src-tauri/tests/test_intent_frame_schema.rs` — serde + validate() on all 7

## Acceptance Criteria

1. JSON Schema exists with all 8 sub-structs + 2 new top-level fields
2. `sync_entities.py` regenerates TS + Rust without error
3. All 7 golden fixtures committed
4. Python: 3 valid accepted, 4 invalid rejected
5. Rust: same 7 fixtures, same outcomes
6. DSP targets have per-parameter confidence + stale flag
7. All DSP defaults are deterministic safe values
8. Existing `intent_ir` code untouched
