# Music Graph Schema (v0.1 Draft)

## Purpose
Provide a canonical intermediate representation for composition, performance, and production controls.

## Design Principles
- Deterministic: same graph + seed + model version yields reproducible outputs.
- Editable: supports graph deltas for conversational edits.
- Interoperable: compiles to MIDI, maps to audio render plans.
- Versioned: explicit schema and generation metadata.

## Top-Level Objects
- `meta`: project id, timestamps, schema version
- `global`: tempo map, time signatures, key map
- `structure`: sections, arrangement, transitions
- `tracks`: instrument roles and note/performance content
- `performance`: articulation, velocity, timing humanization
- `production`: mix macros, spatialization, fx directives
- `generation`: seed, model ids, prompt intent, constraints

## Determinism Policy
- `generation.seed` is required for all generated content.
- `generation.model_versions` records exact symbolic/audio model identifiers.
- Any edit writes a new `generation.revision` entry.

## Validation Notes
- Section boundaries must be monotonic and non-overlapping.
- Notes must satisfy `start < end` and non-negative timing.
- Key and meter changes must align to valid musical positions.
