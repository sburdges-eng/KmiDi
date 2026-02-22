# TASK-0001: Music Graph v0.1
Status: TODO
Owner: local
Epic: EPIC-01
Priority: P0
Estimate: 2d
DependsOn: none

## Goal
Define canonical schema for structure/harmony/rhythm/performance and generation metadata.

## Acceptance Criteria
- `schemas/music-graph.schema.json` validates sample projects.
- Determinism fields are required: seed + model versions + revision.
- Draft architecture notes are documented in `docs/architecture/music-graph-schema.md`.

## Notes
Start with a small strict schema, then expand after round-trip tests.
