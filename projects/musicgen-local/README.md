# musicgen-local

Local-first development workspace for an AI music generation platform with:
- Deep controllable composition and production parameters
- Natural-language and reference-based generation/editing
- Standalone app and AU/VST3 plugin integration via a local sidecar engine

## Local Workflow
- Tasks are tracked as markdown files in `ops/board/`.
- Training/eval runs are logged in `ops/runs/`.
- Milestones and planning docs live in `ops/roadmap/`.
- Release notes and QA signoff live in `ops/releases/`.

## Board Flow
- `ops/board/todo/` -> `ops/board/in-progress/` -> `ops/board/done/`
- Use `ops/board/blocked/` for tasks waiting on dependencies.

## Initial Priorities
1. Finalize `schemas/music-graph.schema.json`
2. Build deterministic MIDI <-> graph round-trip tests
3. Scaffold sidecar IPC and JUCE plugin shell
