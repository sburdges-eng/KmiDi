# KmiDi completion todos

Check off when done. Use with ROADMAP.md and WORKSPACE.md.

## Phase 1 — Package and modules

- [ ] `kmidi` package exists with `kmidi/__init__.py`, `kmidi/audio.py`, `kmidi/midi.py`.
- [ ] `uv run pytest apps/kmidi/tests -v` passes with at least one non-placeholder test.
- [ ] pyproject.toml includes any new deps (e.g. sounddevice, mido, jepa) as needed.
- [ ] main.py or CLI entry exists and is documented in README.

## Phase 2 — Datasets and config

- [ ] config/datasets.toml (or equivalent) describes dataset paths and splits.
- [ ] data/README describes expected audio/MIDI data layout.
- [ ] config/training.yaml stub present for JEPA (or documented as Phase 3).

## Phase 3 — JEPA integration

- [ ] libs/jepa is used for audio embeddings (or integration is documented and deferred).
- [ ] Training script or entrypoint runs from config (or runbook documents manual steps).
- [ ] COMPLETION_TODOS for Phase 1–2 all checked.

## Phase 4 — DAW scope (iDAW)

- [ ] daw.py or equivalent scoped; optional transport/project features documented.
- [ ] Roadmap and completion todos aligned; no critical open items for Phase 1–3.

## Definition of done (release)

- [ ] Phase 1–2 complete; Phase 3 either complete or explicitly deferred with runbook.
- [ ] README and WORKSPACE.md list only optional/future work for Phase 4.
