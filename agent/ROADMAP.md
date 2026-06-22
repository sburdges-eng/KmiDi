# KmiDi roadmap

## Vision

Audio and DAW app (KmiDi + iDAW): audio I/O, MIDI, optional JEPA-based representations, and a clear path to training/eval configs for audio models.

## Phases

| Phase | Focus | Outcomes |
|-------|--------|----------|
| **1** | Package and modules | kmidi package with audio.py, midi.py; pyproject deps; placeholder main or CLI. |
| **2** | Datasets and config | Dataset config (paths, splits); data/ layout; training config stub for JEPA. |
| **3** | JEPA integration | Use libs/jepa for audio embeddings; training script or entrypoint; completion todos done. |
| **4** | DAW scope (iDAW) | Optional daw.py (transport, project); docs and runbook. |

## Milestones

- **M1** — `kmidi` package importable; `uv run pytest apps/kmidi/tests -v` passes with at least one real test.
- **M2** — config/datasets.toml and config/training.yaml in place; data/README describes expected layout.
- **M3** — JEPA training runs from config (or documented as manual); completion checklist up to date.
- **M4** — iDAW scope documented; roadmap and COMPLETION_TODOS aligned.

## Dependencies between apps/libs

- **jepa** (libs/jepa) for audio JEPA training and inference.
- Optional: **sounddevice**, **mido**, **torch**, **numpy** (add in pyproject when implementing).

## Risks and mitigations

- Audio hardware / latency → document supported backends; make training config point to file-based data first.
- JEPA not yet implemented → keep training config as stub; Phase 3 depends on libs/jepa TODO.md.
