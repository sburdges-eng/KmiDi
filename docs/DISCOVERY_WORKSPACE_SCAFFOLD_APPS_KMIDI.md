# Discovery: workspace-scaffold/apps/kmidi vs workspace apps/kmidi

**Date:** 2026-03-10  
**Plan:** KmiDi folders discovery — local drive scan

## Structural comparison

| Aspect | **workspace-scaffold/apps/kmidi** | **Dev/KmiDi (workspace) apps/kmidi** |
|--------|-----------------------------------|--------------------------------------|
| **Location** | `/Users/seanburdges/Dev/workspace-scaffold/apps/kmidi` | [apps/kmidi](apps/kmidi) |
| **Contents** | Full stub app: package `kmidi/`, config, data, tests, docs | Single file: [apps/kmidi/pyproject.toml](apps/kmidi/pyproject.toml) |
| **pyproject** | name = "kmidi", deps (implied by ROADMAP: sounddevice, mido, torch, numpy when implementing) | name = "kmidi", version = "0.1.0", description = "KmiDi desktop and Music Brain API app", dependencies = [] |
| **Package** | `kmidi/` with `agents/` and `tools/` | No package (no kmidi/ dir) |
| **Config** | config/datasets.toml, config/training.yaml | None |
| **Docs** | README.md, ROADMAP.md, WORKSPACE.md, COMPLETION_TODOS.md | None |

## workspace-scaffold/apps/kmidi — contents

- **kmidi/agents:** dataset_packager.py, session_hub.py, training_ops.py, training_runbook.py
- **kmidi/tools:** audio_analysis.py, dataset_packaging.py, daw.py, groove.py, harmony.py, intent.py, session.py, teaching.py
- **config:** datasets.toml, training.yaml
- **data/README.md,** **Experiments/README.md**
- **tests:** test_placeholder.py
- **ROADMAP.md:** Phases 1–4 (package/modules → datasets/config → JEPA integration → DAW scope iDAW); milestones M1–M4; deps on libs/jepa.

## Workspace apps/kmidi — contents

- **pyproject.toml only:** minimal project metadata; no Python package, no config, no tests.

## Conclusion and recommendation

- **workspace-scaffold** holds a fuller **stub** for the KmiDi + iDAW app: agents, tools, config layout, and roadmap. The workspace **apps/kmidi** is a minimal placeholder (pyproject only).
- **Recommendation:** If the monorepo is to adopt the scaffold layout for apps/kmidi, either (1) copy the scaffold’s `kmidi/` package, config, data/README, tests, and ROADMAP into [apps/kmidi](apps/kmidi) and align pyproject.toml, or (2) document workspace-scaffold as the reference layout and keep apps/kmidi as a thin placeholder until a deliberate migration. The scaffold’s ROADMAP and COMPLETION_TODOS are useful for aligning future work.
