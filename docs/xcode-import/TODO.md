# KmiDi Companion — TODO

Governance-aligned and project tasks. Single source of truth: `~/Dev`. See `.cursor/rules/engineering-governance.mdc`.

**Next 90 days started:** 2026-01-31. See `docs/PROJECT_ROADMAP.md` §4 (Next 90 days, first steps). **Phase work complete; Next 90 Days kickoff complete (2026-01-31).** This file is for ongoing governance and incremental improvement.

---

## Boot & stability (BOOT LAW)

- [x] Document deterministic boot path (Brain → penta / orchestrator / gui) — `docs/BOOT.md`
- [x] Optional: add `run_brain.py check` to CI or pre-push hook — CI runs check + stub-creep (`.github/workflows/ci.yml`); pre-push: `cp scripts/pre-push-hook.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push`
- [x] Restore or stub missing music_brain modules; avoid fragile optional imports *(Completed 2026-01-31: optional imports documented in BOOT.md.)*

## Data & paths (DATA LAW, ANTI-BLOAT)

- [x] Document paths and training manifest — `docs/DATA_AND_TRAINING.md`
- [x] One-time: confirm datasets under `~/Datasets` and checkpoints under `~/Models`; no large data in repo *(Completed 2026-01-31: all paths verified and created.)*

## Experiments (EXPERIMENT LAW)

- [x] Add `experiments/` at repo root; use `exp_NNN_description` naming — `experiments/README.md`
- [x] Policy: keep experimental code out of core until validated; promote only after review *(Completed 2026-01-31: promotion policy documented in experiments/README.md.)*

## Training (CUDA / TRAINING GOVERNANCE)

- [x] Add `configs/`, `experiments/` layout; checkpoints → `~/Models/checkpoints` (doc)
- [x] Require experiment naming + run manifest + reproducibility params — `docs/DATA_AND_TRAINING.md`
- [x] No training output in repo dirs; checkpoint/logging paths explicit

## Integration (BRIDGE PRIORITY)

- [x] Document Magenta / stem-jepa integration points in `src/kelly/integrations/` — `src/kelly/integrations/README.md`
- [x] Unify ML ↔ DSP, Python ↔ C++ boundaries where it reduces silos *(Completed 2026-01-31: 3 bridge opportunities documented in src/kelly/integrations/README.md.)*

## Housekeeping

- [x] Remove or archive stray artifacts; keep workspace small *(Completed 2026-01-31: .DS_Store files removed.)*
- [x] Avoid cloud-synced dirs for active dev; use for backup/distribution only *(Verified 2026-01-31: ~/Dev not cloud-synced.)*

---

*Stability > novelty. Clarity > expansion. Systems > fragments.*
