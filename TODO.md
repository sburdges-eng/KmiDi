# KmiDi Companion — TODO

Governance-aligned and project tasks. Single source of truth: `~/Dev`. See `.cursor/rules/engineering-governance.mdc`.

## Boot & stability (BOOT LAW)

- [ ] Document deterministic boot path (Brain → penta / orchestrator / gui)
- [ ] Add `run_brain.py check` to CI or pre-push hook if desired
- [ ] Restore or stub music_brain modules if missing; avoid fragile optional imports

## Data & paths (DATA LAW, ANTI-BLOAT)

- [ ] Confirm datasets live under `~/Datasets`; no large data in repo
- [ ] Confirm model weights / checkpoints under `~/Models` (or `~/Models/checkpoints`)
- [ ] Use symlinks for external data if needed; document in README

## Experiments (EXPERIMENT LAW)

- [ ] Add `experiments/` at repo root; use `exp_NNN_description` naming
- [ ] Keep experimental code out of core until validated; promote only after review

## Training (CUDA / TRAINING GOVERNANCE)

- [ ] Add `configs/`, `experiments/`, `checkpoints/` layout for GPU runs
- [ ] Require experiment naming + run manifest + reproducibility params for any training
- [ ] No training output in repo dirs; checkpoint/logging paths explicit

## Integration (BRIDGE PRIORITY)

- [ ] Unify ML ↔ DSP, Python ↔ C++ boundaries where it reduces silos
- [ ] Document Magenta / stem-jepa integration points in `src/kelly/integrations/`

## Housekeeping

- [ ] Remove or archive stray artifacts; keep workspace small
- [ ] Avoid cloud-synced dirs for active dev; use for backup/distribution only

---

*Stability > novelty. Clarity > expansion. Systems > fragments.*
