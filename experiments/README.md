# Experiments

Experimental code lives here. **Promote to core only after validation.** (Governance: EXPERIMENT LAW.) Roadmap: [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md), [docs/PROJECT_ROADMAP_REIMPLEMENTATION.md](docs/PROJECT_ROADMAP_REIMPLEMENTATION.md).

## Research (reasoning required)

- **`research/`** — Design and strategy docs only (no code). **All research docs must include reasoning** — why a model type fits, why local vs cloud, why a training path — not only tables or conclusions. See [research/README.md](research/README.md).
- **Index:** [AI types → project mapping](research/ai_types_project_mapping.md); [local vs cloud deployment](research/local_vs_cloud_deployment.md) (low-latency local on 16 GB Mac, plugin/DAW; train local first, then cloud using most effective method; cloud financially sustainable / future self-hosted GPU).

## Naming

Use: `exp_NNN_short_description` (e.g. `exp_001_emotion_encoder`, `exp_002_groove_ablation`).

- `NNN` = zero-padded number (001, 002, …).
- `short_description` = lowercase, underscores.

## Layout

```
experiments/
  research/                    # reasoning + strategy; no code
    README.md
    TEMPLATE_proposed_model.md
    ai_types_project_mapping.md
    local_vs_cloud_deployment.md
  exp_001_description/
    README.md       # goal, setup, results summary; reference research if applicable
    config.yaml     # optional
    ...             # code / notebooks
  exp_002_other/
    ...
```

Do not scatter experimental code across `src/` or `KmiDi_CANON/`. Keep experiments isolated; merge into core only after review. When an experiment implements a research idea, its README should reference the research doc and state which assumptions are being tested.

## Promotion Policy

**Experiments remain isolated until validated.** Promotion to core requires:

1. **Success criteria met** — Experiment README documents success metrics; results meet or exceed targets.
2. **Code quality** — Clean, documented, tested code ready for integration.
3. **Review** — At least one reviewer (maintainer or designated peer) approves promotion.
4. **Integration plan** — Clear path for merging into core without breaking spine; update contracts if API changes.
5. **Deprecation** — Original experiment dir archived or marked complete; not deleted (kept for audit).

**Review checklist:**
- [ ] Success criteria documented and met
- [ ] Code passes linter, tests written for critical paths
- [ ] Integration doesn't break `run_brain.py check` or existing tests
- [ ] Contracts/BOOT updated if new module or API
- [ ] Experiment dir marked as promoted (add `PROMOTED.md` noting target and date)

## References

- [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) — current status, next 90 days, governance alignment.
- [docs/PROJECT_ROADMAP_REIMPLEMENTATION.md](docs/PROJECT_ROADMAP_REIMPLEMENTATION.md) — spine, phases, and doc lifecycle; experiments layout and research docs align with roadmap.
