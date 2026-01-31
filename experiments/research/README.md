# Research

Research docs live here. **Status:** Research docs current as of 2026-01-31. **All research documents must include reasoning** — not only tables or conclusions. (Governance: EXPERIMENT LAW; signal-driven discovery.)

**Status:** Research docs current as of 2026-01-31.

## Purpose

- **Reasoning required:** Each doc must explain *why*: why a model type fits a project area, why local vs cloud, why a training path, etc. Summaries and tables are supporting; reasoning is mandatory.
- **No code:** Research is design and strategy only. Experimental code belongs in `exp_NNN_description/` under `experiments/`.
- **Traceability:** When an experiment implements a research idea, the experiment README should reference the research doc and state which assumptions are being tested.

## Layout

```
experiments/research/
  README.md                      # this file
  TEMPLATE_proposed_model.md     # template for one proposed model; copy to <short_name>_proposed.md
  ai_types_project_mapping.md    # AI types → project; reasoning per type
  local_vs_cloud_deployment.md   # low-latency local vs cloud; training path; sustainability
  ...                            # future research docs (e.g. groove_vae_proposed.md)
```

## Index

| Doc | Summary |
|-----|---------|
| [TEMPLATE_proposed_model.md](TEMPLATE_proposed_model.md) | Template for a single proposed model; copy to create new research docs (reasoning, deployment, traceability). |
| [ai_types_project_mapping.md](ai_types_project_mapping.md) | Broad AI types (neural, AR/LLM, diffusion, VAE, flow, GAN, JEPA) and where they fit in KmiDi; reasoning for each. |
| [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md) | Local low-latency (16 GB macOS, plugin/DAW) vs cloud/self-hosted; train local first, then cloud; financial sustainability. |

---

## Adding a new proposed-model research doc

When proposing a new model (e.g. groove VAE, intent flow, next-section JEPA), follow these steps so the doc is integrated with the research index and cross-links.

1. **Create the doc** — Copy [TEMPLATE_proposed_model.md](TEMPLATE_proposed_model.md) to `experiments/research/<short_name>_proposed.md` (e.g. `groove_vae_proposed.md`, `intent_flow_proposed.md`). Fill in all sections; keep reasoning mandatory.
2. **Index it** — Add a row to the **Index** table above (Doc + Summary).
3. **Layout** — Add the new filename to the Layout list above under "future research docs".
4. **Cross-link from type/deployment** — If the model instantiates a specific AI type or deployment case, add a one-line pointer in [ai_types_project_mapping.md](ai_types_project_mapping.md) (e.g. in the "Where in project" or "Future" bullet for that type) or in [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md) (e.g. in the summary table or checklist). Format: "See [short_name_proposed.md](short_name_proposed.md)."
5. **Optional: experiments/README** — If the model is a first-class research theme, add a line to the Research index in [experiments/README.md](../README.md) linking the new doc.
6. **When an experiment implements it** — In `exp_NNN_description/README.md`, reference the research doc and state which assumptions are being tested (already required by experiments/README; reinforce here).

### Integration checklist

- [ ] Doc created from TEMPLATE; all sections filled; reasoning present.
- [ ] Row added to Index in this README.
- [ ] Filename added to Layout in this README.
- [ ] One-line pointer added in ai_types_project_mapping.md or local_vs_cloud_deployment.md (if applicable).
- [ ] Optional: line added to experiments/README Research index.
- [ ] When exp_NNN implements: exp README references this research doc and states assumptions tested.

---

## References

- [experiments/README.md](../README.md) — experiment naming and layout
- [docs/DATA_AND_TRAINING.md](../../docs/DATA_AND_TRAINING.md) — data paths, checkpoints, training safety
- [docs/PROJECT_ROADMAP_REIMPLEMENTATION.md](../../docs/PROJECT_ROADMAP_REIMPLEMENTATION.md) — spine and reimplementation phases
