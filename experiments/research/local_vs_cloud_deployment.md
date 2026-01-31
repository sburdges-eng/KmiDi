# Local vs Cloud Deployment (Research)

**Status:** Research / strategy. Not active implementation.  
**Purpose:** Define which models run locally (low-latency, plugin/DAW, 16 GB macOS) vs cloud (or future self-hosted GPU), and how training fits: local first, then cloud using most effective method; cloud run financially sustainable.  
**Created:** 2026-01-31  
**Refs:** [ai_types_project_mapping.md](ai_types_project_mapping.md), [DATA_AND_TRAINING.md](../../docs/DATA_AND_TRAINING.md), [PROJECT_ROADMAP_REIMPLEMENTATION.md](../../docs/PROJECT_ROADMAP_REIMPLEMENTATION.md)

---

## 1. Reasoning requirement

This doc explains *why* we split workloads between local and cloud, *why* training follows “local first, then cloud,” and *why* cloud usage must be financially sustainable. Tables and checklists support the reasoning; they are not substitutes for it.

---

## 2. Why split local vs cloud

**Reasoning:** The project has two kinds of ML usage:

1. **Low-latency, in-the-loop:** Plugin/DAW features that must respond in milliseconds (e.g. “next bar,” “match intent,” groove/harmony suggestions, stem representation). These cannot depend on network round-trip or variable cloud latency; they must run on the user’s machine. Target hardware: 16 GB macOS (common for musicians and developers).

2. **Batch or non-real-time:** Intent parsing from long text, image generation, audio generation, large-model inference, and heavy training. These can tolerate seconds or minutes. Running them in the cloud (or future self-hosted GPU) avoids overloading the user’s machine and allows use of larger, more effective models. Cost must be sustainable so the product can scale.

**Conclusion:** Low-latency models run locally on 16 GB Mac; everything else can run in cloud (or future self-hosted GPU) with sustainable cost.

---

## 3. Local low-latency (plugin/DAW, 16 GB macOS)

### Scope

- **What runs locally:** Small, fast models used inside the DAW/plugin or Brain in real time: e.g. groove/harmony encoder–decoder, intent-embedding matcher, “next section” predictor (JEPA-style), stem representation (stem_jepa_integration), small AR or flow for in-distribution check. Anything that must complete in tens of milliseconds.
- **Constraint:** 16 GB RAM, CPU or Apple Silicon; no assumption of discrete GPU or large VRAM. Model size and inference path must fit this budget.

### Training path (local first, then cloud)

**Reasoning:** We train these models locally first because:

1. **Iteration speed:** Developers and researchers can iterate on architecture, data, and loss without cloud spend or queue time.
2. **Data locality:** Initial datasets may be small or sensitive; local training keeps data on-device until we have a stable pipeline.
3. **Validation:** We validate that the model fits 16 GB Mac latency and quality before investing in cloud scale-up.

**Then:** Once the pipeline is validated, we move training to cloud (or future self-hosted GPU) using the **most effective method** for that model type:

- **Effective** means: best quality/cost and reproducibility (e.g. same or better metric with documented config, seed, checkpoint path). For small models this might stay local; for larger ablations or full-scale data, cloud training is justified.
- **Method:** Could be same architecture with more data/steps, or distillation from a larger cloud-trained model back to a small local model. The research question is “what training setup gives the best local model,” not “train only locally forever.”

**Conclusion:** Low-latency models are **trained locally first**; **eventually** trained (or refined) in cloud using the most effective method; **inference** stays local on 16 GB Mac for plugin/DAW.

---

## 4. Cloud (or future self-hosted GPU)

### Scope

- **What runs in cloud:** LLM for intent parsing/generation (when model is too large for 16 GB Mac), image generation (e.g. SD), audio generation (e.g. Audiocraft-style), heavy training runs, large ablations, and any inference that is not in the critical low-latency path.
- **Future:** Self-hosted GPU (e.g. on-prem or owned hardware) can replace or supplement cloud for cost control and data control; the same “financially sustainable” rule applies (cap cost per user or per run).

### Financial sustainability

**Reasoning:** Cloud usage must be financially sustainable so that:

1. **Product viability:** Per-user or per-request cost stays within what the product can support (subscription, one-time, or hybrid).
2. **No surprise burn:** Usage is measurable and bounded (e.g. rate limits, quotas, or cost caps per user/month).
3. **Scaling:** As users grow, cost grows in a predictable way (e.g. linear or sub-linear with usage), not runaway.

**Concrete:** Before shipping any cloud-dependent feature:

- Define who pays (user vs product) and how (API key, quota, subscription tier).
- Prefer “small local model + optional cloud upgrade” over “everything in cloud” so 16 GB Mac users can still use core features offline.
- Prefer self-hosted GPU when it reduces long-term cost and when ops can support it; document this as a future option in CONTRACTS or ENV.

**Conclusion:** Cloud (and future self-hosted GPU) runs non–low-latency workloads; cost and usage are bounded and sustainable.

---

## 5. Summary table (supporting)

| Workload | Where runs | Training path | Reasoning |
|----------|------------|---------------|-----------|
| Low-latency (plugin/DAW): groove, intent match, “next section,” stem repr., small flow/AR | **Local** (16 GB Mac) | **Local first**; then cloud (or self-hosted) using most effective method | Latency and offline use require local inference; local training validates before scale-up. |
| LLM (intent parse/generate) when small | Local (16 GB Mac) | Local first; then cloud if scaling | Small LMs can run locally; see penta_core / llm_reasoning_engine. |
| LLM when large; image gen; audio gen; heavy training | **Cloud** (or future self-hosted GPU) | Cloud (or self-hosted) with effective method; optionally distill to local | Not in critical latency path; quality and cost justify cloud; sustainability required. |
| penta_core registered models (e.g. classifiers, embeddings) | Local or cloud per model | Per experiment; local first for small models | Depends on model size and contract. |

---

## 6. Checklist for new ML features (supporting)

When adding a new model or feature:

- [ ] **Latency:** Is it in the critical path for plugin/DAW (e.g. &lt;100 ms)? → If yes, design for local 16 GB Mac.
- [ ] **Training:** Plan “local first” for small/local models; document when and why to move training to cloud (or self-hosted) and which method is “most effective.”
- [ ] **Cloud:** If inference or training uses cloud, document cost model and sustainability (who pays, quotas, caps).
- [ ] **Self-hosted:** If we later support self-hosted GPU, document in CONTRACTS or ENV and keep API compatible so cloud and self-hosted are interchangeable.

---

## 7. References

- [ai_types_project_mapping.md](ai_types_project_mapping.md) — which AI types fit where; references this doc for deployment.
- [DATA_AND_TRAINING.md](../../docs/DATA_AND_TRAINING.md) — data paths, checkpoints, training safety, run manifest.
- [CONTRACTS.md](../../docs/CONTRACTS.md) — LLM→Intent, Intent→MIDI, image/audio contracts; add deployment notes when cloud/local contracts are fixed.
- [ENV_AND_TMUX.md](../../docs/ENV_AND_TMUX.md) — env and long-running jobs; document cloud/API keys and self-hosted URLs when used.
