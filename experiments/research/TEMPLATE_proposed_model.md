# <Model name> — Proposed (Research)

**Status:** Research / stub. Not active implementation.  
**Purpose:** <One sentence: what this model is and why we are proposing it.>  
**Created:** YYYY-MM-DD  
**Refs:** [CONTRACTS.md](../../docs/CONTRACTS.md), [ai_types_project_mapping.md](ai_types_project_mapping.md), [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md), <owning module path if any>

---

## 1. Scope and reasoning requirement

<One paragraph: what this model is, which AI type it instantiates (from [ai_types_project_mapping.md](ai_types_project_mapping.md)), and why we are proposing it. State the problem it solves and the reasoning — not just "we want it.">

---

## 2. What it can do

<Capabilities in plain language. Link to the relevant section of [ai_types_project_mapping.md](ai_types_project_mapping.md) for the AI type.>

---

## 3. Where in project (theoretical)

<Owner module/path (e.g. music_brain, penta_core, new module). Which contract or API it would satisfy. Reference [CONTRACTS.md](../../docs/CONTRACTS.md) or roadmap if relevant.>

---

## 4. Deployment placement

<Local (16 GB Mac, plugin/DAW) vs cloud (or self-hosted). Training path: local first, then cloud most effective. Reference [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md). Give reasoning for this model specifically.>

---

## 5. Summary table (supporting)

| Model name | AI type | Where in project | Local vs cloud | Training path | Status |
|------------|---------|------------------|----------------|---------------|--------|
| <Model name> | <AI type> | <owner path> | <local / cloud> | <local first, then cloud> | stub / future / experiment |

---

## 6. Implementation status

- This doc is **research only**. No code here.
- When implemented: update this doc; add a one-line note in CONTRACTS or the owning module.
- Experiments that test this model should reference this doc and state which assumptions are being tested.

---

## 7. References

- [ai_types_project_mapping.md](ai_types_project_mapping.md) — AI type and project-area mapping
- [local_vs_cloud_deployment.md](local_vs_cloud_deployment.md) — deployment and training path
- [CONTRACTS.md](../../docs/CONTRACTS.md) — spine contracts and APIs
- [DATA_AND_TRAINING.md](../../docs/DATA_AND_TRAINING.md) — data paths, checkpoints, training safety
- <Owning roadmap section if relevant>
