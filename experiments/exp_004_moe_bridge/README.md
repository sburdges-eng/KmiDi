# exp_004: Frozen JEPA encoder + MoE bridge

**Status:** Design / stub. No implementation yet.

## Goal

Prototype a **frozen JEPA encoder** (e.g. WavJEPA) with an **MoE bridge** — role of the bridge (routing, capacity, conditioning, latent-space fusion) to be defined from primary source or design doc.

## Design

- **Docs:** [docs/WAVJEPA_LATENT_PIPELINE.md](../../docs/WAVJEPA_LATENT_PIPELINE.md)
- **Source plan:** [docs/SOURCE_INTEGRATION_PLAN.md](../../docs/SOURCE_INTEGRATION_PLAN.md) — briefing “Prototype a frozen JEPA encoder + MoE bridge”
- **Repo:** `music_brain/jepa/`; no MoE references in repo yet

## Constraints

- Encoder must stay **frozen** per existing design.
- Use only **verified, license-cleared** assets.
- MoE definition (router, experts, trainable vs fixed, input/output) depends on source or design decision.

## When to implement

After: (1) “MoE bridge” is defined (primary source or design doc), (2) encoder and any external code/weights are adopted in `config/source_manifest.yaml`.
